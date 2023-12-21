#include <torch/extension.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/util/MaybeOwned.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include "include/data_spec.hpp"
#include "include/data_spec_packed.cuh"
#include "include/utils_cuda.cuh"
#include "include/utils_grid.cuh"
#include "include/utils_math.cuh"

static constexpr uint32_t MAX_GRID_LEVELS = 8;

namespace {
namespace device {

inline __device__ float _calc_dt(
    const float t, const float cone_angle,
    const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}

__global__ void traverse_grids_kernel(
    // rays
    int32_t n_rays,
    float *rays_o,  // [n_rays, 3]
    float *rays_d,  // [n_rays, 3]
    // grids
    int32_t n_grids,
    int3 resolution,
    bool *binaries, // [n_grids, resx, resy, resz]
    float *aabbs,   // [n_grids, 6]
    // sorted intersections
    bool *hits,         // [n_rays, n_grids]
    float *t_sorted,    // [n_rays, n_grids * 2]
    int64_t *t_indices, // [n_rays, n_grids * 2]
    // options
    float *near_planes,
    float *far_planes,
    float step_size,
    float cone_angle,
    // outputs
    bool first_pass,
    PackedRaySegmentsSpec intervals,
    PackedRaySegmentsSpec samples)
{
    float eps = 1e-6f;

    // parallelize over rays
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_rays; tid += blockDim.x * gridDim.x)
    {
        // skip rays that are empty.
        if (intervals.chunk_cnts != nullptr)
            if (!first_pass && intervals.chunk_cnts[tid] == 0) continue;
        if (samples.chunk_cnts != nullptr)
            if (!first_pass && samples.chunk_cnts[tid] == 0) continue;

        int64_t chunk_start, chunk_start_bin;
        if (!first_pass) {
            if (intervals.chunk_cnts != nullptr)
                chunk_start = intervals.chunk_starts[tid];
            if (samples.chunk_cnts != nullptr)
                chunk_start_bin = samples.chunk_starts[tid];
        }
        float near_plane = near_planes[tid];
        float far_plane = far_planes[tid];

        SingleRaySpec ray = SingleRaySpec(
            rays_o + tid * 3, rays_d + tid * 3, near_plane, far_plane);

        int32_t base_hits = tid * n_grids;
        int32_t base_t_sorted = tid * n_grids * 2;

        // loop over all intersections along the ray.
        int64_t n_intervals = 0;
        int64_t n_samples = 0;
        float t_last = near_plane;
        bool continuous = false;
        for (int32_t i = base_t_sorted; i < base_t_sorted + n_grids * 2 - 1; i++) {
            // whether this is the entering or leaving for this level of grid.
            bool is_entering = t_indices[i] < n_grids;
            int64_t level = t_indices[i] % n_grids;
            // printf("i=%d, level=%lld, is_entering=%d, hits=%d\n", i, level, is_entering, hits[level]);

            if (!hits[base_hits + level]) {
                continue; // this grid is not hit.
            }
            if (!is_entering) {
                // we are leaving this grid. Are we inside the next grid?
                bool next_is_entering = t_indices[i + 1] < n_grids;
                if (next_is_entering) continue; // we are outside next grid.
                level = t_indices[i + 1] % n_grids;
                if (!hits[base_hits + level]) {
                    continue; // this grid is not hit.
                }
            }

            float this_tmin = fmaxf(t_sorted[i], near_plane);
            float this_tmax = fminf(t_sorted[i + 1], far_plane);   
            if (this_tmin >= this_tmax) continue; // this interval is invalid. e.g. (0.0f, 0.0f)
            // printf("i=%d, this_tmin=%f, this_tmax=%f, level=%lld\n", i, this_tmin, this_tmax, level);

            if (!continuous) {
                if (step_size <= 0.0f) { // march to this_tmin.
                    t_last = this_tmin;
                } else {
                    float dt = _calc_dt(t_last, cone_angle, step_size, 1e10f);
                    while (true) { // march until t_mid is right after this_tmin.
                        if (t_last + dt * 0.5f >= this_tmin) break;
                        t_last += dt;
                    }
                }
            }
            // printf(
            //     "[traverse segment] i=%d, this_mip=%d, this_tmin=%f, this_tmax=%f\n", 
            //     i, this_mip, this_tmin, this_tmax);

            AABBSpec aabb = AABBSpec(aabbs + level * 6);

            // init: pre-compute variables needed for traversal
            float3 tdist, delta;
            int3 step_index, current_index, final_index;
            setup_traversal(
                ray, this_tmin, this_tmax, eps,
                aabb, resolution,
                // outputs
                delta, tdist, step_index, current_index, final_index);
            // printf(
            //     "[traverse init], delta=(%f, %f, %f), step_index=(%d, %d, %d)\n",
            //     delta.x, delta.y, delta.z, step_index.x, step_index.y, step_index.z
            // );

            const int3 overflow_index = final_index + step_index;
            while (true) {
                float t_traverse = min(tdist.x, min(tdist.y, tdist.z));
                int64_t cell_id = (
                    current_index.x * resolution.y * resolution.z
                    + current_index.y * resolution.z
                    + current_index.z
                    + level * resolution.x * resolution.y * resolution.z
                );

                if (!binaries[cell_id]) {
                    // skip the cell that is empty.
                    if (step_size <= 0.0f) { // march to t_traverse.
                        t_last = t_traverse;
                    } else {
                        float dt = _calc_dt(t_last, cone_angle, step_size, 1e10f);
                        while (true) { // march until t_mid is right after t_traverse.
                            if (t_last + dt * 0.5f >= t_traverse) break;
                            t_last += dt;
                        }
                    }
                    continuous = false;
                } else {
                    // this cell is not empty, so we need to traverse it.
                    while (true) {
                        float t_next;
                        if (step_size <= 0.0f) {
                            t_next = t_traverse;
                        } else {  // march until t_mid is right after t_traverse.
                            float dt = _calc_dt(t_last, cone_angle, step_size, 1e10f);
                            if (t_last + dt * 0.5f >= t_traverse) break;
                            t_next = t_last + dt;
                        }

                        // writeout the interval.
                        if (intervals.chunk_cnts != nullptr) {
                            if (!continuous) {
                                if (!first_pass) {  // left side of the intervel
                                    int64_t idx = chunk_start + n_intervals;
                                    intervals.vals[idx] = t_last;
                                    intervals.ray_indices[idx] = tid;
                                    intervals.is_left[idx] = true;
                                }
                                n_intervals++;
                                if (!first_pass) {  // right side of the intervel
                                    int64_t idx = chunk_start + n_intervals;
                                    intervals.vals[idx] = t_next;
                                    intervals.ray_indices[idx] = tid;
                                    intervals.is_right[idx] = true;
                                }
                                n_intervals++;
                            } else {
                                if (!first_pass) {  // right side of the intervel
                                    int64_t idx = chunk_start + n_intervals;
                                    intervals.vals[idx] = t_next;
                                    intervals.ray_indices[idx] = tid;
                                    intervals.is_left[idx - 1] = true;
                                    intervals.is_right[idx] = true;
                                }
                                n_intervals++;
                            }
                        }

                        // writeout the sample.
                        if (samples.chunk_cnts != nullptr) {
                            if (!first_pass) {
                                int64_t idx = chunk_start_bin + n_samples;
                                samples.vals[idx] = (t_next + t_last) * 0.5f;
                                samples.ray_indices[idx] = tid;
                            }
                            n_samples++;
                        }

                        continuous = true;
                        t_last = t_next;
                        if (t_next >= t_traverse) break;
                    }
                }

                // printf(
                //     "[traverse], t_last=%f, t_traverse=%f, cell_id=%d, current_index=(%d, %d, %d)\n",
                //     t_last, t_traverse, cell_id, current_index.x, current_index.y, current_index.z
                // );

                if (!single_traversal(tdist, current_index, overflow_index, step_index, delta)) {
                    break;
                }
            }
        }
        
        if (first_pass) {
            if (intervals.chunk_cnts != nullptr)
                intervals.chunk_cnts[tid] = n_intervals;
            if (samples.chunk_cnts != nullptr)
                samples.chunk_cnts[tid] = n_samples;
        }
    }
}


__global__ void ray_aabb_intersect_kernel(
    const int32_t n_rays, float *rays_o, float *rays_d, float near, float far,
    const int32_t n_aabbs, float *aabbs,
    // outputs
    const float miss_value,
    float *t_mins, float *t_maxs, bool *hits)
{
    int32_t numel = n_rays * n_aabbs;
    // parallelize over rays
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id = tid / n_aabbs;
        int32_t aabb_id = tid % n_aabbs;

        float t_min, t_max;
        bool hit = device::ray_aabb_intersect(
            SingleRaySpec(rays_o + ray_id * 3, rays_d + ray_id * 3, near, far), 
            AABBSpec(aabbs + aabb_id * 6), 
            t_min, t_max
        );
        if (hit) {   
            t_mins[tid] = t_min;
            t_maxs[tid] = t_max;
        } else {
            t_mins[tid] = miss_value;
            t_maxs[tid] = miss_value;
        }
        hits[tid] = hit;
    }
}


}  // namespace device
}  // namespace


std::vector<RaySegmentsSpec> traverse_grids(
    // rays
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    // grids
    const torch::Tensor binaries,  // [n_grids, resx, resy, resz]
    const torch::Tensor aabbs,     // [n_grids, 6]
    // intersections
    const torch::Tensor t_mins,  // [n_rays, n_grids]
    const torch::Tensor t_maxs,  // [n_rays, n_grids]
    const torch::Tensor hits,    // [n_rays, n_grids]
    // options
    const torch::Tensor near_planes,
    const torch::Tensor far_planes,
    const float step_size,
    const float cone_angle,
    const bool compute_intervals,
    const bool compute_samples) 
{
    DEVICE_GUARD(rays_o);

    int32_t n_rays = rays_o.size(0);
    int32_t n_grids = binaries.size(0);
    int3 resolution = make_int3(binaries.size(1), binaries.size(2), binaries.size(3));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_threads = 512; 
    int32_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, n_rays));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(n_rays, threads.x)));

    // Sort the intersections. [n_rays, n_grids * 2]
    torch::Tensor t_sorted, t_indices;
    if (n_grids > 1) {
        std::tie(t_sorted, t_indices) = torch::sort(torch::cat({t_mins, t_maxs}, -1), -1);
    }
    else {
        t_sorted = torch::cat({t_mins, t_maxs}, -1);
        t_indices = torch::arange(
            0, n_grids * 2, t_mins.options().dtype(torch::kLong)
        ).expand({n_rays, n_grids * 2}).contiguous();
    }
    
    // outputs
    RaySegmentsSpec intervals, samples;

    // first pass to count the number of segments along each ray.
    if (compute_intervals)
        intervals.memalloc_cnts(n_rays, rays_o.options(), false);
    if (compute_samples)
        samples.memalloc_cnts(n_rays, rays_o.options(), false);
    device::traverse_grids_kernel<<<blocks, threads, 0, stream>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),  // [n_rays, 3]
        rays_d.data_ptr<float>(),  // [n_rays, 3]
        // grids
        n_grids,
        resolution,
        binaries.data_ptr<bool>(), // [n_grids, resx, resy, resz]
        aabbs.data_ptr<float>(),   // [n_grids, 6]
        // sorted intersections
        hits.data_ptr<bool>(),         // [n_rays, n_grids]
        t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
        t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
        // options
        near_planes.data_ptr<float>(), // [n_rays]
        far_planes.data_ptr<float>(),  // [n_rays]
        step_size,
        cone_angle,
        // outputs
        true,
        device::PackedRaySegmentsSpec(intervals),
        device::PackedRaySegmentsSpec(samples));
    
    // second pass to record the segments.
    if (compute_intervals)
        intervals.memalloc_data(true, true);
    if (compute_samples)
        samples.memalloc_data(false, false);
    device::traverse_grids_kernel<<<blocks, threads, 0, stream>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),  // [n_rays, 3]
        rays_d.data_ptr<float>(),  // [n_rays, 3]
        // grids
        n_grids,
        resolution,
        binaries.data_ptr<bool>(), // [n_grids, resx, resy, resz]
        aabbs.data_ptr<float>(),   // [n_grids, 6]
        // sorted intersections
        hits.data_ptr<bool>(),         // [n_rays, n_grids]
        t_sorted.data_ptr<float>(),    // [n_rays, n_grids * 2]
        t_indices.data_ptr<int64_t>(), // [n_rays, n_grids * 2]
        // options
        near_planes.data_ptr<float>(), // [n_rays]
        far_planes.data_ptr<float>(),  // [n_rays]
        step_size,
        cone_angle,
        // outputs
        false,
        device::PackedRaySegmentsSpec(intervals),
        device::PackedRaySegmentsSpec(samples));

    return {intervals, samples};
}


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor aabbs,  // [n_aabbs, 6]
    const float near_plane,
    const float far_plane, 
    const float miss_value)  
{
    DEVICE_GUARD(rays_o);

    int32_t n_rays = rays_o.size(0);
    int32_t n_aabbs = aabbs.size(0);
    int32_t numel = n_rays * n_aabbs;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int32_t max_threads = 512; 
    int32_t max_blocks = 65535;
    dim3 threads = dim3(min(max_threads, numel));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int32_t>(numel, threads.x)));

    // outputs
    torch::Tensor t_mins = torch::empty({n_rays, n_aabbs}, rays_o.options());
    torch::Tensor t_maxs = torch::empty({n_rays, n_aabbs}, rays_o.options());
    torch::Tensor hits = torch::empty({n_rays, n_aabbs}, rays_d.options().dtype(torch::kBool));

    device::ray_aabb_intersect_kernel<<<blocks, threads, 0, stream>>>(
        // rays
        n_rays,
        rays_o.data_ptr<float>(),  // [n_rays, 3]
        rays_d.data_ptr<float>(),  // [n_rays, 3]
        near_plane,
        far_plane,
        // aabbs
        n_aabbs,
        aabbs.data_ptr<float>(),   // [n_aabbs, 6]
        // outputs
        miss_value,
        t_mins.data_ptr<float>(),   // [n_rays, n_aabbs]
        t_maxs.data_ptr<float>(),   // [n_rays, n_aabbs]
        hits.data_ptr<bool>());     // [n_rays, n_aabbs]

    return {t_mins, t_maxs, hits};
}

"""Different model implementation plus a general port for all the models."""
import functools
from typing import Any, Callable, Optional, Tuple, Union
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import math
from jax import lax

from internal import mip
from internal import utils

def min_max_quantize(inputs, bits):
    if bits is 32:
        return inputs
    else:
        scale = jnp.amax(jnp.abs(inputs)).clip(a_min=1e-6)
        n = float(2**(bits-1) - 1)
        out = jnp.round(jnp.abs(inputs / scale) * n) / n * scale
        rounded = out * jnp.sign(inputs)
        return lax.stop_gradient(rounded-inputs) + inputs
    
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]
default_kernel_init = nn.initializers.lecun_normal()

class Dense_Q(nn.module.Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  dot_general: DotGeneralT = lax.dot_general
  bits: int = 32

  @nn.module.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param('kernel',
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,),
                        self.param_dtype)
    else:
      bias = None
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = jnp.asarray(kernel, self.dtype)
    bias = jnp.asarray(bias, self.dtype)
    
    kernel = min_max_quantize(kernel, self.bits)
    y = self.dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y

def cart2sph(cartesian_coords):
    x, y, z = cartesian_coords[:,0], cartesian_coords[:,1], cartesian_coords[:,2]
    azi = jnp.arctan2(y, x) / math.pi / 2 + 0.5
    ele = jnp.arctan2(z, jnp.sqrt(x*x + y*y)) * (-2.0) / math.pi
    spherical_coords = jnp.stack([azi, ele], axis=0)
    return spherical_coords

@gin.configurable
class MipNerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""
  num_samples: int = 128  # The number of samples per level.
  num_levels: int = 2  # The number of sampling levels.
  resample_padding: float = 0.01  # Dirichlet/alpha "padding" on the histogram.
  stop_level_grad: bool = True  # If True, don't backprop across levels')
  use_viewdirs: bool = True  # If True, use view directions as a condition.
  lindisp: bool = False  # If True, sample linearly in disparity, not in depth.
  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 16  # Max degree of positional encoding for 3D points.
  deg_view: int = 4  # Degree of positional encoding for viewdirs.
  density_activation: Callable[..., Any] = nn.softplus  # Density activation.
  density_noise: float = 0.  # Standard deviation of noise added to raw density.
  density_bias: float = -1.  # The shift added to raw densities pre-activation.
  rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  disable_integration: bool = False  # If True, use PE instead of IPE.

  @nn.compact
  def __call__(self, rng, rays, randomized, white_bkgd):
    """The mip-NeRF Model.

    Args:
      rng: jnp.ndarray, random number generator.
      rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
      randomized: bool, use randomized stratified sampling.
      white_bkgd: bool, if True, use white as the background (black o.w.).

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
    # Construct the MLP.
    mlp = MLP()

    ret = []
    for i_level in range(self.num_levels):
      key, rng = random.split(rng)
      if i_level == 0:
        # Stratified sampling along rays
        t_vals, samples = mip.sample_along_rays(
            key,
            rays.origins,
            rays.directions,
            rays.radii,
            self.num_samples,
            rays.near,
            rays.far,
            randomized,
            self.lindisp,
            self.ray_shape,
        )
      else:
        t_vals, samples = mip.resample_along_rays(
            key,
            rays.origins,
            rays.directions,
            rays.radii,
            t_vals,
            weights,
            randomized,
            self.ray_shape,
            self.stop_level_grad,
            resample_padding=self.resample_padding,
        )
      if self.disable_integration:
        samples = (samples[0], jnp.zeros_like(samples[1]))
      samples_enc = mip.integrated_pos_enc(
          samples,
          self.min_deg_point,
          self.max_deg_point,
      )

      # Point attribute predictions
      if self.use_viewdirs:
        viewdirs_enc = mip.pos_enc(
            rays.viewdirs,
            min_deg=0,
            max_deg=self.deg_view,
            append_identity=True,
        )
        raw_rgb, raw_density = mlp(samples_enc, cart2sph(rays.viewdirs), viewdirs_enc)
      else:
        raw_rgb, raw_density = mlp(samples_enc)

      # Add noise to regularize the density predictions if needed.
      if randomized and (self.density_noise > 0):
        key, rng = random.split(rng)
        raw_density += self.density_noise * random.normal(
            key, raw_density.shape, dtype=raw_density.dtype)

      # Volumetric rendering.
      rgb = self.rgb_activation(raw_rgb)
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
      density = self.density_activation(raw_density + self.density_bias)
      comp_rgb, distance, acc, weights = mip.volumetric_rendering(
          rgb,
          density,
          t_vals,
          rays.directions,
          white_bkgd=white_bkgd,
      )
      ret.append((comp_rgb, distance, acc))

    return ret


def construct_mipnerf(rng, example_batch):
  """Construct a Neural Radiance Field.

  Args:
    rng: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  model = MipNerfModel()
  key, rng = random.split(rng)
  init_variables = model.init(
      key,
      rng=rng,
      rays=utils.namedtuple_map(lambda x: x[0], example_batch['rays']),
      randomized=False,
      white_bkgd=False)
  return model, init_variables

class affine(nn.Module):
  res_azi: int = 10
  res_ele: int = 3
  norm: Callable[..., Any] = nn.GroupNorm(1, use_bias=False, use_scale=False)
  bits: int = 32
  @nn.compact
    x = self.norm(x)
    gamma = self.param('gamma', nn.initializers.ones, (self.res_azi, self.res_ele))
    beta = self.param('beta', nn.initializers.zeros, (self.res_azi, self.res_ele))
    gamma = min_max_quantize(gamma, self.bits)
    beta = min_max_quantize(beta, self.bits)
    coord = coord * jnp.array([[(self.res_azi-1)], [(self.res_ele-1)]], dtype=jnp.float32)
    x = jax.scipy.ndimage.map_coordinates(gamma, coord, 1, 'nearest').reshape(-1,1,1)*x + jax.scipy.ndimage.map_coordinates(beta, coord, 1, 'nearest').reshape(-1,1,1)
    return x

@gin.configurable
class MLP(nn.Module):
  """A simple MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  num_density_channels: int = 1  # The number of density channels.

  @nn.compact
  def __call__(self, x, view, condition=None):
    """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.
      condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
      raw_density: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_density_channels].
    """
    batch_size = x.shape[0]
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        Dense_Q, kernel_init=jax.nn.initializers.glorot_uniform())
    affine_layer = functools.partial(affine)

    inputs = x
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = x.reshape([batch_size, num_samples, -1])
      x = affine_layer()(x, view)
      x = x.reshape([batch_size*num_samples, -1])
      x = self.net_activation(x)
      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    raw_density = dense_layer(self.num_density_channels)(x).reshape(
        [-1, num_samples, self.num_density_channels])

    if condition is not None:
      # Output of the first part of MLP.
      bottleneck = dense_layer(self.net_width)(x)
      # Broadcast condition from [batch, feature] to
      # [batch, num_samples, feature] since all the samples along the same ray
      # have the same viewdir.
      condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_samples, feature] tensor to
      # [batch * num_samples, feature] so that it can be fed into nn.Dense.
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([bottleneck, condition], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for i in range(self.net_depth_condition):
        x = dense_layer(self.net_width_condition)(x)
        x = self.net_activation(x)
    raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
        [-1, num_samples, self.num_rgb_channels])
    return raw_rgb, raw_density


def render_image(render_fn, rays, rng, chunk=8192):
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function.
    rays: a `Rays` namedtuple, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    chunk: int, the size of chunks to render sequentially.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays[0].shape[:2]
  num_rays = height * width
  rays = utils.namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.host_id()
  results = []
  for i in range(0, num_rays, chunk):
    # pylint: disable=cell-var-from-loop
    chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
    chunk_size = chunk_rays[0].shape[0]
    rays_remaining = chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = utils.namedtuple_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # host_count.
    rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]),
                                      chunk_rays)
    chunk_results = render_fn(rng, chunk_rays)[-1]
    results.append([utils.unshard(x[0], padding) for x in chunk_results])
    # pylint: enable=cell-var-from-loop
  rgb, distance, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
  rgb = rgb.reshape((height, width, -1))
  distance = distance.reshape((height, width))
  acc = acc.reshape((height, width))
  return (rgb, distance, acc)

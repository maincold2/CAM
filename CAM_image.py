import os,sys
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import imageio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def img_mse(output, gt):
  return 0.5 * ((output - gt) ** 2).mean()

def img_psnr(mse):
  return -10.0 * np.log10(2.0 * mse)

class RandFourierFeature(nn.Module):
  def __init__(self, in_features, num_frequencies=256, sigma=10, scale=80, range=1.):
    super().__init__()
    self.in_features = in_features
    self.sigma = sigma
    self.scale = scale
    self.range = range # range of coordinate, e.g. range = 2 if input belongs to [-1,1], and 1 if [0,1].
    self.num_frequencies = num_frequencies
    self.out_features = self.num_frequencies * 2
    self.register_buffer('proj', torch.Tensor(in_features, num_frequencies))
    self.reset_parameters()
    print('random fourier feature params (sigma and scale): ',self.sigma,self.scale)

  def reset_parameters(self):
    with torch.no_grad():
      self.proj.copy_(torch.randn_like(self.proj) * (2*np.pi/self.range))

  def forward(self, coords):
    pos_enc = torch.mm(coords.flatten(start_dim=0, end_dim=1), self.proj*self.sigma)
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], axis=-1)
    output = pos_enc.view(coords.shape[0], coords.shape[1], self.out_features)
    if self.scale != -1:
      output *= self.scale / np.sqrt(self.num_frequencies) #the magnitude of fourier feature is np.sqrt(self.num_frequencies)
    return output

class FCLayer(nn.Module):
  def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.act = act

  def forward(self, input):
    output = self.linear(input)
    output = self.act(output)
    return output

class FCLayer_aff(nn.Module):
  def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
    super().__init__()
    self.w = 32
    self.h = 32
    self.linear = nn.Linear(in_features, out_features)
    self.affine = nn.Parameter(torch.cat([torch.ones(1,1,self.w, self.h),torch.zeros(1,1,self.w, self.h)], dim=1))
    self.act = act

  def forward(self, input, coord):
    output = self.linear(input)
    output = F.instance_norm(output)
    affine = nn.functional.grid_sample(self.affine, coord, padding_mode='border', align_corners=True).view(2,-1,1)
    output = output*affine[0]+affine[1]
    output = self.act(output)
    return output

class CAM(nn.Module):
  def __init__(self, in_features=2, out_features=1,hidden_features=256, num_layers=3, num_frequencies=256, sigma = 10, scale = 80):
    super().__init__()

    self.pos_enc = RandFourierFeature(in_features,num_frequencies = num_frequencies,sigma = sigma, scale=scale)
    self.num_layers = num_layers
    for i in range(self.num_layers):
      if i==0:
        in_channel = self.pos_enc.out_features
      else:
        in_channel = hidden_features
      setattr(self, f'FC_{i:d}', FCLayer_aff(in_channel, hidden_features, nn.ReLU(inplace=True)))
    self.FC_final = FCLayer(hidden_features, out_features, nn.Sigmoid())

  def forward(self, coords):
    output = self.pos_enc(coords)
    coords = (coords.view(1,-1,1,2)*2.0-1.0)
    for i in range(self.num_layers):
      fc = getattr(self, f'FC_{i:d}')
      output = fc(output, coords)
    output = self.FC_final(output)
    return output

class FFN(nn.Module):
  def __init__(self, in_features=2, out_features=1,hidden_features=256, num_layers=3, num_frequencies=256, sigma = 10, scale = -1):
    super().__init__()

    self.pos_enc = RandFourierFeature(in_features,num_frequencies = num_frequencies,sigma = sigma, scale=scale)
    self.num_layers = num_layers
    for i in range(self.num_layers):
      if i==0:
        in_channel = self.pos_enc.out_features
      else:
        in_channel = hidden_features
      setattr(self, f'FC_{i:d}', FCLayer(in_channel, hidden_features, nn.ReLU(inplace=True)))
    self.FC_final = FCLayer(hidden_features, out_features, nn.Sigmoid())

  def forward(self, coords):
    output = self.pos_enc(coords)
    for i in range(self.num_layers):
      fc = getattr(self, f'FC_{i:d}')
      output = fc(output)
    output = self.FC_final(output)
    return output


def get_mgrid(w,h, dim=2, offset=0.5):
  x = np.arange(0, w, dtype=np.float32)
  y = np.arange(0, h, dtype=np.float32)
  # size = max(w,h)
  # x = (x + offset) / size   # [0, size] -> [0, 1]
  # y = (y + offset) / size   # [0, size] -> [0, 1]
  x = (x + offset) / w   # [0, size] -> [0, 1]
  y = (y + offset) / h   # [0, size] -> [0, 1]
  X,Y = np.meshgrid(x,y, indexing='ij')
  output = np.stack([X,Y], -1)
  output = output.reshape(w*h, dim)
  return output

class ImageLoader(Dataset):
  def __init__(self, filename):
    img = np.asarray(imageio.imread(filename)).astype(np.float32)
    img = img / 255.0 # [0, 1]
    # img = (img-0.5)/0.5
    # img = img * (2.0 / 255.0) - 1.0 # [-1, 1]
    self.w = img.shape[0]
    self.h = img.shape[1]
    self.img = img.reshape(1, img.shape[0]*img.shape[1], -1)
    self.channel = img.shape[-1]
    print('image resolution:', self.w,'x',self.h)
    coords = get_mgrid(self.w, self.h, dim=2, offset=0.5) # [-1, 1]
    self.coords = np.expand_dims(coords, axis=0)

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    return torch.from_numpy(self.coords), torch.from_numpy(self.img)

def downsample(input, img_w, img_h, factor=2):
  if factor == 1: return input # directly return
  channel = input.size(-1)
  output = input.view(img_w, img_h, channel)
  output = output[::factor, ::factor, :].reshape(1, -1, channel)
  return output

def test_image(args):
  if not os.path.isfile(args.ckpt):
    print('error ckpt path')
    exit()

  if args.model == 'CAM':
    model = CAM(in_features=2,out_features=3, num_layers=args.num_layers, num_frequencies = args.num_frequencies, sigma = args.sigma, scale=-1)
  elif args.model == 'FFN':
    model = FFN(in_features=2,out_features=3, num_layers=args.num_layers, num_frequencies = args.num_frequencies, sigma = args.sigma, scale=-1)
  else:
    print('no active network, quit')
    exit()

  print(model)
  model.cuda()

  # load pretrained model if exist
  print('loading checkpoint %s' % args.ckpt)
  model.load_state_dict(torch.load(args.ckpt))

  coords = get_mgrid(args.res, args.res, dim=2, offset=0.5) # [-1, 1]
  coords = np.expand_dims(coords, axis=0)
  coords = torch.from_numpy(coords).cuda()

  model.eval()
  with torch.no_grad():
    img_pred_test = model(coords)

  print('save image to', args.test_file + '.png')
  img_pred_test = img_pred_test.clamp(0.0, 1.0)  # clip pixel velues to [0, 1]
  # img_pred_test = (img_pred_test*0.5+0.5).clamp(0.0, 1.0)  # clip pixel velues to [0, 1]
  img_pred_test = img_pred_test.view(args.res, args.res, -1).detach().cpu().numpy()
  img_pred_test = (img_pred_test * 255).astype(np.uint8)
  imageio.imwrite(args.test_file + '.png', img_pred_test)

def train_image(filepath,args):
  print('read image: ',filepath)
  image_data = ImageLoader(filepath)
  img_w, img_h = image_data.w, image_data.h
  data = image_data[0]
  coords, img_gt = data[0].cuda(), data[1].cuda()
  img_gt: object = img_gt[:,:,:3]

  coords_train, img_train = downsample(coords, img_w, img_h, factor=args.downsample), downsample(img_gt, img_w, img_h, factor=args.downsample)
  coords_test, img_test = downsample(coords, img_w, img_h, factor=1), downsample(img_gt, img_w, img_h, factor=1)

  #init network and optimizer
  if args.model == 'CAM':
    model = CAM(in_features=2,out_features=3, num_layers=args.num_layers, num_frequencies = args.num_frequencies,sigma = args.sigma, scale=-1)
  elif args.model == 'FFN':
    model = FFN(in_features=2,out_features=3, num_layers=args.num_layers, num_frequencies = args.num_frequencies,sigma = args.sigma, scale=-1)
  else:
    print('no active network, quit')
    exit()

  params, grid = [], []
  for name, p in model.named_parameters():
    if 'affine' in name: 
        grid.append(p)
    else:
        params.append(p)
  print(model)
  model.cuda()
  optim = torch.optim.Adam([{'params':params, 'lr':args.lr}, {'params':grid, 'lr':10.0*args.lr}])
  sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[1000, 1500], gamma=0.1)
  #
  filename = filepath.split('/')[-1]
  num_epochs = args.nr_epochs
  test_every_epoch = args.test_epochs
  logdir = os.path.join('logs/'+args.model+'/'+filepath.split('/')[-2],filename.split('.')[0], args.exp)
  writer = SummaryWriter(logdir)
  img_dir = os.path.join(logdir, 'img')
  ckpt_dir = os.path.join(logdir, 'checkpoints')
  if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
  if not os.path.exists(img_dir): os.makedirs(img_dir)
  img_name = os.path.join(img_dir,filename.split('.')[0])
  fid = open(os.path.join(logdir, 'summaries.csv'), 'w')
  tqdm.write("Epoch, Loss, Test PSNR, Train PSNR", fid)

  for epoch in tqdm(range(num_epochs+1), ncols=80):
    # train
    model.train()
    img_pred = model(coords_train)
    train_loss = img_mse(img_pred, img_train)

    # optimize
    optim.zero_grad()
    train_loss.backward()
    optim.step()
    sched.step()

    # test and write summaries
    if epoch % test_every_epoch == 0:
      model.eval()
      with torch.no_grad():
        img_pred_test = model(coords_test)

      img_pred_test = img_pred_test.clamp(0.0, 1.0) # clip pixel velues to [0, 1]
      # img_pred_test = (img_pred_test*0.5+0.5).clamp(0.0, 1.0) # clip pixel velues to [0, 1]
      test_loss = img_mse(img_pred_test, img_test).item()
      psnr = img_psnr(test_loss)
      psnr_train = img_psnr(train_loss.item())

      writer.add_scalar('train_loss', train_loss.item(), epoch)
      writer.add_scalar('total_loss', test_loss, epoch)
      writer.add_scalar('psnr', psnr, epoch)
        
      
      img_pred_test = img_pred_test.view(img_w, img_h, -1).detach().cpu().numpy()
      img_pred_test = (img_pred_test * 255).astype(np.uint8)
      writer.add_image('img', img_pred_test, global_step=epoch, dataformats='HWC')
      imageio.imwrite(img_name + '_%04d.png' % epoch, img_pred_test)
    
      gamma_grid = model.FC_2.affine[0,0].permute(1,0).view(32,32,1).detach().cpu().numpy()
      beta_grid = model.FC_2.affine[0,1].permute(1,0).view(32,32,1).detach().cpu().numpy()
      imageio.imwrite(img_name + '_%04d_gamma.png' % epoch, gamma_grid)
      imageio.imwrite(img_name + '_%04d_beta.png' % epoch, beta_grid)

      ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
      torch.save(model.state_dict(), ckpt_name)
      tqdm.write("%d, %0.6f, %0.6f, %0.6f" % (epoch, test_loss, psnr, psnr_train), fid)
      tqdm.write("Epoch %d, Test Loss %0.6f, Test PSNR %0.6f, Train PSNR %0.6f" % (epoch, test_loss, psnr, psnr_train))

  fid.close()
  
  return psnr

def train_dataset(args):
  dir = args.data
  files = os.listdir(dir)
  files.sort()
  print(len(files),'images for regression')
  psnrs = []
  for filename in files:
    filepath = os.path.join(dir, filename)
    psnrs.append(train_image(filepath,args))
  print(sum(psnrs)/len(psnrs))

class Config(object):

  def __init__(self,):

    parser, self.args = self.parse()
    if self.args.gpu_ids is not None:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_ids)

  def parse(self):
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('basic')
    group.add_argument('-g', '--gpu_ids', type=str, default=None, help='gpu to use, e.g. 0  0,1,2.')
    parser.add_argument('--model', type=str, default='CAM', choices=('CAM', 'FFN'),
                        help='Model to use [\'CAM, FFN\']')

    group = parser.add_argument_group('network')
    group.add_argument('--sigma', type=float, default=10.0)
    group.add_argument('--scale', type=float, default=-1)
    group.add_argument('--num_frequencies', type=int, default=256, help='number of frequencies')
    group.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
    group.add_argument('--downsample', type=int, default=2, help='downsample factor for training images')

    group = parser.add_argument_group('dataset')
    group.add_argument('--data', type=str, default='data', help='where data is')
    group.add_argument('--res', type=int, default=256, help='sdf volume resolution')

    group = parser.add_argument_group('training')
    group.add_argument('--nr_epochs', type=int, default=2000, help='total number of epochs to train')
    group.add_argument('--test_epochs', type=int, default=100, help='total number of epochs to train')
    group.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    group.add_argument('--ckpt', type=str, default='', help='pretrained model')

    group = parser.add_argument_group('testing')
    group.add_argument('--test_file', type=str, default='', help='save file')
    
    group.add_argument('--exp', type=str, default='exp', help='exp name')

    args = parser.parse_args()
    return parser, args

if __name__ == '__main__':
    args = Config().args
    if args.test_file:
      test_image(args)
    else:
      if os.path.isdir(args.data):
        train_dataset(args) #run all images in that folder
      elif os.path.isfile(args.data):
        train_image(args.data,args) #run one image
      else:
        print('error path')


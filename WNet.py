# -*- coding: utf-8 -*-
import astra
import matplotlib.pyplot as plt
from pytorch_ssim import ssim, SSIM
import numpy as np
import odl
from odl.contrib.torch import OperatorModule as OperatorAsModule
import torch
from torch import nn
from torch import optim
import tensorboardX
import util
from util import random_phantom

def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),   
       nn.ReLU(inplace=False),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(inplace=True)  )

# double convolution layer without non-linear activations
def double_conv_nr(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.Conv2d(out_channels, out_channels, 3, padding=1))


    
class UpLayer(nn.Module):
    """ Gradient and convolutions for WNet
    """
    def __init__(self,op, op_adj,fbp,int_size,proj_shape,eta): 
        super().__init__()
        
      
        self.dconvup = double_conv(3, 32)
       
        self.cconvup = nn.Conv2d(32,1,1)
        self.proj_shape = proj_shape
        
        # interior convolutions
        self.iconvup = double_conv_nr(1,32)
        self.icconv = nn.Conv2d(32,1,1)
        
        # operators
        self.op = op
        self.op_adj = op_adj
        self.fbp = fbp
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.eta = eta
        # activation
        self.act = nn.ReLU(inplace=True)
        
    def forward(self,x,y):
        # interpolate data to correct size
        y_i = nn.functional.interpolate(y,self.proj_shape)
        
        # compute gradient and filtered gradient
        normal = self.op(x) - y_i
        normal = self.iconvup(normal)
        normal = self.icconv(normal)
        grad = self.op_adj(normal)
        fb = self.fbp(normal)
        
        # concatenate and filter
        x1 = torch.cat([x,self.eta*grad,fb],dim=1)
        x1 = self.dconvup(x1)
        x1 = self.cconvup(x1)
        
        # take gradient step
        return self.act(x+ self.stepsize*x1)
    
def finalLayer():
    return nn.Sequential(
        nn.Conv2d(2,1,1),
        nn.ReLU(inplace=True))
    

def create_slices(levels=4,size=512):
    """Create slices to separate wavelet decomposition by scale"""
    slices = []
    shape = size//(2**(levels))
    f,l = shape**2,0
    slices.append(slice(l,f))
    for i in range(levels):
        shape = size//(2**(levels-i))
        l = f
        f = l + 3*shape**2
        slices.append(slice(l,f))
    return slices

def select_level(transform,level=4, max_level=4,size=512):
    """Select a single level of the wavelet transform"""
    initial_shape = (size//(2**(max_level)))**2
    if level == max_level:
        return transform
    back = 0
    for i in range(level):
        shape = size//(2**(i+1))
        back += 3*(shape**2)
    transform[:,:,initial_shape:-back] = 0
    if level ==0:
        transform[:,:,initial_shape:] = 0
    return transform

def wavelet_decomp(wv,levels=4,size=512):
    """Separate a vector of wavelet coefficients into a list
    of coefficients at different levels
    """
  x = []
  slices = create_slices(levels,size)
  
  t = wv[:,:,slices[0]]
  x.append(t)
  for i in range(levels):
    
    t = wv[:,:,slices[i+1]]
    x.append(t)
  return x


class basis_change(nn.Module):
  """Wavelet based upscaling through inverse wavelet transform"""  

  def __init__(self, size):#,wav,wav_inv):
    super().__init__()
    # wavelet operator must be defined for specific resolution
    self.space1 = odl.uniform_discr([-128, -128], [128, 128], [2*size, 2*size],dtype='float32')
    self.wav = OperatorAsModule(odl.trafos.WaveletTransform(self.space1,'haar',1))
    self.wav_inv = OperatorAsModule(odl.trafos.WaveletTransform(self.space1,'haar',1).inverse)
    self.size = size
    self.fact  = 2**(np.log2(512/self.size)-1) # scaling factor
    
    self.thresh = nn.Hardshrink(0.01)
    

  def forward(self, x,x1 = None):
    
    # reshape image into vector
    x = torch.reshape(x,(x.shape[0],1,self.size**2))
    if x1 is not None:
        # rescale and concatenate approximation and details
      x = torch.cat([2*x , self.thresh(x1)/self.fact],dim=2) 
      
    # single level wavelet inverse 
    x = self.wav_inv(x)
    return(x) 

class WNet(nn.Module):
  def __init__(self,ops,op_adjs,fbps,proj_shapes,wav,wav_inv,etas):
    super().__init__()
    self.initial_layer = conv_to_one(1)
    self.proj_shapes = proj_shapes
    self.base_change_0 = basis_change(32)
    self.base_change_1 = basis_change(64)
    self.base_change_2 = basis_change(128)
    self.base_change_3 = basis_change(256)
    
    self.layer_up_0 = UpLayer(ops[0], op_adjs[0],fbps[0],256,proj_shapes[0],etas[0])
    self.layer_up_1 = UpLayer(ops[1], op_adjs[1],fbps[1],256,proj_shapes[1],etas[1])
    self.layer_up_2 = UpLayer(ops[2], op_adjs[2],fbps[2],256,proj_shapes[2],etas[2])
    self.layer_up_3 = UpLayer(ops[3], op_adjs[3],fbps[3],256,proj_shapes[3],etas[3])
    self.layer_up_4 = UpLayer(ops[4], op_adjs[4],fbps[4],256,proj_shapes[4],etas[4]) 
    self.final_layer = nn.Conv2d(32,1,1)
    self.sfinal_layer = double_conv(1,32) 
    self.afinal_layer = nn.Conv2d(32,1,1)
    self.asfinal_layer = double_conv(1,32)
    self.relu = nn.ReLU()

  def forward(self,cur,y):
     """
     cur: wavelet decomposition of fbp using wavelet_decomp to split into scales
     y: data (sinogram)
     """ 
     
    # rescale first approximation to image
    x0 = self.relu(cur[0].reshape(cur[0].shape[0],1,32,32)/torch.tensor(2**3.5))
    
    x = self.layer_up_0(x0,y)
    x = self.base_change_0(x,cur[1])
    x = self.layer_up_1(x,y)
    x = self.base_change_1(x,cur[2])
    x = self.layer_up_2(x,y)
    x = self.base_change_2(x,cur[3])     
    x = self.layer_up_3(x,y)   
    x = self.base_change_3(x,cur[4])   
    x = self.layer_up_4(x,y)
    
    x = self.sfinal_layer(x)
    x = self.final_layer(x)
    return self.relu(x)

def operators(start_size,iter4Net,test_images):
    """Generates the operators at different scales for the network
    Inputs:
        start_size: length of image, n for n*n image
        iter4Net: number of layers in network
        test_image: test image for computation of etas
    Outputs:
        ops: list of radon foward transforms (pytorch)
        ops_adjs: list of radon adjoint transforms (pytorch)
        proj_shape: list of shapes for interpolation of data to correct scale
        fbp: list of fbp transforms (pytorch)
        etas: list of constants for scaling of fbp relative to bp
    """
    ops = []
    op_adjs = []
    fbps = []
    proj_shapes = []
    etas = []
    
    for i in range(iter4Net +1):
        # adjust angles
        n = start_size//2**(iter4Net-i)
        stride = start_size//n
        # generate odl operators
        spc = odl.uniform_discr([-128, -128], [128, 128], [n, n],dtype='float32')
        g = odl.tomo.cone_beam_geometry(spc, src_radius=500, det_radius=500, num_angles = nAngles//stride)
        rt = odl.tomo.RayTransform(spc, g, impl=impl)
        fbp_scaled = odl.tomo.fbp_op(rt,filter_type='Hann',frequency_scaling=0.6)
        # wrap to Pytorch
        ops.append(OperatorAsModule(rt).to(device))
        op_adjs.append(OperatorAsModule(rt.adjoint).to(device))
        fbps.append(OperatorAsModule(fbp_scaled).to(device))
        proj_shapes.append(g.partition.shape)
        # compute eta
        test_cur = nn.functional.interpolate(test_images, (n, n), mode='bilinear')
        print(test_cur)
        print(ops[i])
        normal = op_adjs[i](ops[i](test_cur))
        opnorm = torch.sqrt(torch.mean(normal ** 2)) / torch.sqrt(torch.mean(test_cur ** 2))
        etas.append(1 / opnorm)
    
    return ops,op_adjs,proj_shapes,fbps,etas



def summaries(writer, result, fbp, true, loss, it, do_print=False):
    """Save and print training and validation data to tensorboard"""
    residual = result - true
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    maxval = torch.max(true) - torch.min(true)
    psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
      
    relative = torch.mean((result - true) ** 2) / torch.mean((fbp - true) ** 2)
    ssi = ssim(result,true)
    ssi_fbp = ssim(fbp,true)
    relative_ssim = ssi/ssi_fbp
    if do_print:
        print(it, mse.item(), psnr.item(), relative.item(),ssi.item(),relative_ssim.item())

    writer.add_scalar('loss', loss, it)
    writer.add_scalar('psnr', psnr, it)
    writer.add_scalar('relative', relative, it)
    writer.add_scalar('ssim',ssi,it)
    writer.add_scalar('relative ssim',relative_ssim,it)

    util.summary_image(writer, 'result', result, it)
    util.summary_image(writer, 'true', true, it)


######################################################################
##Training
######################################################################

np.random.seed(42);
torch.manual_seed(42);


device = 'cuda' if astra.astra.use_cuda() else 'cpu'
impl = 'astra_cuda' if astra.astra.use_cuda() else 'astra_cpu'
learning_rate = 1e-3
log_interval = 20
iter4Net = 4
size = 512
val_size = 4
nIter = 10000
nAngles = 60

noiseLev = 0.01


n_data = 2

space = odl.uniform_discr([-128, -128], [128, 128], [size, size],dtype='float32')
geometry = odl.tomo.cone_beam_geometry(space, src_radius=500, det_radius=500, num_angles = nAngles)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl) 
fbp_op = odl.tomo.fbp_op(ray_trafo,filter_type='Hann',frequency_scaling=0.6)

    
def generate_data(validation=False, noiseLev = noiseLev):
    """Generate a set of random data."""
    n_generate = val_size if validation else n_data

    x_arr = np.empty((n_generate, 1, ray_trafo.range.shape[0], ray_trafo.range.shape[1]), dtype='float32')
    x_true_arr = np.empty((n_generate, 1, space.shape[0], space.shape[1]), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = (random_phantom(space,n_ellipse=30))
        data = ray_trafo(phantom)
        noisy_data = data + odl.phantom.white_noise(ray_trafo.range) * np.mean(np.abs(data)) * noiseLev
        

        x_arr[i, 0] = noisy_data
        x_true_arr[i, 0] = phantom

    return x_arr, x_true_arr
    

    
# Generate validation data
data, images = generate_data(validation=True)
test_images = torch.from_numpy(images).float().to(device)
fbp_op_mod = OperatorAsModule(fbp_op).to(device)
wav = odl.trafos.WaveletTransform(space,'haar',iter4Net)
wav_f = OperatorAsModule(wav).to(device)
wav_b = OperatorAsModule(wav.inverse).to(device)

test_data = torch.from_numpy(data).float().to(device)

test_fbp = fbp_op_mod(test_data)

x_test = wavelet_decomp(wav_f(test_fbp),iter4Net,512)
x_test.append(test_fbp)

train_writer = tensorboardX.SummaryWriter(comment="/train")
print(train_writer.logdir)
test_writer = tensorboardX.SummaryWriter(comment="/test")

mseloss=nn.MSELoss().to(device)
    




# generate operators
ops,op_adjs,proj_shapes,fbps,etas = operators(size,iter4Net,test_images)

# instatiate net
iter_net = WNet(ops,op_adjs,fbps,proj_shapes,wav_f,wav_b,etas).to(device)

optimizer = optim.Adam(iter_net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, nIter)




PATH = 'best_mse'
PATH2 = 'best_ssim'
train_best = np.inf
test_best = np.inf
ssim_best = -1
for it in range(nIter):
    #print(it)
    iter_net.train()
    data, images = generate_data()
    images = torch.from_numpy(images).float().to(device)
    projs = torch.from_numpy(data).float().to(device)
    xcur_fbp = fbp_op_mod(projs)
    # wavelet decomposition is outside net
    xcur = wavelet_decomp(wav_f(xcur_fbp),iter4Net,512)
    xcur.append(xcur_fbp)
    optimizer.zero_grad()
    output = iter_net(xcur,projs)

    loss = mseloss(output,images)

    loss.backward()
    #plot_grad_flow(iter_net.named_parameters())
    optimizer.step()
    scheduler.step()
    

    if it % 5 == 0: # it%25

        summaries(train_writer, output, xcur_fbp, images, loss, it, do_print=True)
        
        iter_net.eval()
        
        outputTest = iter_net(x_test,test_data)
        lossTest = mseloss(outputTest,test_images)
        #scheduler.step(lossTest)
        if loss < train_best+0.2 and lossTest < test_best:
            train_best = loss
            test_best = lossTest
            torch.save(iter_net.state_dict(), PATH)
            print('save')
        s = ssim(outputTest,test_images)
        if  s > ssim_best:
            torch.save(iter_net.state_dict(), PATH2)
            ssim_best = s
            print('save')
        summaries(test_writer, outputTest, test_fbp, test_images, lossTest, it, do_print=True)
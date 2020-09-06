# -*- coding: utf-8 -*-
import astra
import matplotlib.pyplot as plt
from pytorch_ssim import ssim

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
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),   
       nn.ReLU(inplace=True) )


class DownLayer(nn.Module):
    """Down Layer for U-Net 
        - Gradient computation
        - Convolution
        - Maxpool
    """
    def __init__(self, op, op_adj,fbp,in_size,out_size,num_grad): #fbp
        super().__init__()
        self.op = op
        self.op_adj = op_adj
        self.fbp = fbp
        self.num_grad = num_grad      
        self.dconv_down1 = double_conv(in_size+ 2*num_grad +2, out_size)
        self.maxpool = nn.MaxPool2d(2)

        
    def forward(self,x,y,xcur):
        """Takes 
        x - previous layer
        y - interpolated data
        xcur - fbp of interpolated data
        """
        normal2 = self.op(xcur) - y
        grad2 = self.op_adj(normal2)
        fb2 = self.fbp(normal2)
        x = torch.cat([x,grad2,fb2,grad2+xcur,fb2+xcur], dim=1)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        return x, conv1 
    
    
class UpLayer(nn.Module):
    """Up Layer for U-Net 
        - Concatenation
        - Gradient computation
        - Convolution
        - Interpolation
    """
    def __init__(self,op, op_adj,fbp,in_size,int_size,out_size,num_grad):
        super().__init__()
        self.uconvInv = nn.UpsamplingNearest2d(scale_factor=2) 
        self.dconvup = double_conv(in_size+ 4*num_grad +4, int_size)
        self.op = op
        self.op_adj = op_adj
        self.fbp = fbp
        self.num_grad = num_grad
        self.stepsize = nn.Parameter(torch.ones(1, 1, 1, 1))
    def forward(self,x,xcur,y,x1=None):
        """Takes 
        x - previous layer
        y - interpolated data
        xcur - fbp of interpolated data
        x1 - skip connection data from down path
        """
        if x1 is not None:
            # no concatenation on bottom layer
            x = torch.cat([x,x1],dim=1)
        # gradient computation
        normal = self.op(x[:,0:self.num_grad]) - y
        grad = self.op_adj(normal)
        fb = self.fbp(normal)
        normal2 = self.op(xcur) - y
        grad2 = self.op_adj(normal2)
        fb2 = self.fbp(normal2)
        # concatenate gradients together
        x = torch.cat([x,grad,fb,xcur+grad,xcur+fb,grad2,fb2,grad2+xcur,fb2+xcur], dim=1)
        # convolutions
        x = self.dconvup(x)
        # adjusted gradient step
        return self.uconvInv(xcur + self.stepsize*x)
    
def finalLayer():
    return nn.Sequential(
        nn.Conv2d(2,1,1),
        nn.ReLU(inplace=True))
    
class IterativeNetwork(nn.Module):
    """U- Net"""
    def __init__(self, ops, op_adjs, fbps, down_sizes,up_sizes ,proj_shapes,grads):#, init_op): #, fbps
        super().__init__()
        self.down_sizes = down_sizes
        self.num_layer_down = len(down_sizes)
        self.up_sizes = up_sizes
        self.num_layer_up = len(up_sizes)
        self.proj_shapes = proj_shapes
        self.op = ops
        self.op_adj = op_adjs
        self.fbp = fbps
        self.grads = grads
    

        # generate up and down layers based on size of net and number of convolutional
        # kernels generated outside the network
        for i, (op, op_adj,fbp,size,grad) in enumerate(zip(ops[:self.num_layer_down], op_adjs[:self.num_layer_down],fbps[:self.num_layer_down],down_sizes,grads[:self.num_layer_down])):   #,fbps)):
            
            layer = DownLayer(op=op, op_adj=op_adj,fbp=fbp,in_size=size[0],out_size=size[1],num_grad=grad)  #,fbp=fbp)
            setattr(self, 'layer_down_{}'.format(i), layer)
            
        for i, (op, op_adj,fbp,size,grad) in enumerate(zip(ops[-1:-self.num_layer_up-1:-1], op_adjs[-1:-self.num_layer_up-1:-1],fbps[-1:-self.num_layer_up-1:-1],self.up_sizes,grads[-1:-self.num_layer_up-1:-1])):
            layer = UpLayer(op,op_adj,fbp,size[0],size[1],size[2],grad)
            setattr(self,'layer_up_{}'.format(i),layer)
        
        # final convolutions
        self.semifinal_layer = double_conv(size[2]+self.down_sizes[0][1],1)
        self.final_layer = finalLayer()


    def forward(self, cur, y):
        """
        cur: FBP
        y: data (sinogram)
        """
        x = cur
        x_current = cur
        y_current = y
        xs = [0]*(self.num_layer_down+1)
        xs[0] = x
        xcurs = [0]*(self.num_layer_down+1)
        xcurs[0] = x_current
        
        # iterate over down layers
        for i in range(self.num_layer_down):
            # interpolate y to correct size
            proj_shape = self.proj_shapes[i]
            if i > 0:
                y_current = nn.functional.interpolate(y, proj_shape, mode='area')
                x_current = self.fbp[i](y_current)
                xcurs[i] = x_current
            # get layer and apply
            iteration = getattr(self, 'layer_down_{}'.format(i))
            x, xs[i+1] = iteration(x,y_current,x_current)


        # first up layer  - no concat
        proj_shape = self.proj_shapes[i+1]
        y_current = nn.functional.interpolate(y, proj_shape, mode='area')
        x_current = self.fbp[i+1](y_current)
        iteration = getattr(self, 'layer_up_{}'.format(0))
        x = iteration(x,x_current,y_current)

        
        # other up layers
        for i in range(1,self.num_layer_up):
            
            proj_shape = self.proj_shapes[-i-1]
            if i >0:
                y_current = nn.functional.interpolate(y, proj_shape, mode='area')
                x_current = xcurs[-(i+1)] 
            iteration = getattr(self, 'layer_up_{}'.format(i))
            x = iteration(x,x_current,y_current,xs[-(i)])

        
        # final 2 layers 
        x = self.semifinal_layer(torch.cat([xs[1],x],dim=1))
        
        x = self.final_layer(torch.cat([xs[0],x],dim=1)) # final layer combines intial estimate
        return x
    
def compute_sizes(start_size,num_layers,num_grads = None):
    """Function to compute the sizes which will be fed into NN
    Inputs:
        start_size: length of image, n for n*n image
        num_layers: number of layers in network
        num_grads: number of gradients to be computed at each layer
        
    Outputs:
        down_sizes: number of channels for input and output of each down layer
        up_sizes: number of input, intermediate and output channels for up layer
        num_grads: number of gradients to be computed at each layer
    """
    
    assert start_size%(2**num_layers) == 0
    
    if num_grads is None:
        # number of gradients doubles per scale (default)
        num_grads = np.power(2,range(num_layers))
    
    down_sizes,up_sizes = [0]*num_layers, [0]*num_layers
    down_sizes[0] = ( 1 , 32 ) # start 
    
    # down sizes double each layer
    for i in range(1,num_layers):
        down_sizes[i] = ( 32*(2**(i-1)) , 32*(2**(i)) )
    
    up_sizes[0] = (down_sizes[-1][1],down_sizes[-1][1]//2,down_sizes[-1][1]//2)
    
    # up sizes based on skip connection from downward path and size from 
    # previous layer on upward path
    for i in range(1,num_layers):
        up_sizes[i] = (down_sizes[-(i)][1]+up_sizes[i-1][2],down_sizes[-(i+1)][1]//2,down_sizes[-(i+1)][1]//2)
        
    return(down_sizes,up_sizes,num_grads)
    
def operators(start_size,iter4Net):
    """Generates the operators at different scales for the network
    Inputs:
        start_size: length of image, n for n*n image
        iter4Net: number of layers in network
        
    Outputs:
        ops: list of radon foward transforms (pytorch)
        ops_adjs: list of radon adjoint transforms (pytorch)
        proj_shape: list of shapes for interpolation of data to correct scale
        fbp: list of fbp transforms (pytorch)
    """
   
    ops = []
    op_adjs = []
    fbps = []
    
    proj_shapes = []
    
    for i in range(iter4Net +1):
        # adjust angles
        n = start_size//2**i
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
    
    return ops,op_adjs,proj_shapes,fbps



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
learning_rate = 1e-2
log_interval = 20
iter4Net = 4
size = 512
val_size = 1
nIter = 10000
nAngles = 60

noiseLev = 0.01


n_data = 4

space = odl.uniform_discr([-128, -128], [128, 128], [size, size],dtype='float32')
geometry = odl.tomo.cone_beam_geometry(space, src_radius=500, det_radius=500, num_angles = nAngles)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl) 
fbp_op = odl.tomo.fbp_op(ray_trafo,filter_type='Hann',frequency_scaling=0.6)

    
def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = val_size if validation else n_data

    x_arr = np.empty((n_generate, 1, ray_trafo.range.shape[0], ray_trafo.range.shape[1]), dtype='float32')
    x_true_arr = np.empty((n_generate, 1, space.shape[0], space.shape[1]), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = (random_phantom(space,n_ellipse=75))
        data = ray_trafo(phantom)
        noisy_data = data + odl.phantom.white_noise(ray_trafo.range) * np.mean(np.abs(data)) * noiseLev


        x_arr[i, 0] = noisy_data
        x_true_arr[i, 0] = phantom

    return x_arr, x_true_arr
    

    
# Generate validation data
data, images = generate_data(validation=True)
test_images = torch.from_numpy(images).float().to(device)
test_data = torch.from_numpy(data).float().to(device)


fbp_op_mod = OperatorAsModule(fbp_op).to(device)

test_fbp = fbp_op_mod(test_data)




train_writer = tensorboardX.SummaryWriter(comment="/train")
print(train_writer.logdir)
test_writer = tensorboardX.SummaryWriter(comment="/test")

mseloss=nn.MSELoss().to(device)
    


# compute sizes and operators
down_sizes, up_sizes, grads = compute_sizes(size, iter4Net)
ops,op_adjs,proj_shapes,fbps = operators(size,iter4Net)

# instatiate network
iter_net = IterativeNetwork(ops, op_adjs,fbps, down_sizes, up_sizes, proj_shapes, grads).to(device)

optimizer = optim.Adam(iter_net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, nIter)

# paths to save network parameters
PATH = 'best_mse'
PATH2 = 'best_ssim'
train_best = np.inf
test_best = np.inf
ssim_best = -1

for it in range(nIter):
    
    iter_net.train()
    data, images = generate_data()
    images = torch.from_numpy(images).float().to(device)
    projs = torch.from_numpy(data).float().to(device)
    xcur = fbp_op_mod(projs)
    
    
    optimizer.zero_grad()
    output = iter_net(xcur,projs)
    loss = mseloss(output,images)
    
    loss.backward()
    
    optimizer.step()
    scheduler.step()
    

    if it % 5 == 0: 

        summaries(train_writer, output, xcur, images, loss, it, do_print=True)
        
        iter_net.eval()
        x_test = fbp_op_mod(test_data)
        outputTest = iter_net(x_test,test_data)
        lossTest = mseloss(outputTest,test_images)
        
        # save best parameters
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
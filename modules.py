import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import optim
import numpy as np
import math

class ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation=None, use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2FC, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        if self.out_activation == 'tanh':
            x = torch.tanh(x)
        elif self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation=None):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x
    
class Bern(torch.autograd.Function):
    """
    Custom Bernouli function that supports gradients.
    The original Pytorch implementation of Bernouli function,
    does not support gradients.

    First-Order gradient of bernouli function with prbabilty p, is p.

    Inputs: Tensor of arbitrary shapes with bounded values in [0,1] interval
    Outputs: Randomly generated Tensor of only {0,1}, given Inputs as distributions.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):      
        pvals = ctx.saved_tensors
        return pvals[0] * grad_output
    
    
class MaskedLinear(nn.Module):
    """
    Which is a custom fully connected linear layer that its weights $W_f$ 
    remain constant once initialized randomly.
    A second weight matrix $W_m$ with the same shape as $W_f$ is used for
    generating a binary mask. This weight matrix can be trained through
    backpropagation. Each unit of $W_f$ may be passed through sigmoid
    function to generate the $p$ value of the $Bern(p)$ function.
    """
    def __init__(self, in_features, out_features, device):
        super(MaskedLinear, self).__init__()
        self.device = device

        # Fully Connected Weights
        self.fcw = torch.randn((out_features,in_features),requires_grad=False,device=device)
        # Weights of Mask
        self.mask = nn.Parameter(torch.randn_like(self.fcw,requires_grad=True,device=device))        

    def forward(self, x):        
        # Generate probability of bernouli distributions
        s_m = torch.sigmoid(self.mask)
        # Generate a binary mask based on the distributions
        g_m = Bern.apply(s_m)
        # Keep weights where mask is 1 and set others to 0
        effective_weight = self.fcw * g_m            
        # Apply the effective weight on the input data
#         print(x.shape, effective_weight.shape)
        lin = F.linear(x, effective_weight)

        return lin
        
    def __str__(self):        
        prod = torch.prod(*self.fcw.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.fcw.sum(),torch.abs(self.fcw).sum(),self.mask.sum() / prod)

    
class FC_supermask_encode(nn.Module):
    def __init__(self, device, x_dim=784, h_dim1=512, h_dim2=256,h_dim3=128,h_dim4=64):
        super(FC_supermask_encode, self).__init__()
        self.fc1 = MaskedLinear(x_dim, h_dim1, device)
        self.fc2 = MaskedLinear(h_dim1, h_dim2, device)
        self.fc3 = MaskedLinear(h_dim2, h_dim3, device)
        self.fc4 = MaskedLinear(h_dim3, h_dim4, device)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    
class FC_supermask_decode(nn.Module):
    def __init__(self, device, x_dim=784, h_dim1=512, h_dim2=256,h_dim3=128,h_dim4=64):
        super(FC_supermask_decode, self).__init__()
        self.fc4 = MaskedLinear(h_dim4, h_dim3, device)
        self.fc3 = MaskedLinear(h_dim3, h_dim2, device)
        self.fc2 = MaskedLinear(h_dim2, h_dim1, device)
        self.fc1 = MaskedLinear(h_dim1, x_dim, device)
    
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc1(h))
    
    

class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# class SupermaskConv(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # initialize the scores
#         self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
#         nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

#         # NOTE: initialize the weights like this.
#         nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

#         # NOTE: turn the gradient on the weights off
#         self.weight.requires_grad = False

#     def forward(self, x):
#         subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
#         w = self.weight * subnet
#         x = F.conv2d(
#             x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
#         )
#         return x

class MaskedLinear_nonstochastic(nn.Module):
    def __init__(self, in_features, out_features, device, sparsity, previous_model):
        super(MaskedLinear_nonstochastic, self).__init__()
        
        # initialize the scores'
        if previous_model is None:
            self.fcw = torch.randn((out_features,in_features),requires_grad=False,device=device)
            self.fcb = torch.randn((out_features),requires_grad=False,device=device)
            nn.init.kaiming_normal_(self.fcw, mode="fan_in", nonlinearity="relu")


        else:
            print('hi')
            for sub_network in previous_model.children():
                for layers in sub_network.children():
                    if layers.weight.shape == torch.Size([out_features, in_features]):
                        self.fcw = layers.weight
                        self.fcb = layers.bias


#             for i, j in previous_model.named_parameters():
#                 print(i, j.data.shape)
#                 if j.data.shape==torch.Size([out_features,in_features]):
#                     print('weight updated')
#                     self.fcw = j.data
#                 if j.data.shape ==torch.Size([out_features]):
#                     print('bias updated')
#                     self.fcb = j.data

        self.scores = nn.Parameter(torch.Tensor(self.fcw.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.

        # NOTE: turn the gradient on the weights off
        self.fcw.requires_grad = False
        self.fcb.requires_grad = False
        self.sparsity = sparsity
        self.mask = None
        self.device = device

    def forward(self, x):
        self.mask = GetSubnet.apply(self.scores.abs(), self.sparsity).to(self.device)
        w = self.fcw *  self.mask
        return F.linear(x, w, self.fcb)

# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
# class NonAffineBatchNorm(nn.BatchNorm2d):
#     def __init__(self, dim):
#         super(NonAffineBatchNorm, self).__init__(dim, affine=False)



class Supermask_ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation=None, use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2FC, self).__init__()
        self.conv1 = SupermaskConv(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = SupermaskConv(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = SupermaskConv(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = SupermaskConv(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = SupermaskConv(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = SupermaskConv(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        if self.out_activation == 'tanh':
            x = torch.tanh(x)
        elif self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class Supermask_DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation=None):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = SupermaskDeConv(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = SupermaskDeConv(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = SupermaskDeConv(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = SupermaskDeConv(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = SupermaskDeConv(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x
    


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskDeConv(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


    
class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
#         return x
        
        
class FC_supermask_encode_nonstochastic(nn.Module):
    def __init__(self, device, sparsity, x_dim=784, h_dim1=512, h_dim2=256,h_dim3=128,h_dim4=64, previous_model=None):
        super(FC_supermask_encode_nonstochastic, self).__init__()
        self.fc1 = MaskedLinear_nonstochastic(x_dim, h_dim1, device, sparsity, previous_model)
        self.fc2 = MaskedLinear_nonstochastic(h_dim1, h_dim2, device, sparsity, previous_model)
        self.fc3 = MaskedLinear_nonstochastic(h_dim2, h_dim3, device, sparsity, previous_model)
        self.fc4 = MaskedLinear_nonstochastic(h_dim3, h_dim4, device, sparsity, previous_model)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    
class FC_supermask_decode_nonstochastic(nn.Module):
    def __init__(self, device, sparsity, x_dim=784, h_dim1=512, h_dim2=256,h_dim3=128,h_dim4=64, previous_model=None):
        super(FC_supermask_decode_nonstochastic, self).__init__()
        self.fc4 = MaskedLinear_nonstochastic(h_dim4, h_dim3, device, sparsity, previous_model)
        self.fc3 = MaskedLinear_nonstochastic(h_dim3, h_dim2, device, sparsity, previous_model)
        self.fc2 = MaskedLinear_nonstochastic(h_dim2, h_dim1, device, sparsity, previous_model)
        self.fc1 = MaskedLinear_nonstochastic(h_dim1, x_dim, device, sparsity, previous_model)
    
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc1(h))
            
        
        
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = SupermaskConv(1, 32, 3, 1, bias=False)
#         self.conv2 = SupermaskConv(32, 64, 3, 1, bias=False)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = SupermaskLinear(9216, 128, bias=False)
#         self.fc2 = SupermaskLinear(128, 10, bias=False)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class FC_original_encode(nn.Module):
    def __init__(self, device, x_dim=784, h_dim1=512, h_dim2=256,h_dim3=128,h_dim4=64, bias = True, previous_model = None, mask=None):
        super(FC_original_encode, self).__init__()
        self.fc1 = Linear_modified(x_dim, h_dim1, bias,previous_model, mask)
        self.fc2 = Linear_modified(h_dim1, h_dim2, bias,previous_model, mask)
        self.fc3 = Linear_modified(h_dim2, h_dim3, bias,previous_model, mask)
        self.fc4 = Linear_modified(h_dim3, h_dim4, bias,previous_model, mask)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    
class FC_original_decode(nn.Module):
    def __init__(self, device, x_dim=784, h_dim1=512, h_dim2=256,h_dim3=128,h_dim4=64, bias = True, previous_model = None, mask=None):
        super(FC_original_decode, self).__init__()
        self.fc4 = Linear_modified(h_dim4, h_dim3, bias, previous_model, mask)
        self.fc3 = Linear_modified(h_dim3, h_dim2, bias, previous_model, mask)
        self.fc2 = Linear_modified(h_dim2, h_dim1, bias, previous_model, mask)
        self.fc1 = Linear_modified(h_dim1, x_dim, bias, previous_model, mask)
    
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc1(h))

class Linear_modified(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
#     __constants__ = ['in_features', 'out_features']
#     in_features: int
#     out_features: int
#     weight: Tensor

    def __init__(self, in_features, out_features, bias=True, previous_model=None, mask=None):
        super(Linear_modified, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        if previous_model is not None:
            print(self.out_features, self.in_features)
            self.receive_parameters(previous_model, mask)
        else:
            self.reset_parameters()
        

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def receive_parameters(self, previous_model, mask):
        for sub_network in previous_model.children():
            for layers in sub_network.children():
                try:
                    if layers.fcw.shape == torch.Size([self.out_features, self.in_features]):
                        self.weight.data = layers.fcw.data * layers.mask.data
                        if self.bias is not None:
                            self.bias.data = layers.fcb.data
                except:
                    if layers.weight.shape == torch.Size([self.out_features, self.in_features]):
                        for each_mask in mask:
                            if each_mask.shape == torch.Size([self.out_features, self.in_features]):
                                
                                self.weight.data = layers.weight.data * each_mask
                        if self.bias is not None:
                            self.bias.data = layers.bias.data
#                 if layers.fcb.shape == torch.Size([self.out_features]):
#                     print('bias is updated')
#                     self.bias.data = layers.fcb.data

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os
import torchvision
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import time
from tensorboardX import SummaryWriter
import argparse
import torch.autograd as autograd
import math



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3,h_dim4,sigmoid=False):
        super(AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.encoder = Encoder(x_dim, h_dim1, h_dim2,  h_dim3,h_dim4)
        # decoder part
        self.decoder = Generator(x_dim, h_dim1, h_dim2, h_dim3,h_dim4,sigmoid)
#         self.sparsity_ratios = nn.Parmaeter(torch.Tensor(8))
    
    def recon_error(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def representation(self,x):
        z = self.encoder(x)
        return z
    
    def intermediate(self,x,layer):
        shapes = [(16,32),(16,16),(8,16),(8,8)]
        if layer<4:
            inter= self.encoder.intermediate(x,layer)
            return inter, shapes[layer]
        else:
            inter = self.encoder(x)
            inter= self.decoder.intermediate(inter,layer-4)
            return inter, shapes[2-layer]

class Supermask_AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3,h_dim4,weights_to_prune,sparsity,sigmoid=False):
        super(Supermask_AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
            
        self.encoder = Supermask_Encoder(x_dim, h_dim1, h_dim2,  h_dim3,h_dim4, weights_to_prune,sparsity)
        
        # decoder part
        self.decoder = Supermask_Generator(x_dim, h_dim1, h_dim2, h_dim3,h_dim4, weights_to_prune,sparsity,sigmoid)
        
    
    def recon_error(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def representation(self,x):
        z = self.encoder(x)
        return z
    
    def intermediate(self,x,layer):
        shapes = [(16,32),(16,16),(8,16),(8,8)]
        if layer<4:
            inter= self.encoder.intermediate(x,layer)
            return inter, shapes[layer]
        else:
            inter = self.encoder(x)
            inter= self.decoder.intermediate(inter,layer-4)
            return inter, shapes[2-layer]
        
    def sparsity_per_layer(self):
        
        all_layers = [self.encoder.fc1,self.encoder.fc2,self.encoder.fc3,self.encoder.fc4,self.decoder.fc4,self.decoder.fc3,self.decoder.fc2,self.decoder.fc1]
        
        sparsity_per_layer=[]
        for layer in all_layers:
            if isinstance(layer, Supermask_Linear) or isinstance(layer, Supermask_SVS1_Linear) or isinstance(layer, Supermask_SVS2_Linear):
                sparsity_per_layer.append(layer.show_sparsity())
            else:
                sparsity_per_layer.append(0)
        return sparsity_per_layer
        
        
class Supermask_SVS1_AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3,h_dim4,weights_to_prune,sigmoid=False):
        super(Supermask_SVS1_AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
            
        self.encoder = Supermask_SVS1_Encoder(x_dim, h_dim1, h_dim2,  h_dim3,h_dim4, weights_to_prune)
        
        # decoder part
        self.decoder = Supermask_SVS1_Generator(x_dim, h_dim1, h_dim2, h_dim3,h_dim4, weights_to_prune,sigmoid)
        
    
    def recon_error(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def representation(self,x):
        z = self.encoder(x)
        return z
    
    def intermediate(self,x,layer):
        shapes = [(16,32),(16,16),(8,16),(8,8)]
        if layer<4:
            inter= self.encoder.intermediate(x,layer)
            return inter, shapes[layer]
        else:
            inter = self.encoder(x)
            inter= self.decoder.intermediate(inter,layer-4)
            return inter, shapes[2-layer]
        
    def sparsity_per_layer(self):
        
        all_layers = [self.encoder.fc1,self.encoder.fc2,self.encoder.fc3,self.encoder.fc4,self.decoder.fc4,self.decoder.fc3,self.decoder.fc2,self.decoder.fc1]
        
        sparsity_per_layer=[]
        for layer in all_layers:
            if isinstance(layer, Supermask_Linear) or isinstance(layer, Supermask_SVS1_Linear) or isinstance(layer, Supermask_SVS2_Linear):
                sparsity_per_layer.append(layer.show_sparsity())
            else:
                sparsity_per_layer.append(0)
        return sparsity_per_layer
    
class Supermask_SVS2_AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3,h_dim4,weights_to_prune,global_sparsity,sigmoid):
        super(Supermask_SVS2_AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        
        self.index=dict()
        for p, i0 in enumerate(weights_to_prune):
            self.index[i0] = p
                    
        self.global_sparsity = global_sparsity
        self.sparsity_ratios = nn.Parameter(torch.randn(len(weights_to_prune),requires_grad=True))
            
        self.encoder = Supermask_SVS2_Encoder(x_dim, h_dim1, h_dim2,  h_dim3,h_dim4, weights_to_prune,self.sparsity_ratios, self.global_sparsity, self.index)
        
        # decoder part
        self.decoder = Supermask_SVS2_Generator(x_dim, h_dim1, h_dim2, h_dim3,h_dim4, weights_to_prune,sigmoid, self.sparsity_ratios, self.global_sparsity, self.index)
        
    
    def recon_error(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def representation(self,x):
        z = self.encoder(x)
        return z
    
    def intermediate(self,x,layer):
        shapes = [(16,32),(16,16),(8,16),(8,8)]
        if layer<4:
            inter= self.encoder.intermediate(x,layer)
            return inter, shapes[layer]
        else:
            inter = self.encoder(x)
            inter= self.decoder.intermediate(inter,layer-4)
            return inter, shapes[2-layer]
        
    def sparsity_per_layer(self):
        
        all_layers = [self.encoder.fc1,self.encoder.fc2,self.encoder.fc3,self.encoder.fc4,self.decoder.fc4,self.decoder.fc3,self.decoder.fc2,self.decoder.fc1]
        
        sparsity_per_layer=[]
        for layer in all_layers:
            if isinstance(layer, SupermaskLinear):
                sparsity_per_layer.append(layer.show_sparsity())
            else:
                sparsity_per_layer.append(0)
        return sparsity_per_layer
    
    

    
########## Encoder and Decoders ##################
    
    
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc4 = nn.Linear(h_dim3,h_dim4)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc1(x))
        elif layer==1:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
        elif layer==2:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
        elif layer==3:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
            
        return h


    
class Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,sigmoid=False):
        super(Generator, self).__init__()
        self.fc4 = nn.Linear(h_dim4, h_dim3)
        self.fc3 = nn.Linear(h_dim3, h_dim2)
        self.fc2 = nn.Linear(h_dim2, h_dim1)
        self.fc1 = nn.Linear(h_dim1, x_dim)
        self.sigmoid = sigmoid
    
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        h = self.fc1(h)
        if self.sigmoid:
            h = torch.sigmoid(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc4(x))
        elif layer==1:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
        elif layer==2:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif layer==3:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
            h = self.fc1(h)
        return h

class Supermask_Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4, weights_to_prune, sparsity):
        super(Supermask_Encoder, self).__init__()
        if '1' in weights_to_prune:
            self.fc1 = Supermask_Linear(x_dim, h_dim1)
            self.fc1.sparsity = sparsity
        else:
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
        if '2' in weights_to_prune:
            self.fc2 = Supermask_Linear(h_dim1, h_dim2)
            self.fc2.sparsity = sparsity
        else:
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        if '3' in weights_to_prune:
            self.fc3 = Supermask_Linear(h_dim2, h_dim3)
            self.fc3.sparsity = sparsity
        else:
            self.fc3 = nn.Linear(h_dim2, h_dim3)
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
        if '4' in weights_to_prune:
            self.fc4 = Supermask_Linear(h_dim3,h_dim4)
            self.fc4.sparsity = sparsity
        else:
            self.fc4 = nn.Linear(h_dim3, h_dim4)
            self.fc4.weight.requires_grad = False
            self.fc4.bias.requires_grad = False    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc1(x))
        elif layer==1:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
        elif layer==2:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
        elif layer==3:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
            
        return h


    
class Supermask_Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,weights_to_prune,sparsity,sigmoid=False):
        super(Supermask_Generator, self).__init__()
        self.sigmoid = sigmoid
        if '5' in weights_to_prune:
            self.fc4 = Supermask_Linear(h_dim4,h_dim3)
            self.fc4.sparsity = sparsity
        else:
            self.fc4 = nn.Linear(h_dim4,h_dim3)
            self.fc4.weight.requires_grad = False
            self.fc4.bias.requires_grad = False
        if '6' in weights_to_prune:
            self.fc3 = Supermask_Linear(h_dim3, h_dim2)
            self.fc3.sparsity = sparsity
        else:
            self.fc3 = nn.Linear(h_dim3, h_dim2)
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
        if '7' in weights_to_prune:
            self.fc2 = Supermask_Linear(h_dim2, h_dim1)
            self.fc2.sparsity = sparsity
        else:
            self.fc2 = nn.Linear(h_dim2, h_dim1)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        if '8' in weights_to_prune:
            self.fc1 = Supermask_Linear(h_dim1, x_dim)
            self.fc1.sparsity = sparsity
        else:
            self.fc1 = nn.Linear(h_dim1, x_dim)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False

    
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        h = self.fc1(h)
        if self.sigmoid:
            h = torch.sigmoid(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc4(x))
        elif layer==1:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
        elif layer==2:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif layer==3:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
            h = self.fc1(h)
        return h
    
    
class Supermask_SVS1_Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,weights_to_prune):
        super(Supermask_SVS1_Encoder, self).__init__()
        if '1' in weights_to_prune:
            self.fc1 = Supermask_SVS1_Linear(x_dim, h_dim1)
        else:
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
        if '2' in weights_to_prune:
            self.fc2 = Supermask_SVS1_Linear(h_dim1, h_dim2)
        else:
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        if '3' in weights_to_prune:
            self.fc3 = Supermask_SVS1_Linear(h_dim2, h_dim3)
        else:
            self.fc3 = nn.Linear(h_dim2, h_dim3)
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
        if '4' in weights_to_prune:
            self.fc4 = Supermask_SVS1_Linear(h_dim3,h_dim4)
        else:
            self.fc4 = nn.Linear(h_dim3, h_dim4)
            self.fc4.weight.requires_grad = False
            self.fc4.bias.requires_grad = False
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc1(x))
        elif layer==1:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
        elif layer==2:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
        elif layer==3:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
            
        return h


    
class Supermask_SVS1_Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4, weights_to_prune,sigmoid):
        super(Supermask_SVS1_Generator, self).__init__()
        self.sigmoid = sigmoid
        if '5' in weights_to_prune:
            self.fc4 = Supermask_SVS1_Linear(h_dim4,h_dim3)
        else:
            self.fc4 = nn.Linear(h_dim4,h_dim3)
            self.fc4.weight.requires_grad = False
            self.fc4.bias.requires_grad = False
        if '6' in weights_to_prune:
            self.fc3 = Supermask_SVS1_Linear(h_dim3, h_dim2)
        else:
            self.fc3 = nn.Linear(h_dim3, h_dim2)
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
        if '7' in weights_to_prune:
            self.fc2 = Supermask_SVS1_Linear(h_dim2, h_dim1)
        else:
            self.fc2 = nn.Linear(h_dim2, h_dim1)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        if '8' in weights_to_prune:
            self.fc1 = Supermask_SVS1_Linear(h_dim1, x_dim)
        else:
            self.fc1 = nn.Linear(h_dim1, x_dim)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
            
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        h = self.fc1(h)
        if self.sigmoid:
            h = torch.sigmoid(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc4(x))
        elif layer==1:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
        elif layer==2:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif layer==3:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
            h = self.fc1(h)
        return h    
    

class Supermask_SVS2_Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,weights_to_prune,sparsity_ratios, global_sparsity,  index):
        super(Supermask_SVS2_Encoder, self).__init__()
        if '1' in weights_to_prune:
            self.fc1 = Supermask_SVS2_Linear(x_dim, h_dim1)
            self.fc1.sparsity('1',sparsity_ratios, global_sparsity, index)
        else:
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
        if '2' in weights_to_prune:
            self.fc2 = Supermask_SVS2_Linear(h_dim1, h_dim2)
            self.fc2.sparsity('2',sparsity_ratios, global_sparsity, index)
        else:
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        if '3' in weights_to_prune:
            self.fc3 = Supermask_SVS2_Linear(h_dim2, h_dim3)
            self.fc3.sparsity('3',sparsity_ratios, global_sparsity, index)
        else:
            self.fc3 = nn.Linear(h_dim2, h_dim3)
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
        if '4' in weights_to_prune:
            self.fc4 = Supermask_SVS2_Linear(h_dim3,h_dim4)
            self.fc4.sparsity('4',sparsity_ratios, global_sparsity, index)
        else:
            self.fc4 = nn.Linear(h_dim3, h_dim4)
            self.fc4.weight.requires_grad = False
            self.fc4.bias.requires_grad = False
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc1(x))
        elif layer==1:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
        elif layer==2:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
        elif layer==3:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
            
        return h


    
class Supermask_SVS2_Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4, weights_to_prune,sigmoid, sparsity_ratios, global_sparsity, index):
        super(Supermask_SVS2_Generator, self).__init__()
        self.sigmoid = sigmoid
        if '5' in weights_to_prune:
            self.fc4 = Supermask_SVS2_Linear(h_dim4,h_dim3)
            self.fc4.sparsity('5',sparsity_ratios, global_sparsity, index)
        else:
            self.fc4 = nn.Linear(h_dim4,h_dim3)
            self.fc4.weight.requires_grad = False
            self.fc4.bias.requires_grad = False
        if '6' in weights_to_prune:
            self.fc3 = Supermask_SVS2_Linear(h_dim3, h_dim2)
            self.fc3.sparsity('6',sparsity_ratios, global_sparsity, index)
        else:
            self.fc3 = nn.Linear(h_dim3, h_dim2)
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
        if '7' in weights_to_prune:
            self.fc2 = Supermask_SVS2_Linear(h_dim2, h_dim1)
            self.fc2.sparsity('7',sparsity_ratios, global_sparsity, index)
        else:
            self.fc2 = nn.Linear(h_dim2, h_dim1)
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
        if '8' in weights_to_prune:
            self.fc1 = Supermask_SVS2_Linear(h_dim1, x_dim)
            self.fc1.sparsity('8',sparsity_ratios, global_sparsity, index)
        else:
            self.fc1 = nn.Linear(h_dim1, x_dim)
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
            
    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc2(h))
        h = self.fc1(h)
        if self.sigmoid:
            h = torch.sigmoid(h)
        return h
    
    def intermediate(self,x,layer):
        if layer==0:
            h = F.relu(self.fc4(x))
        elif layer==1:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
        elif layer==2:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif layer==3:
            h = F.relu(self.fc4(x))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
            h = self.fc1(h)
        return h    
    
    
###### Extra Util Functions ###############
    
    
def reconstrucion_errors(model, loader):
    model.eval()
    model.cuda()
    for i, (data,label) in enumerate(loader):
        data = data.reshape(-1,784).cuda()
        with torch.no_grad():
            recon_error = model.recon_error(data)
            # representation = model.representation(data)

        if i==0:
            ood_recon = recon_error
            # ood_latents = representation

        else:
            ood_recon=torch.cat((ood_recon,recon_error),0)
            # ood_latents=torch.cat((ood_latents,representation),0)
            
    return ood_recon

def show_global_sparsity(model, verbose=True):
    sparsity = 100. * float(
            torch.sum(model.encoder.fc1.weight == 0)
            + torch.sum(model.encoder.fc2.weight == 0)
            + torch.sum(model.encoder.fc3.weight == 0)
            + torch.sum(model.encoder.fc4.weight == 0)
            
            + torch.sum(model.decoder.fc1.weight == 0)
            + torch.sum(model.decoder.fc2.weight == 0)
            + torch.sum(model.decoder.fc3.weight == 0)
            + torch.sum(model.decoder.fc4.weight == 0)
        )/ count_parameters(model)
    
    if verbose:
        print(
        "Global sparsity: {:.10f}%".format(sparsity)   )
            
    return sparsity

def calculate_auroc(ind_recon, ood_recon):          
    ind_size=ind_recon.cpu().numpy().shape[0]
    ood_size=ood_recon.cpu().numpy().shape[0]

    auroc = metrics.roc_auc_score(np.concatenate((np.zeros(ind_size),np.ones(ood_size))),np.concatenate((ind_recon.cpu().numpy(),ood_recon.cpu().numpy())))
            
    return auroc

def check_reconstructed_images(model, writer, index, percent, string, ind_loader, ood_loader,result_dir, model_name, sigmoid, run):
    test_data = torch.stack([ind_loader.dataset[i][0] for i in range(10)])
    recon = model(test_data.reshape(-1,784).cuda()).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    
    test_data2 = torch.stack([ood_loader.dataset[i][0] for i in range(10)])
    recon2 = model(test_data2.reshape(-1,784).cuda()).detach().cpu()
    recon2 = torch.clamp(recon2.view(len(recon), 1, 28, 28), 0, 1)


    x_and_recon = torch.cat([test_data, recon, test_data2,recon2])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('reconstructed_images_{}'.format(string), img_grid, index)
    plt.figure()
    plt.imshow(img_grid.cpu().numpy().transpose(1,2,0))
    plt.savefig(os.path.join(result_dir,'{}_{}_sigmoid_{}_sparsity_{:.3f}_run_{}_reconstructed.png'.format(string, model_name, sigmoid, percent, run)))
    plt.close()

def check_representations(model,loader):
    representations = []
    classes= []
    for (data,label) in loader:
        data = data.reshape(-1,784).cuda()
        representations.append(model.representation(data))
        classes.append(label)
        
    representations = torch.cat(representations)
    classes = torch.cat(classes)
    
    return representations, classes

class Supermask_Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.sparsity = None
#         print(self.var_sparsity.is_leaf)
#         self.var_sparsity = torch.sigmoid(self.var_sparsity)
#         self.var_sparsity.requires_grad = True
#         print(self.var_sparsity)
#         self.sparsity= None

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), 1-self.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
    
    def sparsity(self, sparsity):
        self.sparsity = sparsity
        
    def mask(self):
        subnet = GetSubnet.apply(self.scores.abs(), 1-self.sparsity)
        return subnet

    def show_sparsity(self):
#         print(self.var_sparsity)
#         self.var_sparsity.backward
#         print('***')
#         print(self.var_sparsity.grad)
        
        return self.sparsity


class Supermask_SVS1_Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.var_sparsity = nn.Parameter(torch.randn(1, requires_grad=True))
#         print(self.var_sparsity.is_leaf)
#         self.var_sparsity = torch.sigmoid(self.var_sparsity)
#         self.var_sparsity.requires_grad = True
#         print(self.var_sparsity)
#         self.sparsity= None

    def forward(self, x):
#         k = GetSigmoidSparsity.apply(self.var_sparsity)
        subnet = Get_SVS1_subnet.apply(self.scores.abs(), self.var_sparsity)
#         self.sigmoid_sparsity.backward()
#         print(self.sigmoid_sparsity.is_leaf)
#         print(self.sigmoid_sparsity.requires_grad)
#         print(self.sigmoid_sparsity.grad)
#         print(self.var_sparsity.grad)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
#         return x
    
    def sparsity(self, sparsity):
        sparsity = torch.Tensor([sparsity])
        self.var_sparsity.data = torch.log(sparsity/1-sparsity)
        self.var_sparsity.requires_grad = False
        
    def mask(self):
#         k = GetSigmoidSparsity.apply(self.var_sparsity)
        subnet = Get_SVS1_subnet.apply(self.scores.abs(), self.var_sparsity)
#         print('***')
#         print(self.sparsity)
#         print(subnet)
        return subnet
#         return x

    def show_sparsity(self):
#         print(self.var_sparsity)
#         self.var_sparsity.backward
#         print('***')
#         print(self.var_sparsity.grad)
        return torch.sigmoid(self.var_sparsity)




class Supermask_SVS2_Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.var_sparsity = None
        self.sparsity_ratios = None
        self.global_sparsity = None
        self.index = None
        self. i = None

    def forward(self, x):
        subnet = Get_SVS2_subnet.apply(self.scores.abs(), self.var_sparsity, self.global_sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
#         return x
    
    def sparsity_setting(self, i, sparsity_ratios, global_sparsity, index):
        self.i = i
        self.sparsity_ratios = sparsity_ratios
        self.global_sparsity = torch.Tensor(global_sparsity)
        self.index = index
        self.var_sparsity = sparsity_ratios[index[i]]
        
    def mask(self):
        subnet = Get_SVS2_subnet.apply(self.scores.abs(), self.var_sparsity, self.global_sparsity)
        return subnet

    def show_sparsity(self):
        return self.var_sparsity

class Get_SVS1_subnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, var_sparsity):
        out = scores.clone()
        k = 1- torch.sigmoid(var_sparsity)
        ctx.save_for_backward(var_sparsity)
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
#         print(out.shape)
        return out

    @staticmethod
    def backward(ctx, g):
#         print('subnet backward ')
#         print(g.shape)
        var_sparsity, = ctx.saved_tensors
        # send the gradient g straight-through on the backward pass.
        return g , g* (- torch.sigmoid(var_sparsity)* (1-torch.sigmoid(var_sparsity)))
    
    
    
class Get_SVS2_subnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, var_sparsity, global_sparsity):
        
        k = 1- torch.tanh(global_sparsity * var_sparsity**2)
        
        out = scores.clone()
#         k = 1- torch.sigmoid(var_sparsity)
        ctx.save_for_backward(var_sparsity, global_sparsity)
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
#         print(out.shape)
        return out

    @staticmethod
    def backward(ctx, g):
#         print('subnet backward ')
#         print(g.shape)
        var_sparsity,global_sparsity = ctx.saved_tensors
        # send the gradient g straight-through on the backward pass.
        return g , g* (- torch.sigmoid(var_sparsity)* (1-torch.sigmoid(var_sparsity)))
    
    
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

    
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


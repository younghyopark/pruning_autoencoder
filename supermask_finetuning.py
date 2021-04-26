import warnings
warnings.filterwarnings("ignore")

import argparse
import shutil
import os
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ae import AE, NAE
from leaveout_dataset import MNISTLeaveOut
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from utils import roc_btw_arr
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor
import math
import torch.autograd as autograd


from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                    help='Weight decay (default: 0.0005)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

class GetSubnet(autograd.Function):
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



class Supermask_ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation=None, use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(Supermask_ConvNet2FC, self).__init__()
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
        super(Supermask_DeConvNet2, self).__init__()
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
    
class Supermask_ConvNet2FC_finetune(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation=None, use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(Supermask_ConvNet2FC_finetune, self).__init__()
        self.conv1 = SupermaskConv_finetune(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = SupermaskConv_finetune(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = SupermaskConv_finetune(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = SupermaskConv_finetune(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = SupermaskConv_finetune(nh * 16, nh_mlp, kernel_size=4, bias=True)
        self.conv6 = SupermaskConv_finetune(nh_mlp, out_chan, kernel_size=1, bias=True)
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


class Supermask_DeConvNet2_finetune(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation=None):
        """nh: determines the numbers of conv filters"""
        super(Supermask_DeConvNet2_finetune, self).__init__()
        self.conv1 = SupermaskDeConv_finetune(in_chan, nh * 16, kernel_size=4, bias=True)
        self.conv2 = SupermaskDeConv_finetune(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = SupermaskDeConv_finetune(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = SupermaskDeConv_finetune(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = SupermaskDeConv_finetune(nh * 4, out_chan, kernel_size=3, bias=True)
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
        print('sparsity : {}'.format(sparsity))

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
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
        print('sparsity : {}'.format(sparsity))

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
        w = self.weight * subnet
#         x = F.conv_transpose2d(
#             x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
#         )
        
        x = F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding,
            self.padding, self.groups, self.dilation)
        return x

class SupermaskConv_finetune(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = True
        self.scores.requires_grad = False
        print('sparsity : {}'.format(sparsity))

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskDeConv_finetune(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = True
        self.scores.requires_grad = False
        print('sparsity : {}'.format(sparsity))

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
        w = self.weight * subnet
#         x = F.conv_transpose2d(
#             x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
#         )
        
        x = F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding,
            self.padding, self.groups, self.dilation)
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
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        return x
    
class SupermaskLinear_finetune(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = True
        self.score.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        return x
        

args = parser.parse_args()

result_dir = f'experiments/supermask_finetune/leaveout_{args.leave}'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    print(f'creating {result_dir}')
shutil.copy('supermask_based_pruning.py', f'{result_dir}/supermask_based_pruning.py')
print(f'file copied: ', f'{result_dir}/supermask_based_pruning.py')

device = args.device
gamma = 1.
l2_norm_reg = None
l2_norm_reg_en = None #0.0001 
spherical = True 
leave_out = args.leave
batch_size = 64


def predict(m, dl, device, flatten=False):
    l_result = []
    for x, _ in dl:
        with torch.no_grad():
            if flatten:
                x = x.view(len(x), -1)
            pred = m.predict(x.cuda(device)).detach().cpu()
        l_result.append(pred)
    return torch.cat(l_result)


'''load dataset'''
ds = MNISTLeaveOut('dataset', [leave_out], split='training', transform=ToTensor(), download=True)
in_train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10)
ds = MNISTLeaveOut('dataset', [leave_out], split='validation', transform=ToTensor(), download=True)
in_val_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10)
ds = MNISTLeaveOut('dataset', [leave_out], split='evaluation', transform=ToTensor(), download=True)
in_test_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10)

in_digits = list(set(list(range(10)))-set([leave_out]))
ds = MNISTLeaveOut('dataset', in_digits, split='validation', transform=ToTensor(), download=True)
out_val_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10)
ds = MNISTLeaveOut('dataset', in_digits, split='evaluation', transform=ToTensor(), download=True)
out_test_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10)


'''build model for RE loss weight training'''
writer = SummaryWriter(logdir=result_dir)

def train(model, device, optimizer, epoch, i):
    model.train()
    for x, _ in tqdm(in_train_dl):
        x=x.cuda(device)
            
        optimizer.zero_grad()
        z = model.encode(x)
        recon = model.decoder(z)
        error = (x - recon) ** 2
        z_norm = (z ** 2).mean()
        recon_error = error.mean()
        loss = recon_error

        decoder_norm = model.weight_norm(model.decoder)
        encoder_norm = model.weight_norm(model.encoder)

        loss.backward()

        optimizer.step()
        
        d_result = {'loss': loss.item(), 'z_norm': z_norm.item(), 'recon_error': recon_error.item(),
                    'decoder_norm': decoder_norm.item(), 'encoder_norm': encoder_norm.item()}
        
        if i%50==0:
        
            writer.add_scalar('supermask_finetune_{}/loss'.format(sparsity), d_result['loss'], i + 1)
            writer.add_scalar('supermask_finetune_{}/z_norm'.format(sparsity), d_result['z_norm'], i + 1)
            writer.add_scalar('supermask_finetune_{}/recon_error'.format(sparsity), d_result['recon_error'], i + 1)
            writer.add_scalar('supermask_finetune_{}/encoder_l2'.format(sparsity), d_result['encoder_norm'], i + 1)
            writer.add_scalar('supermask_finetune_{}/decoder_l2'.format(sparsity), d_result['decoder_norm'], i + 1)
            
        if i % 200 == 0:

            '''check inlier reconstruction'''
            test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
            recon = model.reconstruct(test_data.cuda(device)).detach().cpu()
            recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
            x_and_recon = torch.cat([test_data, recon])
            img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
            writer.add_image('supermask_finetune_{}/inlier_recon'.format(sparsity), img_grid, i+1)


            '''check outlier reconstruction'''
            test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
            recon = model.reconstruct(test_data.cuda(device)).detach().cpu()
            recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
            x_and_recon = torch.cat([test_data, recon])
            img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
            writer.add_image('supermask_finetune_{}/outier_recon'.format(sparsity), img_grid, i+1)


            '''val recon error'''
            val_err = predict(model, in_val_dl, device, flatten=False)

            in_pred = predict(model, in_test_dl, device, False)
            out_pred = predict(model, out_test_dl, device, False)
            auc = roc_btw_arr(out_pred, in_pred)
            writer.add_scalar('supermask_finetune_{}/auroc'.format(sparsity), auc, i + 1)
            writer.add_scalar('supermask_finetune_{}/val_error'.format(sparsity), val_err.mean().item(), i + 1)
        
        i +=1
        
    return d_result, i


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    global args
    # Training settings
    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} 
    
    z_dim = 17
    encoder = Supermask_ConvNet2FC_finetune(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')
    decoder = Supermask_DeConvNet2_finetune(z_dim, 1, nh=8, out_activation='sigmoid')

    model = NAE(encoder, decoder, l2_norm_reg=l2_norm_reg, l2_norm_reg_en=l2_norm_reg_en, spherical=spherical, z_step=10, z_stepsize=0.2, z_noise_std=0.05, x_step=50, x_stepsize=0.2, x_noise_std=0.05, x_noise_anneal=1., x_bound=(0, 1), z_bound=None, x_clip_langevin_grad=None)
    model.cuda(device);
    model.load_state_dict(torch.load('./experiments/supermask/leaveout_{}/supermask_{}.pkl'.format(args.leave, sparsity),"cuda:{}".format(device)))

    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    iteration=0
    for epoch in range(1, args.epochs + 1):
        print('epoch {} / {}'.format(epoch, args.epochs))
        _, i = train(model, device, optimizer, epoch, iteration)
        scheduler.step()
        iteration = i

    torch.save(model.state_dict(), f"{result_dir}/supermask_finetuned_{sparsity}.pkl")
    
            
    in_pred = predict(model, in_test_dl, device, False)
    out_pred = predict(model, out_test_dl, device, False)
    auc = roc_btw_arr(out_pred, in_pred)
    writer.add_scalar('supermask_finetune/AUROC', auc, index + 1)
    writer.add_scalar('supermask_finetune/IND_RE_mean', torch.mean(in_pred), index)
    writer.add_scalar('supermask_finetune/OOD_RE_mean', torch.mean(out_pred), index)
    writer.add_scalar('supermask_finetune/pruning_ratio',sparsity,index)

    '''check inlier reconstruction'''
    test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
    recon = model.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('supermask_finetune/inlier_recon', img_grid, index)


    '''check outlier reconstruction'''
    test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
    recon = model.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('supermask_finetune/outlier_recon', img_grid, index)
    
    

if __name__ == '__main__':
    
    xaxis_range = [0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99]

    for index, percent in enumerate(xaxis_range) :
        print(percent)

        sparsity = percent

        main()

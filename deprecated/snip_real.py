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
from ae import AE, NAE
from modules import DeConvNet2, ConvNet2FC, FC_supermask_encode, FC_supermask_decode, FC_supermask_encode_nonstochastic, FC_supermask_decode_nonstochastic, FC_original_encode, FC_original_decode
from leaveout_dataset import MNISTLeaveOut
from tqdm import tqdm
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from utils import roc_btw_arr
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor

from tensorboardX import SummaryWriter

from snip import SNIP



parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
# parser.add_argument('--run', type=str, help='experiment name', default='nae')
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
# parser.add_argument('-pr','--pruning_ratio', type=float, help ='if using non-stochastic model proposed by Ramanujan et al., specify the pruning ratio', default=None)


args = parser.parse_args()

result_dir = f'experiments/snip/leaveout_{args.leave}'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    print(f'creating {result_dir}')
shutil.copy('snip_real.py', f'{result_dir}/snip_real.py')
print(f'file copied: ', f'{result_dir}/snip_real.py')

device = args.device
batch_size = 128
leave_out = args.leave
gamma = 1.
spherical = True 

writer = SummaryWriter(logdir=result_dir)

def predict(m, dl, device, flatten=False):
    l_result = []
    for x, _ in dl:
        with torch.no_grad():
            if flatten:
                x = x.view(len(x), -1)
            pred = m.predict(x.cuda(device)).detach().cpu()
        l_result.append(pred)
    return torch.cat(l_result)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


LOG_INTERVAL = 20
INIT_LR = 0.1
WEIGHT_DECAY_RATE = 0.0005
EPOCHS = 250
REPEAT_WITH_DIFFERENT_SEED = 3


def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))

def train(index, pruning_ratio):
    
    '''build model for RE loss weight training'''
    z_dim = 17
    encoder = ConvNet2FC(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')
    decoder = DeConvNet2(z_dim, 1, nh=8, out_activation='sigmoid')
    net = NAE(encoder, decoder, l2_norm_reg=None, l2_norm_reg_en=None,
                    spherical=spherical, z_step=10, z_stepsize=0.2, z_noise_std=0.05,
                x_step=50, x_stepsize=0.2, x_noise_std=0.05, x_noise_anneal=1.,
                x_bound=(0, 1), z_bound=None, x_clip_langevin_grad=None)
    net.cuda(device);
    
#     optimizer = optim.SGD(
#         net.parameters(),
#         lr=INIT_LR,
#         momentum=0.9,
#         weight_decay=WEIGHT_DECAY_RATE)
    
#     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30000, gamma=0.1)

    optimizer = Adam(model.parameters(), lr=0.0001)
    
    # Pre-training pruning using SNIP
    keep_masks = SNIP(net,1-pruning_ratio, in_train_dl, device)

    apply_prune_mask(net, keep_masks)
    net.cuda(device)
    
    in_pred = predict(net, in_test_dl, device, False)
    out_pred = predict(net, out_test_dl, device, False)
    auc = roc_btw_arr(out_pred, in_pred)
    writer.add_scalar('SNIP_pruned/AUROC', auc, index + 1)
    writer.add_scalar('SNIP_pruned/IND_RE_mean', torch.mean(in_pred), index)
    writer.add_scalar('SNIP_pruned/OOD_RE_mean', torch.mean(out_pred), index)
    writer.add_scalar('SNIP_pruned/pruning_ratio',pruning_ratio,index)
 
    '''check inlier reconstruction'''
    test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
    recon = net.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('SNIP_pruned/inlier_recon', img_grid, index)


    '''check outlier reconstruction'''
    test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
    recon = net.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('SNIP_pruned/outlier_recon', img_grid, index)
    
    i = 0
    n_epoch = 250
    for i_epoch in tqdm(range(n_epoch)):
        for x, _ in tqdm(in_train_dl):
            
#             x = x.reshape(-1,784).cuda(device)
            x=x.cuda(device)
            
            optimizer.zero_grad()
            z = net.encode(x)
            recon = net.decoder(z)
            error = (x - recon) ** 2
            z_norm = (z ** 2).mean()
            recon_error = error.mean()
            loss = recon_error

            decoder_norm = net.weight_norm(net.decoder)
            encoder_norm = net.weight_norm(net.encoder)
        
            loss.backward()

            optimizer.step()
#             lr_scheduler.step()

            d_result = {'loss': loss.item(), 'z_norm': z_norm.item(), 'recon_error': recon_error.item(),
                    'decoder_norm': decoder_norm.item(), 'encoder_norm': encoder_norm.item()}
            
            if i%50==0:
        
                writer.add_scalar('SNIP_pruned_finetuning_{}/loss'.format(pruning_ratio), d_result['loss'], i + 1)
                writer.add_scalar('SNIP_pruned_finetuning_{}/z_norm'.format(pruning_ratio), d_result['z_norm'], i + 1)
                writer.add_scalar('SNIP_pruned_finetuning_{}/recon_error'.format(pruning_ratio), d_result['recon_error'], i + 1)
                writer.add_scalar('SNIP_pruned_finetuning_{}/encoder_l2'.format(pruning_ratio), d_result['encoder_norm'], i + 1)
                writer.add_scalar('SNIP_pruned_finetuning_{}/decoder_l2'.format(pruning_ratio), d_result['decoder_norm'], i + 1)

            if i % 200 == 0:

                '''check inlier reconstruction'''
                test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
                recon = net.reconstruct(test_data.cuda(device)).detach().cpu()
                recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
                x_and_recon = torch.cat([test_data, recon])
                img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
                writer.add_image('SNIP_pruned_finetuning_{}/inlier_recon'.format(pruning_ratio), img_grid, i+1)


                '''check outlier reconstruction'''
                test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
                recon = net.reconstruct(test_data.cuda(device)).detach().cpu()
                recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
                x_and_recon = torch.cat([test_data, recon])
                img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
                writer.add_image('SNIP_pruned_finetuning_{}/outier_recon'.format(pruning_ratio), img_grid, i+1)


                '''val recon error'''
                val_err = predict(net, in_val_dl, device, flatten=False)

                in_pred = predict(net, in_test_dl, device, False)
                out_pred = predict(net, out_test_dl, device, False)
                auc = roc_btw_arr(out_pred, in_pred)
                writer.add_scalar('SNIP_pruned_finetuning_{}/auroc'.format(pruning_ratio), auc, i + 1)
                writer.add_scalar('SNIP_pruned_finetuning_{}/val_error'.format(pruning_ratio), val_err.mean().item(), i + 1)

            i+=1
            
    in_pred = predict(net, in_test_dl, device, False)
    out_pred = predict(net, out_test_dl, device, False)
    auc = roc_btw_arr(out_pred, in_pred)
    writer.add_scalar('SNIP_pruned_finetuned/AUROC', auc, index + 1)
    writer.add_scalar('SNIP_pruned_finetuned/IND_RE_mean', torch.mean(in_pred), index)
    writer.add_scalar('SNIP_pruned_finetuned/OOD_RE_mean', torch.mean(out_pred), index)
    writer.add_scalar('SNIP_pruned_finetuned/pruning_ratio',pruning_ratio,index)
 
    '''check inlier reconstruction'''
    test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
    recon = net.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('SNIP_pruned_finetuned/inlier_recon', img_grid, index)


    '''check outlier reconstruction'''
    test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
    recon = net.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('SNIP_pruned_finetuned/outlier_recon', img_grid, index)


        
    # Let's look at the final weights
    # for name, param in net.named_parameters():
    #     if name.endswith('weight'):
    #         writer.add_histogram(name, param)

    writer.close()

    
if __name__ == '__main__':
    
    xaxis_range = [0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.97,0.99]

    for index, percent in enumerate(xaxis_range) :
        print(percent)
        train(index,percent)
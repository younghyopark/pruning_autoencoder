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
from modules import DeConvNet2, ConvNet2FC
from leaveout_dataset import MNISTLeaveOut
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from utils import roc_btw_arr
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor
import torch.nn.utils.prune as prune


from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
parser.add_argument('--run', type=str, help='experiment name', default='nae')
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('--pruning_ratio','-pr',type=float, default = None)
parser.add_argument('--pretrained', type=str, help ='load this pretrained weight', default=None)



args = parser.parse_args()
result_predir = 'experiments/weight_based_pruning'
result_dir = f'experiments/weight_based_pruning/leaveout_{args.leave}'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    print(f'creating {result_dir}')
shutil.copy('nae_experiment.py', f'{result_dir}/weight_based_pruning.py')
print(f'file copied: ', f'{result_dir}/weight_based_pruning.py')

device = args.device
finetune_epoch = 50
gamma = 1.
spherical = True 
leave_out = args.leave
batch_size = 128
pruning_ratio = args.pruning_ratio
l2_norm_reg = None
l2_norm_reg_en = None

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



'''build model'''

writer = SummaryWriter(logdir=result_dir)

finetune_epoch = 30

xaxis_range = [0,0.1,0.3,0.5,0.7,0.8,0.9,0.91,0.92,0.925,0.93,0.935,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.98,0.99]

for index, percent in enumerate(xaxis_range) :
    print(percent)
    z_dim = 17
    encoder = ConvNet2FC(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')
    decoder = DeConvNet2(z_dim, 1, nh=8, out_activation='sigmoid')
    model = NAE(encoder, decoder, l2_norm_reg=l2_norm_reg, l2_norm_reg_en=l2_norm_reg_en,
                    spherical=spherical, z_step=10, z_stepsize=0.2, z_noise_std=0.05,
                x_step=50, x_stepsize=0.2, x_noise_std=0.05, x_noise_anneal=1.,
                x_bound=(0, 1), z_bound=None, x_clip_langevin_grad=None)
    model.cuda(device);
    opt = Adam(model.parameters(), lr=0.0001)
    pruned_model = model
    model_path = './experiments/baseline/leaveout_{}/ae.pkl'.format(args.leave)
    pruned_model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda:{}'.format(args.device))))
    
    parameters_to_prune = (
        (pruned_model.encoder.conv1,'weight'),
        (pruned_model.encoder.conv2,'weight'),
        (pruned_model.encoder.conv3,'weight'),
        (pruned_model.encoder.conv4,'weight'),
        (pruned_model.encoder.conv5,'weight'),
        (pruned_model.encoder.conv6,'weight'),

        (pruned_model.decoder.conv1,'weight'),
        (pruned_model.decoder.conv2,'weight'),
        (pruned_model.decoder.conv3,'weight'),
        (pruned_model.decoder.conv4,'weight'),
        (pruned_model.decoder.conv5,'weight'),
    )
    
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method = prune.L1Unstructured,
    amount=percent)
    
    torch.save(pruned_model.state_dict(), f'{result_dir}/pruned_ae_pr_{percent}.pkl')
    
    pruned_model.eval()
    pruned_model.cuda(device)
    avg_loss = 0
    step = 0
    
    in_pred = predict(pruned_model, in_test_dl, device)
    out_pred = predict(pruned_model, out_test_dl, device)
    auc = roc_btw_arr(out_pred, in_pred)
    writer.add_scalar('pruned/AUROC', auc, index)
    writer.add_scalar('pruned/IND_RE_mean', torch.mean(in_pred), index)
    writer.add_scalar('pruned/OOD_RE_mean', torch.mean(out_pred), index)
    writer.add_scalar('pruned/pruning_ratio',percent,index)

    print('before finetuning : auroc = {}'.format(auc))
    
    
    '''check inlier reconstruction'''
    test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
    recon = pruned_model.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('pruned/inlier_recon', img_grid, index)


    '''check outlier reconstruction'''
    test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
    recon = pruned_model.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('pruned/outlier_recon', img_grid, index)
    

    i = 0
    for i_epoch in tqdm(range(finetune_epoch)):
        for x, _ in in_train_dl:
            x = x.cuda(device)
            d_result = pruned_model.train_step_ae(x, opt, clip_grad=None)

            writer.add_scalar('pruned_finetuning_{}/loss'.format(percent), d_result['loss'], i + 1)
            writer.add_scalar('pruned_finetuning_{}/z_norm'.format(percent), d_result['z_norm'], i + 1)
            writer.add_scalar('pruned_finetuning_{}/recon_error'.format(percent), d_result['recon_error'], i + 1)
            writer.add_scalar('pruned_finetuning_{}/encoder_l2'.format(percent), d_result['encoder_norm'], i + 1)
            writer.add_scalar('pruned_finetuning_{}/decoder_l2'.format(percent), d_result['decoder_norm'], i + 1)

            if i % 100 == 0:
                '''val recon error'''
                val_err = predict(pruned_model, in_val_dl, device)
                writer.add_scalar('pruned_finetuning_{}/val_recon'.format(percent), val_err.mean().item(), i + 1)
                
                in_pred = predict(pruned_model, in_test_dl, device)
                out_pred = predict(pruned_model, out_test_dl, device)
                auc = roc_btw_arr(out_pred, in_pred)
                writer.add_scalar('pruned_finetuning_{}/AUROC'.format(percent), auc, i+1)
                
                '''check inlier reconstruction'''
                test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
                recon = pruned_model.reconstruct(test_data.cuda(device)).detach().cpu()
                recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
                x_and_recon = torch.cat([test_data, recon])
                img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
                writer.add_image('pruned_finetuning_{}/inlier_recon'.format(percent), img_grid, i+1)


                '''check outlier reconstruction'''
                test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
                recon = pruned_model.reconstruct(test_data.cuda(device)).detach().cpu()
                recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
                x_and_recon = torch.cat([test_data, recon])
                img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
                writer.add_image('pruned_finetuning_{}/outlier_recon'.format(percent), img_grid, i+1)
  
            i += 1

    torch.save(pruned_model.state_dict(), f'{result_dir}/pruned_finetuned_ae_pr_{percent}.pkl')

    
    pruned_model.eval()
    pruned_model.cuda(device)
    avg_loss = 0
    step = 0
    
    in_pred = predict(pruned_model, in_test_dl, device)
    out_pred = predict(pruned_model, out_test_dl, device)
    auc = roc_btw_arr(out_pred, in_pred)
    writer.add_scalar('pruned_finetuned/AUROC', auc, index)

    writer.add_scalar('pruned_finetuned/IND_RE_mean', torch.mean(in_pred), index)
    writer.add_scalar('pruned_finetuned/OOD_RE_mean', torch.mean(out_pred), index)

    print('before finetuning : auroc = {}'.format(auc))
    
    
    '''check inlier reconstruction'''
    test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
    recon = pruned_model.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('pruned_finetuned/inlier_recon', img_grid, index)


    '''check outlier reconstruction'''
    test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
    recon = pruned_model.reconstruct(test_data.cuda(device)).detach().cpu()
    recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
    x_and_recon = torch.cat([test_data, recon])
    img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
    writer.add_image('pruned_finetuned/outlier_recon', img_grid, index)


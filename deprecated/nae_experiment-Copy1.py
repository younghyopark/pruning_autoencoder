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

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
parser.add_argument('--run', type=str, help='experiment name', default='nae')
args = parser.parse_args()

result_dir = f'modified_results/experiments/{args.run}'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
    print(f'creating {result_dir}')
shutil.copy('nae_experiment.py', f'{result_dir}/nae_experiment.py')
print(f'file copied: ', f'{result_dir}/nae_experiment.py')

device = args.device
n_ae_epoch = 100
n_nae_epoch = 50
gamma = 1.
l2_norm_reg = None
l2_norm_reg_en = 0.0001 
spherical = True 


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
ds = MNISTLeaveOut('dataset', [9], split='training', transform=ToTensor(), download=True)
in_train_dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)
ds = MNISTLeaveOut('dataset', [9], split='validation', transform=ToTensor(), download=True)
in_val_dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)
ds = MNISTLeaveOut('dataset', [9], split='evaluation', transform=ToTensor(), download=True)
in_test_dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)

in_digits = list(range(9))
ds = MNISTLeaveOut('dataset', in_digits, split='validation', transform=ToTensor(), download=True)
out_val_dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)
ds = MNISTLeaveOut('dataset', in_digits, split='evaluation', transform=ToTensor(), download=True)
out_test_dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=10)


'''build model'''
z_dim = 17
encoder = ConvNet2FC(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')
decoder = DeConvNet2(z_dim, 1, nh=8, out_activation='sigmoid')
model = NAE(encoder, decoder, l2_norm_reg=l2_norm_reg, l2_norm_reg_en=l2_norm_reg_en,
                spherical=spherical, z_step=10, z_stepsize=0.2, z_noise_std=0.05,
            x_step=50, x_stepsize=0.2, x_noise_std=0.05, x_noise_anneal=1.,
            x_bound=(0, 1), z_bound=None, x_clip_langevin_grad=None)
model.cuda(device);
opt = Adam(model.parameters(), lr=0.0001)
writer = SummaryWriter(logdir=result_dir)



'''AE PASS'''
print('starting autoencoder pre-training...')
n_epoch = n_ae_epoch; l_ae_result = []
i = 0
for i_epoch in tqdm(range(n_epoch)):
    for x, _ in in_train_dl:
        x = x.cuda(device)
        d_result = model.train_step_ae(x, opt, clip_grad=0.1)

        writer.add_scalar('ae/loss', d_result['loss'], i + 1)
        writer.add_scalar('ae/z_norm', d_result['z_norm'], i + 1)
        writer.add_scalar('ae/recon_error', d_result['recon_error'], i + 1)
        writer.add_scalar('ae/encoder_l2', d_result['encoder_norm'], i + 1)
        writer.add_scalar('ae/decoder_l2', d_result['decoder_norm'], i + 1)

        if i % 100 == 0:
            '''val recon error'''
            val_err = predict(model, in_val_dl, device)
            writer.add_scalar('ae/val_recon', val_err.mean().item(), i + 1)

        i += 1

torch.save(model.state_dict(), f'{result_dir}/ae.pkl')

'''check inlier reconstruction'''
test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
recon = model.reconstruct(test_data.cuda(device)).detach().cpu()
recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
x_and_recon = torch.cat([test_data, recon])
img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
writer.add_image('ae/inlier_recon', img_grid, 1)


'''check outlier reconstruction'''
test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
recon = model.reconstruct(test_data.cuda(device)).detach().cpu()
recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
x_and_recon = torch.cat([test_data, recon])
img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
writer.add_image('ae/outlier_recon', img_grid, 1)


'''vs9 AUC'''
in_pred = predict(model, in_test_dl, device)
out_pred = predict(model, out_test_dl, device)
auc = roc_btw_arr(out_pred, in_pred)
print(f'[Conventional Autoencoder][vs9 AUC]: {auc}')


'''NAE PASS'''
opt = Adam(model.parameters(), lr=0.00001)

print('starting NAE training...')
i = 0
n_epoch = n_nae_epoch; l_result = []
for i_epoch in tqdm(range(n_epoch)):
    for x, _ in tqdm(in_train_dl):
        x = x.cuda(device)
        d_result = model.train_step(x, opt, clip_grad=None)

        writer.add_scalar('nae/loss', d_result['loss'], i + 1)
        writer.add_scalar('nae/energy_diff', d_result['pos_e'] - d_result['neg_e'], i + 1)
        writer.add_scalar('nae/pos_e', d_result['pos_e'], i + 1)
        writer.add_scalar('nae/neg_e', d_result['neg_e'], i + 1)
        writer.add_scalar('nae/z_norm', d_result['z_norm'], i + 1)
        writer.add_scalar('nae/z_neg_norm', d_result['z_neg_norm'], i + 1)
        writer.add_scalar('nae/encoder_l2', d_result['encoder_norm'], i + 1)
        writer.add_scalar('nae/decoder_l2', d_result['decoder_norm'], i + 1)

        if i % 100 == 0:
            x_neg = d_result['x_neg']
            img_grid = make_grid(x_neg.detach().cpu(), nrow=10, range=(0, 1))
            writer.add_image('nae/sample', img_grid, i + 1)
            save_image(img_grid, f'{result_dir}/nae_sample_{i}.png')

            '''vs9 AUC'''
            in_pred = predict(model, in_test_dl, device)
            out_pred = predict(model, out_test_dl, device)
            auc = roc_btw_arr(out_pred, in_pred)
            writer.add_scalar('nae/vs9AUC', auc, i + 1)
            print(f'[Normalized AE][vs9 AUC]: {auc}')

        i += 1
    torch.save(model.state_dict(), f'{result_dir}/nae_{i_epoch}.pkl')
torch.save(model.state_dict(), f'{result_dir}/nae.pkl')

'''vs9 AUC'''
in_pred = predict(model, in_test_dl, device)
out_pred = predict(model, out_test_dl, device)
auc = roc_btw_arr(out_pred, in_pred)
print(f'[Normalized AE][vs9 AUC]: {auc}')



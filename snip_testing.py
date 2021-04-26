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

from sklearn.metrics import roc_auc_score
from utils import roc_btw_arr
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor

from tensorboardX import SummaryWriter

from snip import SNIP



parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
parser.add_argument('--run', type=str, help='experiment name', default='nae')
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('-pr','--pruning_ratio', type=float, help ='if using non-stochastic model proposed by Ramanujan et al., specify the pruning ratio', default=None)


args = parser.parse_args()

result_dir = f'snip_results/{args.run}/sparsitiy_{args.pruning_ratio}_leaveout_{args.leave}'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    print(f'creating {result_dir}')
shutil.copy('nae_experiment3_RE_supermask_freeze_RE_finetune.py', f'{result_dir}/nae_experiment3_RE_supermask_freeze_RE_finetune.py')
print(f'file copied: ', f'{result_dir}/nae_experiment3_RE_supermask_freeze_RE_finetune.py')

device = args.device
batch_size = 128
leave_out = args.leave


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


'''build model for RE loss weight training'''
encoder = FC_original_encode(device = device, bias = False) #ConvNet2FC(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')
decoder = FC_original_decode(device = device, bias = False) #DeConvNet2(z_dim, 1, nh=8, out_activation='sigmoid')
        
model = NAE(encoder, decoder)
model.cuda(device);

mask = SNIP(model,1-args.pruning_ratio, in_train_dl, device)

'''build model for finetuning after training'''
encoder = FC_original_encode(device = device, bias = False, previous_model = model, mask = mask) #ConvNet2FC(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')
decoder = FC_original_decode(device = device, bias = False, previous_model = model, mask = mask) #ConvNet2FC(1, z_dim, nh=8, nh_mlp=1024, out_activation='linear')

new_model = NAE(encoder, decoder)
new_model.cuda(device);

'''vs_whatever AUC'''
in_pred = predict(new_model, in_test_dl, device, True)
out_pred = predict(new_model, out_test_dl, device, True)
auc = roc_btw_arr(out_pred, in_pred)

print(auc)

torch.save(new_model.state_dict(), f'{result_dir}/snip_pruned.pkl')


'''AE Finetuning with frozen zero-weights'''
opt = Adam(new_model.parameters(), lr=0.001)
writer = SummaryWriter(logdir=result_dir)


print('starting Finetuning AE with frozen weights...')
finetune_epoch = 50

'''Find-out the zero weights'''
locked_masks = dict()
for each_mask in mask:
    locked_masks[each_mask.shape]=each_mask==0

i = 0
for i_epoch in tqdm(range(finetune_epoch)):
    for x, _ in tqdm(in_train_dl):
        x = x.reshape(-1,784).cuda(device)
        d_result = new_model.train_step_ae_finetune(x, opt, locked_masks, clip_grad=None)

        writer.add_scalar('step2/loss', d_result['loss'], i + 1)
        writer.add_scalar('step2/z_norm', d_result['z_norm'], i + 1)
        writer.add_scalar('step2/recon_error', d_result['recon_error'], i + 1)
        writer.add_scalar('step2/encoder_l2', d_result['encoder_norm'], i + 1)
        writer.add_scalar('step2/decoder_l2', d_result['decoder_norm'], i + 1)
        
        if i % 50 == 0:
            '''val recon error'''
            
            
            '''check inlier reconstruction'''
            test_data = torch.stack([in_test_dl.dataset[i][0] for i in range(10)])
            recon = model.reconstruct(test_data.reshape(-1,784).cuda(device)).detach().cpu()
            recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
            x_and_recon = torch.cat([test_data, recon])
            img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
            writer.add_image('step2/inlier_recon', img_grid, i+1)


            '''check outlier reconstruction'''
            test_data = torch.stack([out_test_dl.dataset[i][0] for i in range(10)])
            recon = model.reconstruct(test_data.reshape(-1,784).cuda(device)).detach().cpu()
            recon = torch.clamp(recon.view(len(recon), 1, 28, 28), 0, 1)
            x_and_recon = torch.cat([test_data, recon])
            img_grid = make_grid(x_and_recon.detach().cpu(), nrow=10, range=(0, 1))
            writer.add_image('step2/outlier_recon', img_grid, i+1)

            val_err = predict(new_model, in_val_dl, device, flatten=True)
            
            in_pred = predict(new_model, in_test_dl, device, True)
            out_pred = predict(new_model, out_test_dl, device, True)
            auc = roc_btw_arr(out_pred, in_pred)
            writer.add_scalar('step2/AUROC', auc, i + 1)
            writer.add_scalar('step2/val_recon', val_err.mean().item(), i + 1)
            
            sparsity = float(
            torch.sum(new_model.encoder.fc1.weight == 0)
            + torch.sum(new_model.encoder.fc2.weight == 0)
            + torch.sum(new_model.encoder.fc3.weight == 0)
            + torch.sum(new_model.encoder.fc4.weight == 0)
            
            + torch.sum(new_model.decoder.fc1.weight == 0)
            + torch.sum(new_model.decoder.fc2.weight == 0)
            + torch.sum(new_model.decoder.fc3.weight == 0)
            + torch.sum(new_model.decoder.fc4.weight == 0)

            )/ count_parameters(model)
            
            writer.add_scalar('step2/sparsity', sparsity, i + 1)
            
        i += 1

torch.save(new_model.state_dict(), f'{result_dir}/ae_finetune.pkl')


'''vs_whatever AUC'''
in_pred = predict(new_model, in_test_dl, device, True)
out_pred = predict(new_model, out_test_dl, device, True)
auc = roc_btw_arr(out_pred, in_pred)

print(auc)
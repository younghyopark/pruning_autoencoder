"""
ae.py
=====
Autoencoders
"""
import warnings
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.autograd as autograd
from energybased import SampleBuffer
from langevin import sample_langevin


class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err 

    def predict_and_reconstruct(self, x):
        recon = self(x)
        recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon

    def validation_step(self, x):
        recon = self(x)
        loss = torch.mean((recon - x) ** 2)
        predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon}

    def train_step(self, x, optimizer, clip_grad=None):
        optimizer.zero_grad()
        recon = self(x)
        if hasattr(self.decoder, 'square_error'):
            loss = torch.mean(self.decoder.square_error(x, recon))
        else:
            loss = torch.mean((recon - x) ** 2)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {'loss': loss.item()}

    def reconstruct(self, x):
        return self(x)

    def sample(self, N, z_shape=None, device='cpu'):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x


def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x


class NAE(AE):
    """sampling on z space"""
    def __init__(self, encoder, decoder,
                 z_step=50, z_stepsize=0.2, z_noise_std=0.2, z_noise_anneal=None,
                 x_step=50, x_stepsize=10, x_noise_std=0.05, x_noise_anneal=1,
                 x_bound=(0, 1), z_bound=None,
                 z_clip_langevin_grad=None, x_clip_langevin_grad=0.01, 
                 l2_norm_reg=None, l2_norm_reg_en=None, spherical=True,
                 buffer_size=10000, replay_ratio=0.95, replay=True,
                 gamma=1):
        super().__init__(encoder, decoder)
        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_clip_langevin_grad = x_clip_langevin_grad

        self.x_bound = x_bound
        self.z_bound = z_bound
        self.l2_norm_reg = l2_norm_reg  # decoder
        self.l2_norm_reg_en = l2_norm_reg_en
        self.spherical = spherical
        self.gamma = gamma

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        bound = 'spherical' if self.spherical else z_bound
        self.buffer = SampleBuffer(max_samples=buffer_size, replay_ratio=replay_ratio, bound=bound)

        self.z_shape = None

    def normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z

    def encode(self, x):
        return self.normalize(self.encoder(x))

    def sample(self, n_sample, device, replay=False):
        '''latent chain'''
        z0 = self.buffer.sample((n_sample,) + self.z_shape, device=device, replay=replay)
        energy = lambda z: self.predict(self.decoder(z))
        sample_z = sample_langevin(z0, energy, stepsize=self.z_stepsize, n_steps=self.z_step,
                                   noise_scale=self.z_noise_std,
                                   clip_x=None, clip_grad=self.z_clip_langevin_grad, intermediate_samples=False,
                                   spherical=self.spherical)
        sample_x_1 = self.decoder(sample_z).detach()
        self.buffer.push(sample_z)

        '''visible chain'''
        x_energy = lambda x: self.predict(x)
        sample_x_2 = sample_langevin(sample_x_1.detach(), x_energy, stepsize=self.x_stepsize, n_steps=self.x_step,
                                     noise_scale=self.x_noise_std,
                                     intermediate_samples=False,
                                     clip_x=self.x_bound, noise_anneal=self.x_noise_anneal,
                                     clip_grad=self.x_clip_langevin_grad, spherical=False)
        return {'sample_x': sample_x_2, 'sample_z': sample_z.detach(), 'sample_x0': sample_x_1} 

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        # self.register_buffer('z_shape', z_shape[1:])
        self.z_shape = z_shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param ** 2).sum()
        return norm

    def train_step_ae(self, x, opt, clip_grad=None):
        opt.zero_grad()
        z = self.encode(x)
        recon = self.decoder(z)
        error = (x - recon) ** 2
        z_norm = (z ** 2).mean()
        recon_error = error.mean()
        loss = recon_error

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm
        loss.backward()

        if clip_grad is not None:
            all_params = list(chain(*[g['params'] for g in opt.param_groups]))
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_grad)
        opt.step()
        d_result = {'loss': loss.item(), 'z_norm': z_norm.item(), 'recon_error': recon_error.item(),
                    'decoder_norm': decoder_norm.item(), 'encoder_norm': encoder_norm.item()}
        return d_result
    
    
    def train_step(self, x, opt, clip_grad=None):
        self._set_z_shape(x)

        # negative sample
        d_sample = self.sample(len(x), x.device, replay=self.replay)
        x_neg = d_sample['sample_x']

        opt.zero_grad()
        z_neg = self.encode(x_neg)
        recon_neg = self.decoder(z_neg)
        neg_e = (x_neg - recon_neg) ** 2

        # ae recon pass
        z = self.encode(x)
        recon = self.decoder(z)
        pos_e = (x - recon) ** 2

        loss = pos_e.mean() - neg_e.mean()

        if self.gamma is not None:
            loss += self.gamma * (neg_e ** 2).mean()

        # regularization
        z_norm = (z ** 2).mean()
        z_neg_norm = (z_neg ** 2).mean()

        # weight regularization
        decoder_norm = self.weight_norm(self.decoder)
        encoder_norm = self.weight_norm(self.encoder)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * decoder_norm
        if self.l2_norm_reg_en is not None:
            loss = loss + self.l2_norm_reg_en * encoder_norm

        loss.backward()

        if clip_grad is not None:
            all_params = list(chain(*[g['params'] for g in opt.param_groups]))
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_grad)
        opt.step()
        d_result = {'pos_e': pos_e.mean().item(), 'neg_e': neg_e.mean().item(),
                    'x_neg': x_neg.detach().cpu(), 'recon_neg': recon_neg.detach().cpu(),
                    'loss': loss.item(), 'sample': x_neg.detach().cpu(),
                    'z_norm': z_norm.item(), 'z_neg_norm': z_neg_norm.item(),
                    'decoder_norm': decoder_norm.item(), 'encoder_norm': encoder_norm.item()}
        return d_result


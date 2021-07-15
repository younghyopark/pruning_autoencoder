import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from langevin import sample_langevin

class SampleBuffer:
    def __init__(self, max_samples=10000, replay_ratio=0.95, bound=None):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio
        if bound is None:
            self.bound = (0, 1)
        else:
            self.bound = bound

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples

    def sample(self, shape, device, replay=False):
        if len(self.buffer) < 1 or not replay:  # empty buffer
            return self.random(shape, device)

        n_replay = (np.random.rand(shape[0]) < self.replay_ratio).sum()

        replay_sample = self.get(n_replay).to(device)
        n_random = shape[0] - n_replay
        if n_random > 0:
            random_sample = self.random((n_random,) + shape[1:], device)
            return torch.cat([replay_sample, random_sample])
        else:
            return replay_sample

    def random(self, shape, device):
        if self.bound is None:
            r = torch.rand(*shape, dtype=torch.float).to(device)

        elif self.bound == 'spherical':
            r = torch.randn(*shape, dtype=torch.float).to(device)
            norm = r.view(len(r), -1).norm(dim=-1)
            if len(shape) == 4:
                r = r / norm[:, None, None, None]
            elif len(shape) == 2:
                r = r / norm[:, None]
            else:
                raise NotImplementedError

        elif len(self.bound) == 2:
            r = torch.rand(*shape, dtype=torch.float).to(device)
            r = r * (self.bound[1] - self.bound[0]) + self.bound[0]
        return r

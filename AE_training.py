import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import calculate_log as callog
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms


# download path 정의
download_root = './dataset'

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",type=int, default=0, help="cuda index")
parser.add_argument("--max_epoch",type=int, default=100, help="warming epoch")
parser.add_argument("--save_dir",default='trained_models', help="warming epoch")
parser.add_argument("--lr",default=0.001, help="warming epoch")
parser.add_argument("--layer_num",type=int,default=784, help="warming epoch")
parser.add_argument("--h_dim1",type=int,default=512, help="warming epoch")
parser.add_argument("--h_dim2",type=int,default=256, help="warming epoch")
parser.add_argument("--h_dim3",type=int,default=64, help="warming epoch")
parser.add_argument("--h_dim4",type=int,default=16, help="warming epoch")
parser.add_argument("--h_dim5",type=int,default=0, help="warming epoch")
parser.add_argument("--h_dim6",type=int,default=0, help="warming epoch")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--one_class",type=int)

opt = parser.parse_args()
os.makedirs(os.path.join(opt.save_dir), exist_ok=True)

class AE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3,h_dim4,h_dim5,h_dim6):
        super(AE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.encoder = Encoder(x_dim, h_dim1, h_dim2,  h_dim3,h_dim4,h_dim5, h_dim6)
        # decoder part
        self.decoder = Generator(x_dim, h_dim1, h_dim2, h_dim3,h_dim4,h_dim5,h_dim6)
    
    def recon_error(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return torch.norm((x_recon - x), dim=1)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    
class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,h_dim5,h_dim6):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        if h_dim3>0:
            self.fc3 = nn.Linear(h_dim2, h_dim3)
        if h_dim4>0:
            self.fc4 = nn.Linear(h_dim3,h_dim4)
        if h_dim5 >0:
            self.fc5 = nn.Linear(h_dim4,h_dim5)
        if h_dim6 >0:
            self.fc6 = nn.Linear(h_dim5,h_dim6)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        if opt.h_dim6 >0:
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc5(h))
            h = self.fc6(h)
        elif opt.h_dim5 >0:
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc4(h))
            h = self.fc5(h)
        elif opt.h_dim4 >0:
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            h = self.fc4(h)
        elif opt.h_dim3 >0:
            h = F.relu(self.fc2(h))
            h = self.fc3(h)
        else:
            h = self.fc2(h)
        return h
    
    
class Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,h_dim5, h_dim6):
        super(Generator, self).__init__()
        if h_dim6 >0:
            self.fc6 = nn.Linear(h_dim6,h_dim5)
        if h_dim5 >0:
            self.fc5 = nn.Linear(h_dim5,h_dim4)
        if h_dim4 >0:
            self.fc4 = nn.Linear(h_dim4,h_dim3)
        if h_dim3 >0:
            self.fc3 = nn.Linear(h_dim3, h_dim2)
        self.fc2 = nn.Linear(h_dim2, h_dim1)
        self.fc1 = nn.Linear(h_dim1, x_dim)
    
    def forward(self, z):
        if opt.h_dim6 >0:
            h = F.relu(self.fc6(z))
            h = F.relu(self.fc5(h))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif opt.h_dim5 >0:
            h = F.relu(self.fc5(z))
            h = F.relu(self.fc4(h))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif opt.h_dim4>0:
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        elif opt.h_dim3>0:
            h = F.relu(self.fc3(h))
            h = F.relu(self.fc2(h))
        else:
            h = F.relu(self.fc2(z))
        h = self.fc1(h)
#         h = torch.sigmoid(h)
        return h


model = AE(opt.layer_num, opt.h_dim1, opt.h_dim2, opt.h_dim3,opt.h_dim4,opt.h_dim5,opt.h_dim6)
optimizer = torch.optim.Adam(model.parameters(), opt.lr)
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epoch, eta_min=0, last_epoch=-1)

mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
#     transforms.Normalize((0.5,), (1.0,))
])

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
# valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
# test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

idx = train_dataset.targets!=opt.one_class
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                         batch_size=opt.batch_size,
                         shuffle=True)

# valid_loader = DataLoader(dataset=test_dataset, 
#                          batch_size=opt.batch_size,
#                          shuffle=True)

# test_loader = DataLoader(dataset=test_dataset, 
#                          batch_size=opt.batch_size,
#                          shuffle=True)

device = torch.device('cuda')
torch.cuda.set_device(opt.gpu)

model_name = '{}_{}_{}_{}_{}_{}'.format(opt.h_dim1, opt.h_dim2, opt.h_dim3, opt.h_dim4, opt.h_dim5, opt.h_dim6)
model.to(device)
print(model)

model.train()
for epoch in range(1, opt.max_epoch+ 1):
    avg_loss = 0
    step = 0
    for i, (data,label) in enumerate(train_loader):
#         data = data+0.5
#         print(data.min(),data.max())
        step += 1
        data = data.reshape(-1,784).cuda()
        optimizer.zero_grad()
        recon_error = model.recon_error(data)
        loss = torch.mean(recon_error)
        loss.backward()
        optimizer.step()
        avg_loss += loss
        if i % 100 == 0:    
            print('Epoch [{}/{}] Batch [{}/{}]=> Loss: {:.5f}'.format(epoch, opt.max_epoch, i,len(train_loader), avg_loss / step))
    
    if epoch % 100 == 0:
        model_state = model.state_dict()
        #print(model_state)
        ckpt_name = 'MNIST_{}_holdout_{}_epoch_{}_no_sigmoid'.format(model_name,opt.one_class,epoch)
        ckpt_path = os.path.join(opt.save_dir,ckpt_name + ".pth")
        torch.save(model_state, ckpt_path)
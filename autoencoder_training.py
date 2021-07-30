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
from model import Fully_Connected_AE
from tensorboardX import SummaryWriter


# download path 정의
download_root = './dataset'

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",type=int, default=0, help="cuda index")
parser.add_argument("--max_epoch",type=int, default=100, help="training_epoch")
parser.add_argument("--save_dir",default='trained_models', help="saving_directions_for_trained_weights")
parser.add_argument("--lr",default=0.001, help="learning_rate")
parser.add_argument("--input_dim",type=int,default=784, help="input_dimensions")
parser.add_argument("--dimensions",type=str, help="input 6 dimensions separated by commas", default = '512,256,64,16,0,0')
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--leave",type=int)
parser.add_argument("--sigmoid", action='store_true')
parser.add_argument("--no-bias", action='store_true')
parser.add_argument("--run",type=str,default = None)


opt = parser.parse_args()
os.makedirs(os.path.join(opt.save_dir, 'pretrained','leave_out_{}'.format(opt.leave)), exist_ok=True)


writer = SummaryWriter(logdir='./tensorboard/autoencoder_training/leaveout_{}_dim_{}_sigmoid_{}_{}'.format(opt.leave, opt.dimensions,opt.sigmoid,opt.run))


dimensions = list(map(int,opt.dimensions.split(',')))
if len(dimensions)!=6:
    raise('give me 6 dimensions for autoencoder network!')

model = Fully_Connected_AE(opt.input_dim, dimensions,opt.sigmoid, not opt.no_bias)
optimizer = torch.optim.Adam(model.parameters(), opt.lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epoch, eta_min=0, last_epoch=-1)

mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
])

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
# valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
# test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

idx = train_dataset.targets!=opt.leave
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

# device = torch.device('cuda')
# torch.cuda.set_device(opt.gpu)

model_name = "_".join(opt.dimensions.split(','))
model = model.cuda()
print(model)



model.train()
global_step=0
for epoch in range(1, opt.max_epoch+ 1):

    if epoch == 1 :
        model_state = model.state_dict()
        #print(model_state)
        ckpt_name = '{}_sigmoid_{}_epoch_0_run_{}'.format(model_name,opt.sigmoid, opt.run)
        ckpt_path = os.path.join(opt.save_dir,'pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")
        torch.save(model_state, ckpt_path)

    avg_loss = 0
    step = 0
    for i, (data,label) in enumerate(train_loader):
#         data = data+0.5
#         print(data.min(),data.max())
        # print(data[0])
        step += 1
        data = data.reshape(-1,784).cuda()
        optimizer.zero_grad()
        recon_error = model.recon_error(data)
        loss = torch.mean(recon_error)
        loss.backward()
        optimizer.step()
        avg_loss += loss
        global_step+=1
        writer.add_scalar('leaveout_{}'.format(opt.leave), loss, global_step)
        if i % 100 == 0:    
            print('Epoch [{}/{}] Batch [{}/{}]=> Loss: {:.5f}'.format(epoch, opt.max_epoch, i,len(train_loader), avg_loss / step))
    # scheduler.step()


    if epoch % 50 == 0 :
        model_state = model.state_dict()
        #print(model_state)
        ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,epoch, opt.run)
        ckpt_path = os.path.join(opt.save_dir,'pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")
        torch.save(model_state, ckpt_path)
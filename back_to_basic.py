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
from utils_for_me import *
from torch.optim.lr_scheduler import CosineAnnealingLR


torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
parser.add_argument('--run', type=str, help='experiment name', required=True)
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('--epoch', type=int, help ='finetune_epoch', default=30)
parser.add_argument('--OE_ratio', type=float, help ='OOD_loss_ratio', default=0)
parser.add_argument('--supermask_lr', type=float, help ='supermask_learning_rate', default=0.01)
parser.add_argument('--supermask_epoch', type=int, help ='finetune_epoch', default=20)

parser.add_argument('--mask', help ='how_to_choose_mask', required=True)
parser.add_argument("--weights_to_prune",'-wtp', nargs="+", default=["1", "2","3","4","5", "6","7","8"])
parser.add_argument("--sigmoid", action='store_true')



# parser.add_argument('--feature', dest='feature', default=False, action='store_true')


args = parser.parse_args()
print(args)

torch.cuda.set_device(args.device)



class opt:
    layer_num=784
    h_dim1=512
    h_dim2=256
    h_dim3=64
    h_dim4=16
    h_dim5 = 0
    h_dim6 = 0
    one_class=args.leave
    lr=0.001
    finetune_epoch=args.epoch


mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize((0.5,), (1.0,))
])

ind_dataset = torchvision.datasets.MNIST('./dataset', transform=mnist_transform, train=False, download=True)
train_dataset = torchvision.datasets.MNIST('./dataset', transform=mnist_transform, train=True, download=True)

idx = train_dataset.targets!=opt.one_class
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                         batch_size=64,
                         shuffle=True)

idx = ind_dataset.targets!=opt.one_class
ind_dataset.targets = ind_dataset.targets[idx]
ind_dataset.data = ind_dataset.data[idx]

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, 
                         batch_size=64,
                         shuffle=False)

mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize((0.5,), (1.0,))
])

ood_dataset = torchvision.datasets.MNIST('./dataset', transform=mnist_transform, train=False, download=True)

idx = ood_dataset.targets==opt.one_class
ood_dataset.targets = ood_dataset.targets[idx]
ood_dataset.data = ood_dataset.data[idx]

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, 
                         batch_size=64,
                         shuffle=False)

ood_train = torchvision.datasets.EMNIST(root='./dataset', split='letters',train=True, download=True,transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

idx = ood_train.targets==opt.one_class
ood_train.targets = ood_train.targets[idx]
ood_train.data = ood_train.data[idx]

ood_train_loader = torch.utils.data.DataLoader(dataset=ood_train, 
                         batch_size=64,
                         shuffle=True)        
#==================================================            
            
result_dir = f'./SVS_experiments/{args.run}/leaveout_{opt.one_class}'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    print(f'creating {result_dir}')

            
writer = SummaryWriter(logdir=result_dir)

if args.mask == 'svs1':
    xaxis_range = [0.85]# 0,0.1,0.3,0.5,0.7,0.9,0.92,0.93,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.99]
elif args.mask == 'svs2':
    xaxis_range = [0.5,1,1.5,2,2.5]
elif args.mask == 'supermask':
    xaxis_range = [0.85]

for i, percent in enumerate(xaxis_range):
    
    print(percent)
    print(args.sigmoid)
    
    pruned_model = AE(opt.layer_num, opt.h_dim1, opt.h_dim2, opt.h_dim3,opt.h_dim4,args.sigmoid)
    pruned_model.cuda()

    all_layers = [(pruned_model.encoder.fc1,'weight'),(pruned_model.encoder.fc2,'weight'),(pruned_model.encoder.fc3,'weight'),(pruned_model.encoder.fc4,'weight'),(pruned_model.decoder.fc4,'weight'),(pruned_model.decoder.fc3,'weight'),(pruned_model.decoder.fc2,'weight'),(pruned_model.decoder.fc1,'weight')] 

    parameters_to_prune = []

    for i0 in args.weights_to_prune:

        parameters_to_prune.append(all_layers[int(i0)-1])

    parameters_to_prune = tuple(parameters_to_prune)

    print(parameters_to_prune)


    if args.mask=='magnitude':
        model_name = 'MNIST_{}_{}_{}_{}_{}_{}'.format(opt.h_dim1, opt.h_dim2, opt.h_dim3, opt.h_dim4, opt.h_dim5, opt.h_dim6)
        model_path = model_name+'_holdout_{}_epoch_100_no_sigmoid.pth'.format(opt.one_class)
        pruned_model.load_state_dict(torch.load(os.path.join('./trained_models',model_path),map_location=torch.device('cpu')))
        
#         parameters_to_prune = (
#     #         (pruned_model.encoder.fc1,'weight'),
#     #         (pruned_model.encoder.fc2,'weight'),
#             (pruned_model.encoder.fc3,'weight'),
#     #         (pruned_model.encoder.fc4,'weight'),

#     #         (pruned_model.decoder.fc1,'weight'),
#     #         (pruned_model.decoder.fc2,'weight'),
#             (pruned_model.decoder.fc3,'weight'),
#     #         (pruned_model.decoder.fc4,'weight'),
#         )
    
        
        prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount=percent)
    
    elif args.mask == 'snip':
        
        from snip import *
        
        parameter_shapes_to_prune = []
        
        for i1 in args.weights_to_prune:
            
            parameter_shapes_to_prune.append(all_layers[int(i1)-1][0].weight.shape)
        
        keep_masks = SNIP(pruned_model, 1.0-percent, train_loader, parameter_shapes_to_prune, args.device)
        print((keep_masks[0]==0).sum())
        
        prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount=percent)

        print(pruned_model.encoder.fc3.weight_mask)
        print(keep_masks[0])

        for indd, i1 in enumerate(args.weights_to_prune):
            
            prune.custom_from_mask(all_layers[int(i1)-1][0], 'weight', mask=keep_masks[indd])
    
    elif args.mask == 'supermask':
        supermask_model = Supermask_AE(opt.layer_num, opt.h_dim1, opt.h_dim2, opt.h_dim3,opt.h_dim4,args.weights_to_prune, percent,args.sigmoid)
        supermask_model.cuda()   

        optimizer = torch.optim.SGD(
            [p for p in supermask_model.parameters() if p.requires_grad],
            lr=args.supermask_lr,
            momentum=0.9,
            weight_decay=0.0005,
        )

        
        plus =0
        for p in supermask_model.parameters():
            if p.requires_grad:
                print(p.shape)
                plus+=1
                
        print('number of parameters with gradient true : {}'.format(plus))
        
        scheduler = CosineAnnealingLR(optimizer, T_max=args.supermask_epoch)

        supermask_model.train()
        
        
        if args.OE_ratio==0:
            total_step=0
            for epoch in trange(1, args.supermask_epoch+ 1):
                avg_loss = 0
                step = 0                
                for (data,label) in train_loader:
                    step += 1
                    total_step +=1
                    data = data.reshape(-1,784).cuda()
                    optimizer.zero_grad()
                    recon_error = supermask_model.recon_error(data)
                    loss = torch.mean(recon_error)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss
                    writer.add_scalar('supermask/sparsity_{}/loss'.format(percent), avg_loss/step,total_step)
                    sparsity_per_layer = supermask_model.sparsity_per_layer()
                    writer.add_scalar('supermask/sparsity_{}/layer1_sparsity'.format(percent), sparsity_per_layer[0],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer2_sparsity'.format(percent), sparsity_per_layer[1],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer3_sparsity'.format(percent), sparsity_per_layer[2],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer4_sparsity'.format(percent), sparsity_per_layer[3],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer5_sparsity'.format(percent), sparsity_per_layer[4],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer6_sparsity'.format(percent), sparsity_per_layer[5],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer7_sparsity'.format(percent), sparsity_per_layer[6],total_step)
                    writer.add_scalar('supermask/sparsity_{}/layer8_sparsity'.format(percent), sparsity_per_layer[7],total_step)
                    
        else:
            total_step=0
            for epoch in trange(1, args.supermask_epoch+ 1):
                avg_loss = [0,0,0]
                step = 0                
                for ((data,_),(ood_data,_)) in zip(train_loader, ood_train_loader):
                    step += 1
                    total_step +=1
                    data = data.reshape(-1,784).cuda()
                    ood_data = ood_data.reshape(-1,784).cuda()
                    optimizer.zero_grad()
                    recon_error = supermask_model.recon_error(data)
                    ood_recon_error = supermask_model.recon_error(ood_data)
                    loss = torch.mean(recon_error) - args.OE_ratio * torch.mean(ood_recon_error)
                    loss.backward()
                    optimizer.step()
                    avg_loss[0] += loss
                    avg_loss[1] += torch.mean(recon_error)
                    avg_loss[2] += torch.mean(ood_recon_error)
                    
                    writer.add_scalar('supermask_OE/sparsity_{}/loss'.format(percent), avg_loss[0]/step,total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/IND_loss'.format(percent), avg_loss[1]/step,total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/OOD_loss'.format(percent), avg_loss[2]/step,total_step)

                    sparsity_per_layer = supermask_model.sparsity_per_layer()
                    writer.add_scalar('supermask_OE/sparsity_{}/layer1_sparsity'.format(percent), sparsity_per_layer[0],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer2_sparsity'.format(percent), sparsity_per_layer[1],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer3_sparsity'.format(percent), sparsity_per_layer[2],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer4_sparsity'.format(percent), sparsity_per_layer[3],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer5_sparsity'.format(percent), sparsity_per_layer[4],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer6_sparsity'.format(percent), sparsity_per_layer[5],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer7_sparsity'.format(percent), sparsity_per_layer[6],total_step)
                    writer.add_scalar('supermask_OE/sparsity_{}/layer8_sparsity'.format(percent), sparsity_per_layer[7],total_step)        
        init_weights = []
        weight_masks = []
        
        prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount=percent)   
        
        for layer_from in supermask_model.modules():
            for layer_to in all_layers:
                if isinstance(layer_from, nn.Linear) or isinstance(layer_from, Supermask_Linear):
                    if layer_from.weight.shape==layer_to[0].weight.shape:
                        if isinstance(layer_from, Supermask_Linear):
                            layer_to[0].weight_orig.data = layer_from.weight.data
                            layer_to[0].weight_mask.data = layer_from.mask().data
                            layer_to[0].weight.data = layer_from.weight.data * layer_from.mask().data
                            layer_to[0].bias.data = layer_from.bias.data
                            
                        else:
                            layer_to[0].weight.data = layer_from.weight.data
                            layer_to[0].bias.data = layer_from.bias.data

    elif args.mask == 'svs1':
        supermask_model = Supermask_SVS1_AE(opt.layer_num, opt.h_dim1, opt.h_dim2, opt.h_dim3,opt.h_dim4,args.weights_to_prune,args.sigmoid)
        supermask_model.cuda()   

        optimizer = torch.optim.SGD(
            [p for p in supermask_model.parameters() if p.requires_grad],
            lr=args.supermask_lr,
            momentum=0.9,
            weight_decay=0.0005,
        )

        
        plus =0
        for p in supermask_model.parameters():
            if p.requires_grad:
                plus+=1
                
        print('number of parameters with gradient true : {}'.format(plus))
        
        scheduler = CosineAnnealingLR(optimizer, T_max=args.supermask_epoch)
        
        supermask_model.train()
        
        if args.OE_ratio==0:
            total_step=0
            for epoch in trange(1, args.supermask_epoch+ 1):
                avg_loss = 0
                step = 0
                for (data,label) in train_loader:
                    step += 1
                    total_step +=1
                    data = data.reshape(-1,784).cuda()
                    optimizer.zero_grad()
                    recon_error = supermask_model.recon_error(data)
                    loss = torch.mean(recon_error)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss
                    writer.add_scalar('SVS1/loss', avg_loss/step,total_step)
                    sparsity_per_layer = supermask_model.sparsity_per_layer()
                    writer.add_scalar('SVS1/layer1_sparsity', sparsity_per_layer[0],total_step)
                    writer.add_scalar('SVS1/layer2_sparsity', sparsity_per_layer[1],total_step)
                    writer.add_scalar('SVS1/layer3_sparsity', sparsity_per_layer[2],total_step)
                    writer.add_scalar('SVS1/layer4_sparsity', sparsity_per_layer[3],total_step)
                    writer.add_scalar('SVS1/layer5_sparsity', sparsity_per_layer[4],total_step)
                    writer.add_scalar('SVS1/layer6_sparsity', sparsity_per_layer[5],total_step)
                    writer.add_scalar('SVS1/layer7_sparsity', sparsity_per_layer[6],total_step)
                    writer.add_scalar('SVS1/layer8_sparsity', sparsity_per_layer[7],total_step)
                    
        else:
            total_step=0
            for epoch in trange(1, args.supermask_epoch+ 1):
                avg_loss = [0,0,0]
                step = 0
                for ((data,_),(ood_data,_)) in zip(train_loader, ood_train_loader):
                    step += 1
                    total_step +=1
                    data = data.reshape(-1,784).cuda()
                    ood_data = ood_data.reshape(-1,784).cuda()
                    optimizer.zero_grad()
                    recon_error = supermask_model.recon_error(data)
                    OOD_recon_error = supermask_model.recon_error(ood_data)

                    loss = torch.mean(recon_error) - args.OE_ratio * torch.mean(OOD_recon_error)
                    loss.backward()
                    optimizer.step()
                    avg_loss[0] += loss
                    avg_loss[1] += torch.mean(recon_error)
                    avg_loss[2] += torch.mean(OOD_recon_error)
                    writer.add_scalar('SVS1_OE/loss', avg_loss[0]/step,total_step)
                    writer.add_scalar('SVS1_OE/IND_loss', avg_loss[1]/step,total_step)
                    writer.add_scalar('SVS1_OE/OOD_loss', avg_loss[2]/step,total_step)

                    sparsity_per_layer = supermask_model.sparsity_per_layer()
                    writer.add_scalar('SVS1_OE/layer1_sparsity', sparsity_per_layer[0],total_step)
                    writer.add_scalar('SVS1_OE/layer2_sparsity', sparsity_per_layer[1],total_step)
                    writer.add_scalar('SVS1_OE/layer3_sparsity', sparsity_per_layer[2],total_step)
                    writer.add_scalar('SVS1_OE/layer4_sparsity', sparsity_per_layer[3],total_step)
                    writer.add_scalar('SVS1_OE/layer5_sparsity', sparsity_per_layer[4],total_step)
                    writer.add_scalar('SVS1_OE/layer6_sparsity', sparsity_per_layer[5],total_step)
                    writer.add_scalar('SVS1_OE/layer7_sparsity', sparsity_per_layer[6],total_step)
                    writer.add_scalar('SVS1_OE/layer8_sparsity', sparsity_per_layer[7],total_step)        
        init_weights = []
        weight_masks = []
        
        prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount=0.5)   
        
        for layer_from in supermask_model.modules():
            for layer_to in all_layers:
                if isinstance(layer_from, nn.Linear) or isinstance(layer_from, Supermask_SVS1_Linear):
                    if layer_from.weight.shape==layer_to[0].weight.shape:
                        if isinstance(layer_from, Supermask_SVS1_Linear):
                            layer_to[0].weight_orig.data = layer_from.weight.data
                            layer_to[0].weight_mask.data = layer_from.mask().data
                            layer_to[0].weight.data = layer_from.weight.data * layer_from.mask().data
                            layer_to[0].bias.data = layer_from.bias.data
                            
                        else:
                            layer_to[0].weight.data = layer_from.weight.data
                            layer_to[0].bias.data = layer_from.bias.data

    elif args.mask == 'svs2':
        supermask_model = Supermask_SVS2_AE(opt.layer_num, opt.h_dim1, opt.h_dim2, opt.h_dim3,opt.h_dim4,args.weights_to_prune,percent,args.sigmoid)
        supermask_model.cuda()   

        optimizer = torch.optim.SGD(
            [p for p in supermask_model.parameters() if p.requires_grad],
            lr=args.supermask_lr,
            momentum=0.9,
            weight_decay=0.0005,
        )

        
        plus =0
        for p in supermask_model.parameters():
            if p.requires_grad:
                plus+=1
                
        print('number of parameters with gradient true : {}'.format(plus))
        
        scheduler = CosineAnnealingLR(optimizer, T_max=args.supermask_epoch)
        
        supermask_model.train()
        
        if args.OE_ratio==0:
            total_step=0
            for epoch in trange(1, args.supermask_epoch+ 1):
                avg_loss = 0
                step = 0
                for (data,label) in train_loader:
                    step += 1
                    total_step +=1
                    data = data.reshape(-1,784).cuda()
                    optimizer.zero_grad()
                    recon_error = supermask_model.recon_error(data)
                    loss = torch.mean(recon_error)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss
                    writer.add_scalar('SVS2/knob_{}/loss'.format(percent), avg_loss/step,total_step)
                    sparsity_per_layer = supermask_model.sparsity_per_layer()
                    writer.add_scalar('SVS2/knob_{}/layer1_sparsity'.format(percent), sparsity_per_layer[0],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer2_sparsity'.format(percent), sparsity_per_layer[1],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer3_sparsity'.format(percent), sparsity_per_layer[2],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer4_sparsity'.format(percent), sparsity_per_layer[3],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer5_sparsity'.format(percent), sparsity_per_layer[4],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer6_sparsity'.format(percent), sparsity_per_layer[5],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer7_sparsity'.format(percent), sparsity_per_layer[6],total_step)
                    writer.add_scalar('SVS2/knob_{}/layer8_sparsity'.format(percent), sparsity_per_layer[7],total_step)
                    
        init_weights = []
        weight_masks = []
        
        prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount=0.5)   
        
        for layer_from in supermask_model.modules():
            for layer_to in all_layers:
                if isinstance(layer_from, nn.Linear) or isinstance(layer_from, Supermask_SVS2_Linear):
                    if layer_from.weight.shape==layer_to[0].weight.shape:
                        if isinstance(layer_from, Supermask_SVS2_Linear):
                            layer_to[0].weight_orig.data = layer_from.weight.data
                            layer_to[0].weight_mask.data = layer_from.mask().data
                            layer_to[0].weight.data = layer_from.weight.data * layer_from.mask().data
                            layer_to[0].bias.data = layer_from.bias.data
                            
                        else:
                            layer_to[0].weight.data = layer_from.weight.data
                            layer_to[0].bias.data = layer_from.bias.data        
                            
                            
    ind_recon = reconstrucion_errors(pruned_model, ind_loader)
    ood_recon = reconstrucion_errors(pruned_model, ood_loader)
            
    auroc = calculate_auroc(ind_recon, ood_recon)
    print('before finetuning : auroc = {}'.format(auroc))
        
    writer.add_scalar('AUROC_before_FT', auroc, i)
    writer.add_scalar('ind_RE_before_FT', torch.mean(ind_recon), i)
    writer.add_scalar('ood_RE_before_FT', torch.mean(ood_recon), i)
    
    check_reconstructed_images(pruned_model, writer, i, percent, "before_FT", ind_loader, ood_loader, result_dir)
    
#     print(pruned_model.encoder.fc3.weight)
#     print(pruned_model.encoder.fc3.weight_mask)
    
    global_sparsity = show_global_sparsity(pruned_model)
    writer.add_scalar('global_sparsity'.format(args.leave), global_sparsity, i)
    
    torch.save(pruned_model.state_dict(), f'{result_dir}/sparsity_{percent}_before_FT.pkl')

            
    ### Finetune the remaining weights
            
    optimizer = torch.optim.Adam(pruned_model.parameters(), opt.lr)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.finetune_epoch, eta_min=0, last_epoch=-1)

    time.sleep(5)
    pruned_model.train()
    
    loss_list = []
    total_step=0
    for epoch in trange(1, opt.finetune_epoch+ 1):
        avg_loss = 0
        step = 0
        for (data,label) in train_loader:
            step += 1
            total_step+=1
            data = data.reshape(-1,784).cuda()
            optimizer.zero_grad()
            recon_error = pruned_model.recon_error(data)
            loss = torch.mean(recon_error)
            loss.backward()
            optimizer.step()
            avg_loss += loss
            loss_list.append(avg_loss/step)
            writer.add_scalar('remaining_weight_training_curve/sparsity_{}'.format(percent), avg_loss/step,total_step)


    representations, classes = check_representations(pruned_model, ind_loader)
    torch.save(representations, f'{result_dir}/sparsity_{percent}_after_FT_IND_representations.pkl')
    torch.save(classes, f'{result_dir}/sparsity_{percent}_after_FT_IND_classes.pkl')

    representations, classes = check_representations(pruned_model, ood_loader)
    torch.save(representations, f'{result_dir}/sparsity_{percent}_after_FT_OOD_representations.pkl')
    torch.save(classes, f'{result_dir}/sparsity_{percent}_after_FT_OOD_classes.pkl')

    
    
    loss_list = torch.stack(loss_list)
    plt.figure()
    plt.plot(range(step*args.epoch),loss_list.detach().cpu().numpy())
    plt.xlabel('iteration')
    plt.ylabel('training_loss')
    plt.title('finetune_epoch_{}_pruning_ratio_{}'.format(args.epoch,percent))
    plt.savefig('{}/pruning_ratio_{}_learning_curve.png'.format(result_dir, percent))
    plt.close()

    model_state = pruned_model.state_dict()
    #print(model_state)
    torch.save(pruned_model.state_dict(), f'{result_dir}/sparsity_{percent}_after_FT.pkl')


    ind_recon = reconstrucion_errors(pruned_model, ind_loader)
    ood_recon = reconstrucion_errors(pruned_model, ood_loader)
            
    auroc = calculate_auroc(ind_recon, ood_recon)
    print('after finetuning : auroc = {}'.format(auroc))

            
    writer.add_scalar('AUROC_after_FT', auroc, i)
    writer.add_scalar('ind_RE_after_FT', torch.mean(ind_recon), i)
    writer.add_scalar('ood_RE_after_FT', torch.mean(ood_recon), i)

    check_reconstructed_images(pruned_model, writer, i, percent, "after_FT", ind_loader, ood_loader, result_dir)

    show_global_sparsity(pruned_model)
            
#     torch.save(pruned_model.state_dict(), f'{result_dir}/sparsity_{percent}_after_FT.pkl')




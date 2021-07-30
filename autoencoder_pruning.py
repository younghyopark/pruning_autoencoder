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
from model import Fully_Connected_AE

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='cuda device index', default=0)
parser.add_argument('--pretrained_run', type=str, help='run_name_for_pretrained_weights', default = 'retraining')
parser.add_argument('--pruning_run', type=str, help='run_name_for_pruning_procedure', default = None)
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('--epoch', type=int, help ='finetune_epoch', default=10)
parser.add_argument("--input_dim",type=int,default=784, help="input_dimensions")
parser.add_argument("--pruning_technique","-tech", type=int,default=0, help="prune_weights_based_on_what?")
parser.add_argument("--dimensions",type=str, help="input 6 dimensions separated by commas", default = '512,256,64,16,0,0')
parser.add_argument("--lr",type=float, default =0.001)
parser.add_argument("--batch_size",type=int, default =256)
parser.add_argument("--weights_to_prune",'-wtp', nargs="+", default=["1", "2","3","4","5", "6","7","8"])
parser.add_argument("--sigmoid", action='store_true')
parser.add_argument("--no-bias", action='store_true')
parser.add_argument("--layerwise-pruning", action='store_true')



opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.device)


mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize((0.5,), (1.0,))
])

ind_dataset = torchvision.datasets.MNIST('./dataset', transform=mnist_transform, train=False, download=True)
train_dataset = torchvision.datasets.MNIST('./dataset', transform=mnist_transform, train=True, download=True)

idx = train_dataset.targets!=opt.leave
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                         batch_size=opt.batch_size,
                         shuffle=True)

idx = ind_dataset.targets!=opt.leave
ind_dataset.targets = ind_dataset.targets[idx]
ind_dataset.data = ind_dataset.data[idx]

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, 
                         batch_size=opt.batch_size,
                         shuffle=False)

mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
#     torchvision.transforms.Normalize((0.5,), (1.0,))
])

ood_dataset = torchvision.datasets.MNIST('./dataset', transform=mnist_transform, train=False, download=True)

idx = ood_dataset.targets==opt.leave
ood_dataset.targets = ood_dataset.targets[idx]
ood_dataset.data = ood_dataset.data[idx]

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, 
                         batch_size=opt.batch_size,
                         shuffle=False)

# ood_train = torchvision.datasets.EMNIST(root='./dataset', split='letters',train=True, download=True,transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

# idx = ood_train.targets==opt.leave
# ood_train.targets = ood_train.targets[idx]
# ood_train.data = ood_train.data[idx]

# ood_train_loader = torch.utils.data.DataLoader(dataset=ood_train, 
#                          batch_size=64,
#                          shuffle=True)        
#==================================================            
            
dimensions = list(map(int,opt.dimensions.split(',')))
if len(dimensions)!=6:
    raise('give me 6 dimensions for autoencoder network!')

model_name = "_".join(opt.dimensions.split(','))

writer = SummaryWriter(logdir='./tensorboard/autoencoder_pruning_new/pruning_{}/leaveout_{}_dim_{}_sigmoid_{}_{}'.format(opt.pruning_technique, opt.leave, opt.dimensions,opt.sigmoid,opt.pruning_run))

result_dir = os.path.join('pruning_results','pruning_technique_{}'.format(opt.pruning_technique),'leave_out_{}'.format(opt.leave),opt.pruning_run)
os.makedirs(result_dir,exist_ok = True)


range_1 = np.arange(0,0.7,0.1)
range_2 = np.arange(0.7,0.9,0.05)
range_3 = np.arange(0.9,1,0.001)
xaxis_range = np.concatenate((range_1, range_2, range_3),0)

# model = 

layer_sparsity_count =0 

for i, percent in enumerate(tqdm(xaxis_range)):
    
    print(percent)
    print(opt.sigmoid)
    
    pruned_model = Fully_Connected_AE(opt.input_dim, dimensions,opt.sigmoid)

    ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'100', opt.pretrained_run)
    ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")
    print(ckpt_path)
    pruned_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    pruned_model = pruned_model.cuda()

    all_layers = [(pruned_model.encoder.fc1,'weight'),(pruned_model.encoder.fc2,'weight'),(pruned_model.encoder.fc3,'weight'),(pruned_model.encoder.fc4,'weight'),(pruned_model.decoder.fc4,'weight'),(pruned_model.decoder.fc3,'weight'),(pruned_model.decoder.fc2,'weight'),(pruned_model.decoder.fc1,'weight')] 

    parameters_to_prune = []

    print(opt.weights_to_prune)

    for i0 in opt.weights_to_prune:

        parameters_to_prune.append(all_layers[int(i0)-1])

    parameters_to_prune = tuple(parameters_to_prune)

    print(parameters_to_prune)

    if not opt.layerwise_pruning:
        print('pruning GLOBALLY!')
        if opt.pruning_technique == 0 :

            ## weights with small final weights are pruned
            scores = pruned_model.state_dict()
            scores = {k:abs(v) for k,v in scores.items()}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

        elif opt.pruning_technique == 1 :

            ## weights with large final weights are pruned
            scores = pruned_model.state_dict()
            scores = {k:1/abs(v) for k,v in scores.items()}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

        elif opt.pruning_technique == 2 :


            ## weights with small initial weights are pruned
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            scores = torch.load(ckpt_path)
            scores = {k:abs(v) for k,v in scores.items()}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

        elif opt.pruning_technique == 3 :


            ## weights with large initial weights are pruned
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            scores = torch.load(ckpt_path)
            scores = {k:1/abs(v) for k,v in scores.items()}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

        elif opt.pruning_technique == 4 :

            ## weights with weights moved a lot while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            scores = {k1:abs(v2-v1) for ((k1,v1), (k2,v2)) in zip(init_weights.items(),trained_weights.items())}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)
        
        elif opt.pruning_technique == 5 :

            ## weights with weights' magnitude moved a lot while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            scores = {k1:abs(abs(v2)-abs(v1)) for ((k1,v1), (k2,v2)) in zip(init_weights.items(),trained_weights.items())}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

        elif opt.pruning_technique == 6 :

            ## weights with weights moved little while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            scores = {k1:1/abs(v2-v1) for ((k1,v1), (k2,v2)) in zip(init_weights.items(),trained_weights.items())}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)
        
        elif opt.pruning_technique == 7 :

            ## weights with weights' magnitude moved little while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            scores = {k1:1/abs(abs(v2)-abs(v1)) for ((k1,v1), (k2,v2)) in zip(init_weights.items(),trained_weights.items())}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

        elif opt.pruning_technique == 8 :

            ## random pruning!
            scores = pruned_model.state_dict()
            scores = {k:torch.rand_like(v) for k,v in scores.items()}

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            importance_scores = scores,
            amount=percent)

    else:
    
        print('pruning LAYER-WISE!')

        if opt.pruning_technique == 0 :

            ## weights with small final weights are pruned
            trained_weight = pruned_model.state_dict()
            
            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = abs(trained_weight[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)

        elif opt.pruning_technique == 1 :

            ## weights with large final weights are pruned
            trained_weight = pruned_model.state_dict()
            
            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = 1/abs(trained_weight[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)

        elif opt.pruning_technique == 2 :

            ## weights with small initial weights are pruned
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weight = torch.load(ckpt_path)
            
            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = abs(init_weight[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)


        elif opt.pruning_technique == 3 :


            ## weights with large initial weights are pruned
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weight = torch.load(ckpt_path)
            
            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = 1/abs(init_weight[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)

        elif opt.pruning_technique == 4 :

            ## weights with weights moved a lot while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = abs(trained_weights[name]-init_weights[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)
        
        elif opt.pruning_technique == 5 :

            ## weights with weights' magnitude moved a lot while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = abs(abs(trained_weights[name])-abs(init_weights[name]))
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)

        elif opt.pruning_technique == 6 :

            ## weights with weights moved little while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = 1/abs(trained_weights[name]-init_weights[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)
        
        elif opt.pruning_technique == 7 :

            ## weights with weights' magnitude moved little while training
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'0', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")

            init_weights= torch.load(ckpt_path)
            trained_weights = pruned_model.state_dict()

            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = 1/abs(abs(trained_weights[name])-abs(init_weights[name]))
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)

        elif opt.pruning_technique == 8 :
            
            trained_weights = pruned_model.state_dict()

            ## random pruning!
            for pruning_weight_num in opt.weights_to_prune:
                if int(pruning_weight_num) <5 : 
                    name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                else:
                    name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                scores = torch.rand_like(trained_weights[name])
                prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', percent,scores)                       
                            
    ind_recon = reconstrucion_errors(pruned_model, ind_loader)
    ood_recon = reconstrucion_errors(pruned_model, ood_loader)
            
    auroc = calculate_auroc(ind_recon, ood_recon)
    print('before finetuning : auroc = {}'.format(auroc))
        
    writer.add_scalar('AUROC_before_FT', auroc, i)
    writer.add_scalar('ind_RE_before_FT', torch.mean(ind_recon), i)
    writer.add_scalar('ood_RE_before_FT', torch.mean(ood_recon), i)
    
    check_reconstructed_images(pruned_model, writer, i, percent, "before_FT", ind_loader, ood_loader, result_dir, model_name, opt.sigmoid, opt.pruning_run)
        
    global_sparsity = show_global_sparsity(pruned_model)
    writer.add_scalar('global_sparsity'.format(opt.leave), global_sparsity, i)
    writer.add_scalar('percentage', percent, i)


    for this_layer in range(8):
        layer_sparsity = show_layer_sparsity(pruned_model, this_layer)
        if layer_sparsity == 100:
            layer_sparsity_count+=1
        writer.add_scalar('layer_sparsity/layer_{}'.format(this_layer+1), layer_sparsity, i)

    if layer_sparsity_count >3 : 
        raise('layer_sparsity_reached_100% 4 times.')


    model_paths = os.path.join('trained_models','pruning_technique_{}'.format(opt.pruning_technique),'leave_out_{}'.format(opt.leave), opt.pruning_run)
    os.makedirs(model_paths,exist_ok = True)
    model_name = "_".join(opt.dimensions.split(','))
    ckpt_name = 'before_FT_{}_sigmoid_{}_sparsity_{}_run_{}.pth'.format(model_name,opt.sigmoid, percent, opt.pruning_run)
    torch.save(pruned_model.state_dict(), os.path.join(model_paths,ckpt_name))

            
    ### Finetune the remaining weights
            
    optimizer = torch.optim.Adam(pruned_model.parameters(), opt.lr)

    time.sleep(5)
    pruned_model.train()
    
    loss_list = []
    total_step=0
    for epoch in trange(1, opt.epoch+ 1):
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
            writer.add_scalar('finetuning_curve/sparsity_{:.3f}'.format(percent), avg_loss/step,total_step)


    # representations, classes = check_representations(pruned_model, ind_loader)
    #     # ckpt_name = 'before_FT_{}_sigmoid_{}_run_{}.pth'.format(model_name,opt.sigmoid, opt.pruning_run)

    # torch.save(representations, f'{model_paths}/sparsity_{percent}_after_FT_IND_representations.pkl')
    # torch.save(classes, f'{result_dir}/sparsity_{percent}_after_FT_IND_classes.pkl')

    # representations, classes = check_representations(pruned_model, ood_loader)
    # torch.save(representations, f'{result_dir}/sparsity_{percent}_after_FT_OOD_representations.pkl')
    # torch.save(classes, f'{result_dir}/sparsity_{percent}_after_FT_OOD_classes.pkl')

    
    
    loss_list = torch.stack(loss_list)
    plt.figure()
    plt.plot(range(step*opt.epoch),loss_list.detach().cpu().numpy())
    plt.xlabel('iteration')
    plt.ylabel('training_loss')
    plt.title('finetune_epoch_{}_pruning_ratio_{:.3f}'.format(opt.epoch,percent))
    plt.savefig('{}/pruning_ratio_{:.3f}_learning_curve.png'.format(result_dir, percent))
    plt.close()

    model_paths = os.path.join('trained_models','pruning_technique_{}'.format(opt.pruning_technique),'leave_out_{}'.format(opt.leave), opt.pruning_run)
    model_name = "_".join(opt.dimensions.split(','))
    ckpt_name = 'after_FT_{}_sigmoid_{}_run_{}.pth'.format(model_name,opt.sigmoid, opt.pruning_run)
    torch.save(pruned_model.state_dict(), os.path.join(model_paths,ckpt_name))


    ind_recon = reconstrucion_errors(pruned_model, ind_loader)
    ood_recon = reconstrucion_errors(pruned_model, ood_loader)
            
    auroc = calculate_auroc(ind_recon, ood_recon)
    print('after finetuning : auroc = {}'.format(auroc))

            
    writer.add_scalar('AUROC_after_FT', auroc, i)
    writer.add_scalar('ind_RE_after_FT', torch.mean(ind_recon), i)
    writer.add_scalar('ood_RE_after_FT', torch.mean(ood_recon), i)

    check_reconstructed_images(pruned_model, writer, i, percent, "after_FT", ind_loader, ood_loader, result_dir, model_name, opt.sigmoid, opt.pruning_run)

    show_global_sparsity(pruned_model)
            
#     torch.save(pruned_model.state_dict(), f'{result_dir}/sparsity_{percent}_after_FT.pkl')




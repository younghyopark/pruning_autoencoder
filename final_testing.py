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
from utils_for_me import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Fully_Connected_AE
import logging
# from notify_run import Notify
import telegram


my_token = '1923368350:AAFkUmZwb6OCljMTai031vY0A4zmi3tVDJQ'

bot = telegram.Bot(token = my_token)   #bot을 선언합니다.

# updates = bot.getUpdates()  #업데이트 내역을 받아옵니다.

# for u in updates :   # 내역중 메세지를 출력합니다.


CHAT_ID: int = 1938870254


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_run', type=str, help='run_name_for_pretrained_weights', default = 'retraining')
parser.add_argument('--pruning_run', type=str, help='run_name_for_pruning_procedure', default = None)
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('--epoch', type=int, help ='finetune_epoch', default=20)
parser.add_argument("--input_dim",type=int,default=784, help="input_dimensions")
parser.add_argument("--pruning_technique","-tech", type=int,default=0, help="prune_weights_based_on_what?")
parser.add_argument("--dimensions",type=str, help="input 6 dimensions separated by commas", default = '512,256,64,16,0,0')
parser.add_argument("--lr",type=float, default =0.001)
parser.add_argument("--batch_size",type=int, default =256)
parser.add_argument("--weights_to_prune",'-wtp', nargs="+", default=["1", "2","3","4","5", "6","7","8"])
parser.add_argument("--sigmoid", action='store_true')
parser.add_argument("--bias", default = True)
parser.add_argument("--layerwise-pruning", action='store_true')
parser.add_argument("--name", type=str, default = None)
parser.add_argument("--layer", type=str, required=True)



opt = parser.parse_args()
print(opt)

if opt.name is None:
    opt.name = 'leave_{}_{}'.format(opt.leave, opt.layer.replace('.','_'))
print(opt.name)

# notify = Notify()

# class opt:
#     leave =1
#     batch_size = 64
#     dimensions = '512,256,64,16,0,0'
#     input_dim= 784
#     sigmoid = False
#     pretrained_run = 'retraining'
#     lr=0.001
#     epoch = 10
#     weights_to_prune = ['1','2','3','4','5','6','7','8']
#     bias = True

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

dimensions = list(map(int,opt.dimensions.split(',')))
if len(dimensions)!=6:
    raise('give me 6 dimensions for autoencoder network!')

model_name = "_".join(opt.dimensions.split(','))



parameter_nums = {
    'encoder.fc1':784*512,  #784*512,
    'encoder.fc2':512*256, #512*256,
    'encoder.fc3':256*64,#256*64
    'encoder.fc4':64*16, # 64*16
    'decoder.fc4':16*64, #16*64 ,
    'decoder.fc3':64*256,#64*256
    'decoder.fc2':256*512,#256*512
    'decoder.fc1':512*784,
}

count =0

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

os.makedirs('./new_logs/{}/'.format(opt.name),exist_ok = True)

file_handler = logging.FileHandler('./new_logs/{}/logfile.log'.format(opt.name))
logger.addHandler(file_handler)

if os.path.exists(os.path.join('new_logs',opt.name, 'logfile.log')):
    x = open(os.path.join('new_logs',opt.name, 'logfile.log'),"r").readlines()
else:
    x = None


for tech in range(9):
    opt.pruning_technique = tech
    os.makedirs('./new_logs/{}/saliency_measure_{}/images'.format(opt.name, tech),exist_ok = True)

    for remaining_connection in range(1,17):
        count+=1
        if x is None or "Starting the process with remaining connection {} using tech {}!\n".format(remaining_connection, tech) not in x:
            logger.info(" ")
            logger.info("Currently running {} experiment. ".format(opt.name))
            logger.info("Starting the process with remaining connection {} using tech {}!".format(remaining_connection, tech))

            sparsity_levels={
                'encoder.fc1.weight':0,  #784*512,
                'encoder.fc2.weight':0, #512*256,
                'encoder.fc3.weight':0,#256*64
                'encoder.fc4.weight':0 , # 64*16
                'decoder.fc4.weight':0, #16*64 ,
                'decoder.fc3.weight':0,#64*256
                'decoder.fc2.weight':0,#256*512
                'decoder.fc1.weight':0,
            }

            sparsity_levels[opt.layer+'.weight'] = parameter_nums[opt.layer] - remaining_connection

            pruned_model = Fully_Connected_AE(opt.input_dim, dimensions,opt.sigmoid, opt.bias)

            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'100', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")
            print(ckpt_path)
            pruned_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

            logger.info("Loaded pretrained model")

            pruned_model = pruned_model.cuda()

            ind_recon = reconstrucion_errors(pruned_model, ind_loader)
            ood_recon = reconstrucion_errors(pruned_model, ood_loader)

            auroc = calculate_auroc(ind_recon, ood_recon)
            # print('auroc = {}'.format(auroc))
            # print('ind_recon_mean = {}'.format(torch.mean(ind_recon)))
            # print('ood_recon_mean = {}'.format(torch.mean(ood_recon)))
            orig_auroc = auroc
            orig_ind_recon = ind_recon

            logger.info("Pretrained model has AUROC / IND / OOD = {} / {} / {}".format(auroc, torch.mean(ind_recon), torch.mean(ood_recon)))

            img_grid = check_reconstructed_images(pruned_model, None, 0, 0, "after_FT", ind_loader, ood_loader, None, model_name, opt.sigmoid, None, False)
            # plt.imshow(img_grid.cpu().numpy().transpose(1,2,0))
            plt.imsave("./new_logs/{}/saliency_measure_{}/images/original_connection_{}.jpg".format(opt.name, tech, remaining_connection),img_grid.cpu().numpy().transpose(1,2,0))


            all_layers = [(pruned_model.encoder.fc1,'weight'),(pruned_model.encoder.fc2,'weight'),(pruned_model.encoder.fc3,'weight'),(pruned_model.encoder.fc4,'weight'),(pruned_model.decoder.fc4,'weight'),(pruned_model.decoder.fc3,'weight'),(pruned_model.decoder.fc2,'weight'),(pruned_model.decoder.fc1,'weight')] 

            parameters_to_prune = []

            print(opt.weights_to_prune)

            for i0 in opt.weights_to_prune:

                parameters_to_prune.append(all_layers[int(i0)-1])

            parameters_to_prune = tuple(parameters_to_prune)

            if opt.pruning_technique == 0 :

                ## weights with small final weights are pruned
                trained_weight = pruned_model.state_dict()
                
                for pruning_weight_num in opt.weights_to_prune:
                    if int(pruning_weight_num) <5 : 
                        name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                    else:
                        name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                    scores = abs(trained_weight[name])
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)

            elif opt.pruning_technique == 1 :

                ## weights with large final weights are pruned
                trained_weight = pruned_model.state_dict()
                
                for pruning_weight_num in opt.weights_to_prune:
                    if int(pruning_weight_num) <5 : 
                        name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                    else:
                        name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                    scores = 1/abs(trained_weight[name])
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)

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
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)


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
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)

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
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)
            
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
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)

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
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)
            
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
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)

            elif opt.pruning_technique == 8 :
                
                trained_weights = pruned_model.state_dict()

                ## random pruning!
                for pruning_weight_num in opt.weights_to_prune:
                    if int(pruning_weight_num) <5 : 
                        name = 'encoder.fc{}.weight'.format(pruning_weight_num)
                    else:
                        name = 'decoder.fc{}.weight'.format(9-int(pruning_weight_num))
                    scores = torch.rand_like(trained_weights[name])
                    prune.l1_unstructured(all_layers[int(pruning_weight_num)-1][0], 'weight', sparsity_levels[name],scores)                       
            
            ind_recon = reconstrucion_errors(pruned_model, ind_loader)
            ood_recon = reconstrucion_errors(pruned_model, ood_loader)
                    
            auroc = calculate_auroc(ind_recon, ood_recon)

            logger.info("Pruned model (before finetuning) has AUROC / IND / OOD = {} / {} / {}".format(auroc, torch.mean(ind_recon), torch.mean(ood_recon)))

            # print('auroc = {}'.format(auroc))
            # print('ind_recon_mean = {}'.format(torch.mean(ind_recon)))
            # print('ood_recon_mean = {}'.format(torch.mean(ood_recon)))


            img_grid = check_reconstructed_images(pruned_model, None, 0, 0, "after_FT", ind_loader, ood_loader, None, model_name, opt.sigmoid, None, False)
            plt.imsave("./new_logs/{}/saliency_measure_{}/images/before_FT_connection_{}.jpg".format(opt.name, tech, remaining_connection),img_grid.cpu().numpy().transpose(1,2,0))

            for this_layer in range(8):
                layer_sparsity = show_layer_sparsity(pruned_model, this_layer, verbose=False)
                # print(layer_sparsity)

            optimizer = torch.optim.Adam(pruned_model.parameters(), opt.lr)

            # time.sleep(5)
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


            ind_recon = reconstrucion_errors(pruned_model, ind_loader)
            ood_recon = reconstrucion_errors(pruned_model, ood_loader)
                    
            auroc = calculate_auroc(ind_recon, ood_recon)

            logger.info("Pruned model (after finetuning) has AUROC / IND / OOD = {} / {} / {}".format(auroc, torch.mean(ind_recon), torch.mean(ood_recon)))
            # notify.send("({:.2f}%) L{}/T{}/AUC {:.4f}/Loss {:.4f} for Experiment {}, sparse {}, connection {}, ".format(100*count/12, opt.leave, opt.pruning_technique, auroc, torch.mean(ind_recon),opt.name, remaining_sparsity, remaining_connection))
            logger.info("Notification sent to chrome.")

            sending_text = "Original AUC {:.4f} --> Pruned AUC {:.4f}\nOriginal IND {:.4f} --> Pruned IND {:.4f}\n\n***Details***\n - Leave out {}\n - Remaining Connections {} \n - Layer {} \n - Saliency Measure {}".format(orig_auroc, auroc, torch.mean(orig_ind_recon), torch.mean(ind_recon), opt.leave, remaining_connection, opt.layer, opt.pruning_technique)
            if auroc > orig_auroc:
                bot.sendMessage(chat_id = CHAT_ID, text=sending_text)
            
            img_grid = check_reconstructed_images(pruned_model, None, 0, 0, "after_FT", ind_loader, ood_loader, None, model_name, opt.sigmoid, None, False)
            plt.imsave("./new_logs/{}/saliency_measure_{}/images/after_FT_connection_{}.jpg".format(opt.name,tech, remaining_connection),img_grid.cpu().numpy().transpose(1,2,0))
        else:
            print("Already have the result for this experiment")

bot.sendMessage(chat_id = CHAT_ID, text="HOORAY! TRAINING DONE FOR {}!".format(opt.name))

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
parser.add_argument('--pretrained_run', type=str, help='run_name_for_pretrained_weights', default = 'alternative')
parser.add_argument('--pruning_run', type=str, help='run_name_for_pruning_procedure', default = None)
parser.add_argument('--leave', type=int, help ='leave out this class MNIST', required=True)
parser.add_argument('--epoch', type=int, help ='finetune_epoch', default=200)
parser.add_argument("--input_dim",type=int,default=784, help="input_dimensions")
parser.add_argument("--dimensions",type=str, help="input 6 dimensions separated by commas", default = '512,256,64,16,0,0')
# parser.add_argument("--lr",type=float, default =0.001)
parser.add_argument("--batch_size",type=int, default =256)
parser.add_argument("--sigmoid", action='store_true')
parser.add_argument("--bias", default = True)
parser.add_argument("--layerwise-pruning", action='store_true')
parser.add_argument("--name", type=str, default = None)
parser.add_argument("--layer", type=str, default = 'bottleneck.bottleneck')
parser.add_argument("--alternative",action='store_true')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                    help='Weight decay (default: 0.0005)')
parser.add_argument("--finetune", action='store_true')
parser.add_argument("--freeze", action='store_true')
parser.add_argument("--switching", type = int, default=None)



opt = parser.parse_args()
print(opt)

if opt.name is None:
    opt.name = 'leave_{}_{}'.format(opt.leave, opt.layer.replace('.','_'))
    if opt.finetune:
        opt.name = opt.name + '_finetune'
    if opt.freeze:
        opt.name = opt.name + '_freeze'
    if opt.switching is not None:
        opt.name = opt.name + '_switch_{}'.format(opt.switching)



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


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%   == (1-k)% is pruned. 
        out = scores.clone()
        _, idx = scores.flatten().sort()
        if isinstance(k, float):
            j = int((1 - k) * scores.numel())   # j is the number of weights being pruned. 
        else:   # if k is int, it means the number of "alive = remaining" connections
            j = scores.numel() - k   # j is the number of weights being pruned. 

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.remaining_connection = None

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.remaining_connection)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        # return x
        


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
                         shuffle=True,num_workers=8)

idx = ind_dataset.targets!=opt.leave
ind_dataset.targets = ind_dataset.targets[idx]
ind_dataset.data = ind_dataset.data[idx]

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, 
                         batch_size=opt.batch_size,
                         shuffle=False,num_workers=8)

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
                         shuffle=False,num_workers=8)

dimensions = list(map(int,opt.dimensions.split(',')))
if len(dimensions)!=6:
    raise('give me 6 dimensions for autoencoder network!')

model_name = "_".join(opt.dimensions.split(','))


# parameter_nums = {
#     'encoder.fc1':784*512,  #784*512,
#     'encoder.fc2':512*256, #512*256,
#     'encoder.fc3':256*64,#256*64
#     'encoder.fc4':64*16, # 64*16
#     'decoder.fc4':16*64, #16*64 ,
#     'decoder.fc3':64*256,#64*256
#     'decoder.fc2':256*512,#256*512
#     'decoder.fc1':512*784,
#     'bottleneck.bottleneck':16*16
# }

count =0

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

os.makedirs('./{}_supermask_logs/{}/images'.format(opt.dimensions, opt.name),exist_ok = True)

file_handler = logging.FileHandler('./{}_supermask_logs/{}/logfile.log'.format(opt.dimensions, opt.name))
logger.addHandler(file_handler)

if os.path.exists(os.path.join('{}_supermask_logs'.format(opt.dimensions),opt.name, 'logfile.log')):
    x = open(os.path.join('{}_supermask_logs'.format(opt.dimensions),opt.name, 'logfile.log'),"r").readlines()
else:
    x = None

best_auroc =0
for remaining_connection in range(1,17):
    count+=1
    if x is None or "Starting the process with remaining connection {} on bottleneck using supermask!\n".format(remaining_connection) not in x:
        logger.info(" ")
        logger.info("Currently running {} experiment. ".format(opt.name))
        logger.info("Starting the process with remaining connection {} on bottleneck using supermask!".format(remaining_connection))

        pruned_model = Fully_Connected_AE(opt.input_dim, dimensions,opt.sigmoid, opt.bias, opt.alternative)
        
        if opt.finetune:
            ckpt_name = '{}_sigmoid_{}_epoch_{}_run_{}'.format(model_name,opt.sigmoid,'100', opt.pretrained_run)
            ckpt_path = os.path.join('trained_models','pretrained','leave_out_{}'.format(opt.leave), ckpt_name + ".pth")
            print(ckpt_path)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            pruned_model.load_state_dict(state_dict)

        if opt.freeze:
            for param in pruned_model.parameters():
                param.requires_grad = False

        if opt.layer == 'bottleneck.bottleneck':
            pruned_model.bottleneck.bottleneck = SupermaskLinear(dimensions[3],dimensions[3])
            pruned_model.bottleneck.bottleneck.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.bottleneck.bottleneck.weight = state_dict['bottleneck.bottleneck.weight']
            if opt.freeze:
                pruned_model.bottleneck.bottleneck.weight.requires_grad = False

        elif opt.layer == 'encoder.fc4':
            pruned_model.encoder.fc4 = SupermaskLinear(dimensions[2],dimensions[3])
            pruned_model.encoder.fc4.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.encoder.fc4.weight.data = state_dict['encoder.fc4.weight']
            if opt.freeze:
                pruned_model.encoder.fc4.weight.requires_grad = False

        elif opt.layer == 'decoder.fc4':
            pruned_model.decoder.fc4 = SupermaskLinear(dimensions[3],dimensions[2])
            pruned_model.decoder.fc4.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.decoder.fc4.weight.data = state_dict['decoder.fc4.weight']
            if opt.freeze:
                pruned_model.decoder.fc4.requires_grad = False

        elif opt.layer == 'encoder.fc3':
            pruned_model.encoder.fc3 = SupermaskLinear(dimensions[1],dimensions[2])
            pruned_model.encoder.fc3.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.encoder.fc3.weight.data = state_dict['encoder.fc3.weight']
            if opt.freeze:
                pruned_model.encoder.fc3.weight.requires_grad = False

        elif opt.layer == 'decoder.fc3':
            pruned_model.decoder.fc3 = SupermaskLinear(dimensions[2],dimensions[1])
            pruned_model.decoder.fc3.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.decoder.fc3.weight.data = state_dict['decoder.fc3.weight']
            if opt.freeze:
                pruned_model.decoder.fc3.weight.requires_grad = False


        elif opt.layer == 'encoder.fc2':
            pruned_model.encoder.fc2 = SupermaskLinear(dimensions[0],dimensions[1])
            pruned_model.encoder.fc2.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.encoder.fc2.weight.data = state_dict['encoder.fc2.weight']
            if opt.freeze:
                pruned_model.encoder.fc2.weight.requires_grad = False


        elif opt.layer == 'decoder.fc2':
            pruned_model.decoder.fc2 = SupermaskLinear(dimensions[1],dimensions[0])
            pruned_model.decoder.fc2.remaining_connection = remaining_connection
            if opt.finetune:
                pruned_model.decoder.fc2.weight.data = state_dict['decoder.fc2.weight']
            if opt.freeze:
                pruned_model.decoder.fc2.weight.requires_grad = False


        else:
            raise("Layer is wrong.")

        logger.info("Loaded Supermask model")

        pruned_model = pruned_model.cuda()

        ind_recon = reconstrucion_errors(pruned_model, ind_loader)
        ood_recon = reconstrucion_errors(pruned_model, ood_loader)

        auroc = calculate_auroc(ind_recon, ood_recon)
        # print('auroc = {}'.format(auroc))
        # print('ind_recon_mean = {}'.format(torch.mean(ind_recon)))
        # print('ood_recon_mean = {}'.format(torch.mean(ood_recon)))
        orig_auroc = auroc
        orig_ind_recon = ind_recon

        logger.info("Original model has AUROC / IND / OOD = {} / {} / {}".format(auroc, torch.mean(ind_recon), torch.mean(ood_recon)))

        img_grid = check_reconstructed_images(pruned_model, None, 0, 0, "after_FT", ind_loader, ood_loader, None, model_name, opt.sigmoid, None, False)
        # plt.imshow(img_grid.cpu().numpy().transpose(1,2,0))
        plt.imsave("./{}_supermask_logs/{}/images/original_connection_{}.jpg".format(opt.dimensions, opt.name, remaining_connection),img_grid.cpu().numpy().transpose(1,2,0))




        
        # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!

        t = trange(1, opt.epoch + 1, desc='Bar desc', leave=True)

        for epoch in t:
            # print('epoch {} / {}'.format(epoch, opt.epoch))
            # print(epoch%opt.switching)
            if epoch%opt.switching ==1 :
                if (epoch//opt.switching)%2==0:
                    if opt.layer == 'encoder.fc4':
                        print("updating supermask saliency score")
                        pruned_model.encoder.fc4.weight.requires_grad = False
                        pruned_model.encoder.fc4.scores.requires_grad = True

                    elif opt.layer == 'bottleneck.bottleneck':
                        print("updating supermask saliency score")
                        pruned_model.bottleneck.bottleneck.weight.requires_grad = False
                        pruned_model.bottleneck.bottleneck.scores.requires_grad = True


                else:
                    if opt.layer == 'encoder.fc4':

                        print("updating supermask weight")
                        pruned_model.encoder.fc4.weight.requires_grad = True
                        pruned_model.encoder.fc4.scores.requires_grad = False
                    elif opt.layer == 'bottleneck.bottleneck':

                        print("updating supermask weight")
                        pruned_model.bottleneck.bottleneck.weight.requires_grad = True
                        pruned_model.bottleneck.bottleneck.scores.requires_grad = False

                optimizer = torch.optim.Adam(
                    [p for p in pruned_model.parameters() if p.requires_grad],
                    lr=opt.lr,
                    # momentum=opt.momentum,
                    # weight_decay=opt.wd,
                )
                
                require_grad_cnt =0
                for p in pruned_model.parameters():
                    if p.requires_grad:
                        require_grad_cnt+=1 

                print("Parameters requiring gradients : {}".format(require_grad_cnt))

            loss_list = []
            total_step=0

            pruned_model.train()
            scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch)
            iteration=0

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
                t.set_description("Average Loss {:.5f}".format(avg_loss/step))
            # scheduler.step()

        ind_recon = reconstrucion_errors(pruned_model, ind_loader)
        ood_recon = reconstrucion_errors(pruned_model, ood_loader)
                
        auroc = calculate_auroc(ind_recon, ood_recon)

        logger.info("Pruned model (after finetuning) has AUROC / IND / OOD = {} / {} / {}".format(auroc, torch.mean(ind_recon), torch.mean(ood_recon)))
        # notify.send("({:.2f}%) L{}/T{}/AUC {:.4f}/Loss {:.4f} for Experiment {}, sparse {}, connection {}, ".format(100*count/12, opt.leave, opt.pruning_technique, auroc, torch.mean(ind_recon),opt.name, remaining_sparsity, remaining_connection))
        logger.info("Notification sent to chrome.")

        sending_text = "***Best AUROC!***\nOriginal AUC {:.4f} --> Pruned AUC {:.4f}\nOriginal IND {:.4f} --> Pruned IND {:.4f}\n\n***Details***\n - Leave out {}\n - Remaining Connections {} \n - Layer {} \n - Pretrained {} \n - Alternative {} \n - Name {}".format(orig_auroc, auroc, torch.mean(orig_ind_recon), torch.mean(ind_recon), opt.leave, remaining_connection, opt.layer, opt.finetune, opt.alternative, opt.name)
        if auroc > orig_auroc and auroc > best_auroc:
            best_auroc = auroc
            bot.sendMessage(chat_id = CHAT_ID, text=sending_text)
        
        img_grid = check_reconstructed_images(pruned_model, None, 0, 0, "after_FT", ind_loader, ood_loader, None, model_name, opt.sigmoid, None, False)
        plt.imsave("./{}_supermask_logs/{}/images/after_FT_connection_{}.jpg".format(opt.dimensions, opt.name, remaining_connection),img_grid.cpu().numpy().transpose(1,2,0))
    else:
        print("Already have the result for this experiment")

bot.sendMessage(chat_id = CHAT_ID, text="HOORAY! TRAINING DONE FOR {}!".format(opt.name))

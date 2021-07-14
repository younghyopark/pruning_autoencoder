import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Linear_modified

import copy
import types


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


    
def apply_prune_mask(net, keep_masks, pick):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(
            layer, nn.Linear) and layer in pick, net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))
    
def SNIP(net, keep_ratio, train_dataloader, pick, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
#         print('**')
#         print(layer)
        
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.shape in pick:
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
            else:
                layer.weight_mask = torch.ones_like(layer.weight)
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
                layer.weight_mask.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
#             print(layer.weight.shape)
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
#     print(inputs.shape)
    inputs = inputs.reshape(-1,784).cuda(device)
    inputs.cuda(device)
#     print(inputs.shape)
    z = net.encoder(inputs)
#     print(z.shape)
    recon = net.decoder(z)
    error = (inputs - recon) ** 2
    z_norm = (z ** 2).mean()
    recon_error = error.mean()
    loss = recon_error
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.shape in pick:
                grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

#     print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return keep_masks 
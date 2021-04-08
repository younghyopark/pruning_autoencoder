"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np
import torch 

from collections import OrderedDict

import errno
import os


# Recursive mkdir
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


from sklearn.metrics import roc_auc_score
def roc_btw_arr(arr1, arr2):
    true_label = np.concatenate([np.ones_like(arr1),
                                 np.zeros_like(arr2)])
    score = np.concatenate([arr1, arr2])
    return roc_auc_score(true_label, score)


def batch_run(m, dl, device, flatten=False, method='predict', input_type='first', no_grad=True):
    """
    m: model
    dl: dataloader
    device: device
    """
    method = getattr(m, method)
    l_result = []
    for batch in dl:
        if input_type == 'first':
            x = batch[0]

        if no_grad:
            with torch.no_grad():
                if flatten:
                    x = x.view(len(x), -1)
                pred = method(x.cuda(device)).detach().cpu()
        else:
            if flatten:
                x = x.view(len(x), -1)
            pred = method(x.cuda(device)).detach().cpu()

        l_result.append(pred)
    return torch.cat(l_result)



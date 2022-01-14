import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np


def num_flat_features(x):
    size = x.size()[1:] # all dim except batch dim
    num_features = 1
    for s in size:
        num_features*=s
    return num_features

def gradient_penalty_ProGAN(model_D, real, fake, alpha, train_step, device):
    BATCH_SIZE, C, H, W = real.shape
    eta = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated = real*eta+fake*(1-eta)

    # calculate Critic (Discriminater) Scores
    prob_interpolated = model_D(interpolated, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=prob_interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty


def gradient_penalty(model_D, real, fake,device):
    BATCH_SIZE, C, H, W = real.shape
    eta = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated = real*eta+fake*(1-eta)

    # calculate Critic (Discriminater) Scores
    prob_interpolated = model_D(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=prob_interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty



def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

def initialize_weights(net):
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        
        elif isinstance(m, torch.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass
        
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            try: m.bias.data.zero_()
            except: pass

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            if np.isscalar(value):
                self.writer.add_scalar(key, value)
                self._data.total[key] += value * n
                self._data.counts[key] += n
                self._data.average[key] = self._data.total[key] / self._data.counts[key]
            else:
                self._data.total[key] += value 
                self._data.counts[key] += 1
                self._data.average[key] += value

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

'''
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
'''
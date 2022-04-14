import torch
import torch.nn
import itertools
import model
import random
import numpy as np

from model import Model
import os
import yaml

if __name__ == '__main__':
    cfg = {}
    
    with open('./configs/cfg.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    
    # same env
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed']) # if use multi-GPU
    
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    model = Model(cfg)
    
    
    output = model.predict('text')
    
    
    
    
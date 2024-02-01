import os
import sys
# os.chdir('../../')
print(os.getcwd())
sys.path.append('./')
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import pggf.pfg as pfg
import pggf.pggf_model as pggfm
import pggf.path as pggfp
import torch

torch.manual_seed(41)

for i in range(50):
    # mean = 10*torch.rand(4, 8)-5
    # mean = 0.5*torch.randn(4, 8)
    val = 1.
    mean = torch.tensor([[val, 0., 0., 0., 0., 0., 0., 0.],
                         [0., -val, 0., 0., 0., 0., 0., 0.],
                         [0., 0., val, 0., 0., 0., 0., 0.],
                         [0., 0., 0., -val, 0., 0., 0., 0.]])
    torch.save(mean, f'./mean_{i}.ts')
    # ratio = torch.exp(torch.randn(4))
    ratio = torch.exp(torch.randn(4))
    ratio /= torch.sum(ratio)
    torch.save(ratio, f'./ratio_{i}.ts')
    print(mean)
    print(ratio)
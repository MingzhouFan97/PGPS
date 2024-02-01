import os
import sys
# os.chdir('../../')
print(os.getcwd())
sys.path.append('./')
import torch
import math
import torch.nn as nn
import pggf.pfg as pfg
import pggf.pggf_model as pggfm
import pggf.path as pggfp
import torch
import math
import torch.nn as nn
import pggf.path as pggfp
import pggf.pggf_model as pggfm
import torch.distributions as D


torch.manual_seed(41)

def Langevian_step(x, grad, step_size):
    x = x.detach()
    x = x + step_size * grad + math.sqrt(2*step_size)*torch.randn_like(grad)
    return x

def exp_ratio(samples_cat, means):
    return torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < 1.) / sample_num for i in range(4)])

if __name__ == "__main__":
    device='cpu'
    lrrs = []
    for i in range(30):
        # means = torch.eye(10) + torch.randn(10)
        means = torch.load(f'./experiment/high_dim_mix/mean_{i}.ts')
        P0 = D.MultivariateNormal(torch.zeros(8), 1**2*torch.eye(8))
        ratio = torch.load(f'./experiment/high_dim_mix/ratio_{i}.ts')
        mix = D.Categorical(ratio)
        comp = D.Independent(D.Normal(means, torch.ones(4, 8) * 0.15), 1)
        P1 = D.MixtureSameFamily(mix, comp)


        sample_num = 500
        samples = P1.sample([sample_num])
        exp_ratio_iid = exp_ratio(samples, means)
        print(torch.norm(ratio - exp_ratio_iid))

        max_iter = int(4e3)
        #
        samples_cat = P0.sample([sample_num])
        #
        step_size = torch.tensor(5e-4)


        def adaptive_step_path(path, name):
            lrr = []
            path = path(P0, P1, device)
            Ker = pfg.RBF()
            pggf = pfg.SVGD(path, Ker, sample_num, device=device)
            iteration = 0
            for j in range(max_iter):
                print(f'Iteration {j + 1}, start at t={pggf.t.item():.5f}')
                pggf.step(step_size)
                iteration += 1
            samples = pggf.X.detach().squeeze()
            exp_ratio_iid = exp_ratio(samples, means)
            print(torch.norm(ratio - exp_ratio_iid))
            torch.save(samples.clone().detach(), f'./experiment/high_dim_mix/samples_{name}_{i}.ts')
            torch.save(iteration, f'./experiment/high_dim_mix/iteration_{name}_{i}.ts')
            return 0


        path = pggfp.ExpPath
        adaptive_step_path(path, 'svgd')

import os
import sys
# os.chdir('../../')
print(os.getcwd())
sys.path.append('./')
import torch
# import matplotlib.pyplot as plt
import math
import torch.nn as nn
import pggf.path as pggfp
import pggf.pggf_model as pggfm
import torch.distributions as D


torch.manual_seed(42)


class Net_X(torch.nn.Module):
    def __init__(self, x_dim, width):
        super().__init__()
        self.il = nn.Linear(x_dim, width)
        self.act = nn.Sigmoid()
        self.ol = nn.Linear(width, x_dim)

    def forward(self, X):
        x_1 = self.act(self.il(X))
        x_2 = (x_1 * (1 - x_1)).unsqueeze(2)
        mat = self.il.weight * self.ol.weight.t()
        return self.ol(x_1), torch.sum(x_2 * mat.unsqueeze(0), dim=[1, 2])


def exp_ratio(samples_cat, means):
    # print(torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < .9) / samples_num for i in range(4)]))
    return torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < 1) / samples_num for i in range(4)])#/torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < .9) / samples_num for i in range(4)]).sum() if torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < .9) / samples_num for i in range(4)]).sum() > 0 else 1
if __name__ == "__main__":
    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    print(device)
    for i in range(30):
        if device == 'cpu':
            means = torch.load(f'mean_{i}.ts').to(device)
            ratio = torch.load(f'ratio_{i}.ts').to(device)
        else:
            means = torch.load(f'./experiment/high_dim_mix/mean_{i}.ts').to(device)
            ratio = torch.load(f'./experiment/high_dim_mix/ratio_{i}.ts').to(device)
        P0 = D.MultivariateNormal(torch.zeros(8).to(device), 1 ** 2 * torch.eye(8).to(device))

        mix = D.Categorical(ratio.to(device))
        comp = D.Independent(D.Normal(means.to(device), torch.ones(4, 8).to(device) * 0.15), 1)
        P1 = D.MixtureSameFamily(mix, comp)
        samples_num = 1000
        iid_samples = P1.sample([samples_num])
        exp_ratio_iid = exp_ratio(iid_samples, means)
        # print(ratio)
        # print(exp_ratio_iid)

        step_size = 1e-4

        def adaptive_step_path(size, path, name, exp_ratio_iid, adjust_steps=0, adj_delta=0.01):
            adj_t = adj_delta
            path = path(P0, P1, device)
            net = Net_X(8, 128)
            net.to(device)
            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-1)
            pggf = pggfm.PGGF_2(path, net, optimizer, samples_num, device=device)
            iteration = 0
            for j in range(10000):
                print(f'Iteration {j + 1}, start at t={pggf.t.item():.5f}')
                pggf.prepare_training()
                for k in range(20001):
                    v = pggf.train_one_iter()
                    if not k%1000:
                        print(k, v)
                    if v < 100:
                        optimizer.param_groups[0]['lr'] = 1e-2
                    if v < .1:
                        break
                pggf.adaptive_step(size)
                samples = pggf.X.detach()
                exp_ratio_pggf = exp_ratio(samples, means)
                print(i, exp_ratio_pggf, torch.norm(exp_ratio_pggf - exp_ratio_iid))
                iteration += 1
                if pggf.t >= adj_t and adjust_steps > 0:
                    for _ in range(adjust_steps):
                        pggf.Langevin_adjust(step_size)
                        iteration += 1
                    adj_t += adj_delta
                    print('ADJ!')
                if pggf.done:
                    break
            for _ in range(10*adjust_steps):
                pggf.Langevin_adjust(step_size)
                iteration += 1
            samples = pggf.X.detach()
            if device == 'cpu':
                torch.save(samples.clone().detach(), f'./samples_{name}.ts')
                torch.save(iteration, f'./iteration_{name}.ts')
            else:
                torch.save(samples.clone().detach(), f'./experiment/high_dim_mix/samples_{name}_{i}.ts')
                torch.save(iteration, f'./experiment/high_dim_mix/iteration_{name}_{i}.ts')
            # exp_ratio_pggf = exp_ratio(samples, means)
            # print(exp_ratio_iid)
            # print(i, exp_ratio_pggf, torch.norm(exp_ratio_pggf - exp_ratio_iid))
            return samples, iteration
        adj_steps = 100
        # path = lambda p0, p1, device: pggfp.ExpTelePath(p0, p1, device, 0.5)
        path = pggfp.ExpPath
        # adaptive_step_path(torch.tensor(0.1), path, f'tel_p5_p1_adj_{adj_steps}', exp_ratio_iid, adjust_steps=adj_steps, adj_delta=0.01)
        # adaptive_step_path(torch.tensor(0.05), path, f'tel_p5_p05_adj_{adj_steps}', exp_ratio_iid, adjust_steps=adj_steps, adj_delta=0.01)
        adaptive_step_path(torch.tensor(0.01), path, f'exp_p01_adj_{adj_steps}', exp_ratio_iid, adjust_steps=adj_steps, adj_delta=0.01)
        adaptive_step_path(torch.tensor(0.005), path, f'exp_p005_adj_{adj_steps}', exp_ratio_iid, adjust_steps=adj_steps, adj_delta=0.01)
        # adaptive_step_path(torch.tensor(0.001), path, f'tel_p4_p001_adj_{adj_steps}', exp_ratio_iid, adjust_steps=adj_steps, adj_delta=0.01)
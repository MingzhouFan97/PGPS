# %%

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as D
import torch.optim as optim
import time
import os
from scipy.optimize import minimize
import math
import matplotlib
import pggf.pggf_model as pggfm
import pggf.path as pggfp
import pggf.pfg as pfgm
import torch
import seaborn as sns
# %%

torch.manual_seed(42)
font = {'weight': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
# %%

P0 = D.MultivariateNormal(2 * torch.tensor([0., 0]), torch.tensor([[.1, 0.], [0, .1]]))
mix = D.Categorical(torch.ones(2, ))
comp = D.Independent(D.Normal(
    torch.tensor([[1.5, 0], [1, .0]]),
    torch.ones(2, 2) * 0.05), 1)
P1 = D.MixtureSameFamily(mix, comp)


# %%

class LangenvianMonteCarlo():
    def __init__(self, p):
        self.p = p

    def one_step(self, X, dt):
        X = X.detach().requires_grad_(True)
        log_l = self.p.log_prob(X)
        score_funcs = torch.autograd.grad(log_l.sum(), X)[0].detach()
        X = X + dt * score_funcs + math.sqrt(2 * dt) * torch.randn_like(X)
        return X


# %%
#
X1 = P1.sample(torch.Size([200]))
fig = plt.figure(figsize=(5, 3))
ax = fig.add_subplot(1, 1, 1)
helper = torch.tensor(
    [[0.7, -0.3], [0.7, -0.3], [1.6, -0.3], [1.8, -0.3], [0.7, 0.3], [0.7, 0.3], [1.6, 0.3], [1.8, 0.3]])
X_plot = torch.cat([helper, X1], dim=0)
sns.kdeplot(x=X_plot[:, 0], y=X_plot[:, 1], fill=True, thresh=0, levels=100, cmap="Blues")
ax.scatter(X1[:, 0], X1[:, 1], marker='x', label='sample', c='r', alpha=1., linewidths=0.8)
ax.tick_params(left=False, bottom=False)
plt.xlim([0.75, 1.75])
plt.ylim([-0.3, 0.3])

plt.savefig('./independent')
plt.show()
#
# # %%
#
steps = 4001
langenvian = LangenvianMonteCarlo(P1)
X = P0.sample(torch.Size([200]))
for i in range(steps):
    if not i % 100:
        with torch.no_grad():
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_subplot(1, 1, 1)

            helper = torch.tensor([[0.7, -0.3], [0.7, -0.3], [1.6, -0.3], [1.8, -0.3], [0.7, 0.3], [0.7, 0.3], [1.6, 0.3], [1.8, 0.3]])
            X_plot = torch.cat([helper, X], dim=0)
            sns.kdeplot(x=X_plot[:, 0], y=X_plot[:, 1], fill=True, thresh=0, levels=100, cmap="Blues")
            ax.scatter(X[:, 0], X[:, 1], marker='x', label='sample', c='r', alpha=1., linewidths=0.8)
            ax.tick_params(left=False, bottom=False)
            # for r in [0.1, 0.2, 0.3]:
            #     for j in range(2):
            #         circle = plt.Circle(comp.mean[j], r, color='r', fill=False)
            #         ax.add_patch(circle)
            plt.xlim([0.75, 1.75])
            plt.ylim([-0.3, 0.3])
            # plt.title(f'Perform LD for {i} iterations')
            plt.savefig(f'./LD{i}iterations')
            plt.show()
    X = langenvian.one_step(X, 0.001)


# %%

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


# %%

path = pggfp.ExpTelePath(P0, P1, base=0.1)
net = Net_X(2, 64)
f = torch.tensor(.1, requires_grad=True)
optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-1)
pggf = pggfm.PGGF_2(path, net, optimizer, 200)

# %%

ts = []
iterations = 0
for j in range(1000):
    print(j, pggf.t)
    pggf.prepare_training()
    for _ in range(1000):
        v = pggf.train_one_iter()
        # print(v)
    pggf.adaptive_step(0.05)
    iterations += 1
    ts.append(pggf.t.detach().clone().item())
    for _ in range(10):
        pggf.Langevin_adjust(0.001)
        iterations += 1
    with torch.no_grad():
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(1, 1, 1)
        helper = torch.tensor(
            [[0.7, -0.3], [0.7, -0.3], [1.6, -0.3], [1.8, -0.3], [0.7, 0.3], [0.7, 0.3], [1.6, 0.3], [1.8, 0.3]])
        X_plot = torch.cat([helper, pggf.X], dim=0)
        sns.kdeplot(x=X_plot[:, 0], y=X_plot[:, 1], fill=True, thresh=0, levels=100, cmap="Blues")
        ax.scatter(pggf.X[:, 0], pggf.X[:, 1], marker='x', label='sample', c='r', alpha=1., linewidths=0.8)
        ax.tick_params(left=False, bottom=False)
        plt.xlim([0.75, 1.75])
        plt.ylim([-0.3, 0.3])
        # plt.savefig(f'./pggf_scatter')
        # plt.title('Final Samples')
        plt.show()
    if pggf.done:
        break
print('The end')
print(iterations)
#

with torch.no_grad():
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    helper = torch.tensor(
        [[0.7, -0.3], [0.7, -0.3], [1.6, -0.3], [1.8, -0.3], [0.7, 0.3], [0.7, 0.3], [1.6, 0.3], [1.8, 0.3]])
    X_plot = torch.cat([helper, pggf.X], dim=0)
    sns.kdeplot(x=X_plot[:, 0], y=X_plot[:, 1], fill=True, thresh=0, levels=100, cmap="Blues")
    ax.scatter(pggf.X[:, 0], pggf.X[:, 1], marker='x', label='sample', c='r', alpha=1., linewidths=0.8)
    ax.tick_params(left=False, bottom=False)
    plt.xlim([0.75, 1.75])
    plt.ylim([-0.3, 0.3])
    plt.savefig(f'./pggf_scatter')
    # plt.title('Final Samples')
    plt.show()


plt.plot(torch.arange(len(ts) + 1), [0.] + ts)
plt.ylabel('t')
plt.xlabel('Iteration')
plt.savefig(f'./pggf_time')
plt.show()
#

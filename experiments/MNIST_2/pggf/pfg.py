import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as D
import torch.optim as optim
from . import util
import time
import torch.autograd as autograd

def divergence_bf(dx, y):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

class PFG():
    def __init__(self, path, net, optimizer, sample_num, device='cpu'):
        self.path = path
        self.net = net
        self.optim = optimizer
        self.t = torch.tensor(0.)
        self.X = self.path.sample_P0(sample_num).to(device)

    def step(self, dt):
        self.X += dt * self.net(self.X)
        self.t += dt.detach()

    def prepare_training(self):
        X = self.X.detach().requires_grad_(True)
        logt = self.path.log_prob(torch.tensor(1.), X)
        self.score_func = torch.autograd.grad(logt.sum(), X)[0].detach()

    def train_one_iter(self):
        X = self.X.detach().requires_grad_(True)
        St = self.net(X)
        div = divergence_bf(St, X)
        A = torch.sum(self.score_func * St, dim=1) + div
        A = A.unsqueeze(1)
        self.net.train()
        self.optim.zero_grad()
        loss = torch.mean(-A + torch.norm(St, dim=1)**2)
        loss.backward()
        loss_v = loss.detach().item()
        self.optim.step()
        return loss_v


class PFG_2():
    def __init__(self, path, net, optimizer, sample_num, device='cpu'):
        self.path = path
        self.net = net
        self.optim = optimizer
        self.t = torch.tensor(0.)
        self.X = self.path.sample_P0(sample_num).to(device)

    def step(self, dt):
        St, _ = self.net(self.X)
        self.X += dt * St
        self.t += dt.detach()

    def prepare_training(self):
        X = self.X.detach().requires_grad_(True)
        self.score_func = self.path.P1.score(X).detach()

    def train_one_iter(self):
        X = self.X.detach().requires_grad_(True)
        St, div = self.net(X)
        A = torch.sum(self.score_func * St, dim=1) + div
        A = A.unsqueeze(1)
        self.net.train()
        self.optim.zero_grad()
        loss = torch.mean(-A + torch.norm(St, dim=1)**2)
        loss.backward()
        loss_v = loss.detach().item()
        self.optim.step()
        return loss_v

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach()
            h = torch.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = torch.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

    def dK(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach()
            h = torch.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = torch.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()  # bX x bY

        X_Y = X.view(X.shape[0], 1, X.shape[1]) - Y.view(1, Y.shape[0], Y.shape[1])  # bX x bY x d
        # print(X_Y.shape,K_XY.shape)
        dKXY = torch.sum(K_XY.view(K_XY.shape[0], K_XY.shape[1], 1) * (-gamma) * X_Y, 0).detach()

        return dKXY


    # Let us initialize a reusable instance right away.



    # %%

class SVGD:
    def __init__(self, path, K, sample_num, device='cpu'):
        self.path = path
        self.t = torch.tensor(0.)
        self.X = self.path.sample_P0(sample_num).to(device)
        self.K = K

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob = self.path.log_prob(torch.tensor(1.), X)
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = self.K.dK(X.detach(), X)
        # -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return phi

    def step(self, dt):
        self.X += dt * self.phi(self.X)
        self.t += dt.detach()



class SVGD_2:
    def __init__(self, path, K, sample_num, device='cpu'):
        self.path = path
        self.t = torch.tensor(0.)
        self.X = self.path.sample_P0(sample_num).to(device)
        self.K = K

    def phi(self, X):
        X = X.detach().requires_grad_(True)
        score_func = self.path.P1.score(X).detach()

        K_XX = self.K(X, X.detach())
        grad_K = self.K.dK(X.detach(), X)
        # -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return phi

    def step(self, dt):
        self.X += dt * self.phi(self.X)
        self.t += dt.detach()
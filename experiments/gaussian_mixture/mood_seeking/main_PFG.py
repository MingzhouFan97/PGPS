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

torch.manual_seed(42)

class Net_X(torch.nn.Module):
    def __init__(self, x_dim, width):
        super().__init__()
        self.il = nn.Linear(x_dim, width)
        self.act = nn.Sigmoid()
        self.ol = nn.Linear(width, x_dim)

    def forward(self, X):
        x_1 = self.act(self.il(X))
        return self.ol(x_1)

def KDE(samples, x):
    sig = 0.5*torch.std(samples).to(device)*samples.shape[0]**(-0.2)
    normals = [torch.distributions.Normal(s_mean, sig) for s_mean in samples]
    probs = torch.cat([torch.exp(normal.log_prob(x)).unsqueeze(0) for normal in normals], dim=0)
    prob = torch.mean(probs, dim=0)
    return prob


class mixture():
    def __init__(self, weight, mu1, mu2, std1, std2):
        self.weight = weight
        self.p1 = torch.distributions.Normal(mu1, std1)
        self.p2 = torch.distributions.Normal(mu2, std2)

    def log_prob(self, X):
        return torch.log(
            self.weight[0] * torch.exp(self.p1.log_prob(X)) + self.weight[1] * torch.exp(self.p2.log_prob(X)))

    def sample(self, num):
        samples_1 = self.p1.sample([round(num[0] * self.weight[0])])
        samples_2 = self.p2.sample([round(num[0] * self.weight[1])])
        samples_cat = torch.cat([samples_1, samples_2]).unsqueeze(1)
        return samples_cat

if __name__ == "__main__":
    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    print(device)

    m1 = 0.
    m2 = 8.

    tar_w1 = 0.5
    tar_weight = [tar_w1, 1 - tar_w1]

    P0 = torch.distributions.Normal(0., 3.)
    P1 = mixture([tar_w1, 1 - tar_w1], m1, m2, 1., 1.)

    samples_num = 5000


    step_size = torch.tensor(1e-2)

    samples_cat = P0.sample([samples_num]).to(device)
    x = 20 * torch.arange(1001).to(device)/1000 - 5

    def adaptive_step_path(path, name):
        lrr = []
        path = path(P0, P1, device)
        net = Net_X(1, 64)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        pggf = pfg.PFG(path, net, optimizer, samples_num, device=device)
        iteration = 0
        for j in range(5000):
            print(f'Iteration {j + 1}, start at t={pggf.t.item():.5f}')
            pggf.prepare_training()
            if not j:
                for k in range(1000):
                    pggf.train_one_iter()
            else:
                for k in range(20):
                    pggf.train_one_iter()
            pggf.step(step_size)
            iteration += 1
            samples = pggf.X.detach().squeeze()
            lrr.append(torch.sum(samples>5)/samples_num)
        samples = pggf.X.detach().squeeze()
        probs = KDE(samples, x)
        if device == 'cpu':
            torch.save(probs, './probs_PFG.ts')
            torch.save(torch.tensor(lrr), './lrr_PFG.ts')
        else:
            torch.save(probs, './experiment/mode_seeking_toy/probs_PFG.ts')
            torch.save(torch.tensor(lrr), './experiment/mode_seeking_toy/lrr_PFG.ts')

        return probs, samples, iteration


    path = pggfp.ExpPath
    adaptive_step_path(path, 'PFG')

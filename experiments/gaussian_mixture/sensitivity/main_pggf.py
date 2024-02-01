import os
import sys
# os.chdir('../../')
print(os.getcwd())
sys.path.append('./')
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import pggf.path as pggfp
import pggf.pggf_model as pggfm

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
    sig = 1.06*torch.std(samples).to(device)*samples.shape[0]**(-0.2)
    normals = [torch.distributions.Normal(s_mean, sig) for s_mean in samples]
    probs = torch.cat([torch.exp(normal.log_prob(x)).unsqueeze(0) for normal in normals], dim=0)
    prob = torch.mean(probs, dim=0)
    return prob


def Langevian_step(x, grad, step_size):
    x = x.detach()
    x = x + step_size * grad + math.sqrt(2*step_size)*torch.randn_like(grad)
    return x

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

    m1 = -5.
    m2 = 5.

    tar_w1 = 0.001
    tar_weight = [tar_w1, 1 - tar_w1]

    P0 = torch.distributions.Normal(0., 2.)
    P1 = mixture([tar_w1, 1 - tar_w1], m1, m2, 1., 1.)

    samples_num = 5000


    step_size = 1e-2

    samples_cat = P0.sample([samples_num]).to(device)
    x = 20 * torch.arange(1001).to(device)/1000 - 10

    def adaptive_step_path(size, path, name, adjust_steps=0):
        path = path(P0, P1, device)
        net = Net_X(1, 64)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        pggf = pggfm.PGGF(path, net, optimizer, samples_num, device=device)
        iteration = 0
        for j in range(10000):
            print(f'Iteration {j + 1}, start at t={pggf.t.item():.5f}')
            pggf.prepare_training()
            for k in range(1000):
                v = pggf.train_one_iter()
                if v < 1e-2:
                    break
            pggf.adaptive_step(size)
            iteration += 1
            for _ in range(adjust_steps):
                pggf.Langevin_adjust(step_size)
                iteration += 1
            if device == 'cpu':
                samples = pggf.X.detach().squeeze()
                probs = KDE(samples, x)
                plt.plot(torch.arange(probs.shape[0]), probs)
                plt.show()
            if pggf.done:
                break
        samples = pggf.X.detach().squeeze()
        probs = KDE(samples, x)
        if device == 'cpu':
            torch.save(samples.clone().detach(), f'./samples_{name}.ts')
            torch.save(probs.clone().detach(), f'./probs_{name}.ts')
            torch.save(iteration, f'./iteration_{name}.ts')
        else:
            torch.save(samples.clone().detach(), f'./experiment/weight_reserving_toy/samples_{name}.ts')
            torch.save(probs.clone().detach(), f'./experiment/weight_reserving_toy/probs_{name}.ts')
            torch.save(iteration, f'./experiment/weight_reserving_toy/iteration_{name}.ts')
        return probs, samples, iteration

    adj_steps = 0

    if device == 'cuda':
        path = pggfp.ExpPath
        adaptive_step_path(torch.tensor(1.), path, 'exp_1', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.5), path, 'exp_p5', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.1), path, 'exp_p1', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.05), path, 'exp_p05', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.01), path, 'exp_p01', adjust_steps=adj_steps)

        path = pggfp.ExpTelePath
        adaptive_step_path(torch.tensor(1.), path, 'tel_1', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.5), path, 'tel_p5', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.1), path, 'tel_p1', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.05), path, 'tel_p05', adjust_steps=adj_steps)
        adaptive_step_path(torch.tensor(.01), path, 'tel_p01', adjust_steps=adj_steps)
    else:
        path = pggfp.ExpTelePath
        adaptive_step_path(torch.tensor(.1), path, 'tel_p1', adjust_steps=adj_steps)
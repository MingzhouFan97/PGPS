import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import pggf.path as pggfp
import pggf.pggf_model as pggfm

torch.manual_seed(41)

def KDE(samples, x):
    # sig = 1.06*torch.std(samples)*samples.shape[0]**(-0.2)
    sig = .5*torch.std(samples)*samples.shape[0]**(-0.2)
    normals = [torch.distributions.Normal(s_mean, sig) for s_mean in samples]
    probs = torch.cat([torch.exp(normal.log_prob(x)).unsqueeze(0) for normal in normals], dim=0)
    prob = torch.mean(probs, dim=0)
    return prob


def Langevian_step(x, grad, step_size):
    x = x.detach()
    x = x + step_size * grad + math.sqrt(2*step_size)*torch.randn_like(grad)
    return x

if __name__ == "__main__":
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


    tar_w1 = 0.5
    tar_weight = [tar_w1, 1 - tar_w1]
    m1 = 0.
    m2 = 8.
    P0 = torch.distributions.Normal(0, 3)
    P1 = mixture([tar_w1, 1 - tar_w1], m1, m2, 1., 1.)

    samples_num = 5000


    step_size = 1e-2
    max_iter = int(1e4)

    samples_cat = P0.sample([samples_num])
    x = 20 * torch.arange(1001)/1000 - 5

    lrr = []
    for iter in range(max_iter):
        if not iter % 1000:
            print(iter)
        samples_cat = samples_cat.detach().requires_grad_(True)
        logp = P1.log_prob(samples_cat)
        grad_mix = torch.autograd.grad(logp.sum(), samples_cat)[0]
        samples_cat = Langevian_step(samples_cat, grad_mix, step_size)

        lrr.append(torch.sum(samples_cat>=5)/samples_num)
    probs = KDE(samples_cat, x)
    torch.save(probs, './probs_LD.ts')
    torch.save(torch.tensor(lrr), './lrr_LD.ts')
import torch
import matplotlib.pyplot as plt
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
    # print(torch.tensor(
    #     [torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < 1.5) / sample_num for i in range(4)]))
    return torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < .8) / sample_num for i in range(4)])#/torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < 1.8) / sample_num for i in range(4)]).sum() if torch.tensor([torch.sum(torch.norm(samples_cat - means[i:i + 1, :], dim=1) < 1.5) / sample_num for i in range(4)]).sum() > 0 else 1

if __name__ == "__main__":
    for sig in [1.]:
        lrrs = []
        for i in range(30):
            means = torch.load(f'mean_{i}.ts')
            P0 = D.MultivariateNormal(torch.zeros(8), sig ** 2 * torch.eye(8))
            ratio = torch.load(f'ratio_{i}.ts')
            ratio /= torch.sum(ratio)
            mix = D.Categorical(ratio)
            comp = D.Independent(D.Normal(means, torch.ones(4, 8) * 0.15), 1)
            P1 = D.MixtureSameFamily(mix, comp)


            sample_num = 1000
            samples = P1.sample([sample_num])
            exp_ratio_iid = exp_ratio(samples, means)
            # print(exp_ratio_iid)
            # print(ratio)
            # print(torch.norm(ratio - exp_ratio_iid))

            step_size = 1e-4
            max_iter = int(5e3)
            #
            samples_cat = P0.sample([sample_num])
            #
            lrr = []
            for iter in range(max_iter):
                samples_cat = samples_cat.detach().requires_grad_(True)
                logp = P1.log_prob(samples_cat)
                grad_mix = torch.autograd.grad(logp.sum(), samples_cat)[0]
                samples_cat = Langevian_step(samples_cat, grad_mix, step_size)
                torch.save(samples_cat.clone().detach(), f'./samples_LD_{i}.ts')
                    # torch.save(iteration, f'./iteration_{name}.ts')
                    # exp_ratio_LD = exp_ratio(samples_cat, means)
                    # print(exp_ratio_iid)
                    # print(exp_ratio_LD, torch.norm(exp_ratio_iid - exp_ratio_LD))
                    # lrr.append(torch.norm(exp_ratio_iid - exp_ratio_LD))
                lrrs.append(lrr)
                exp_ratio_LD = exp_ratio(samples_cat, means)
                if not iter % 1000:
                    print(exp_ratio_iid)
                    print(i, iter, exp_ratio_LD, torch.norm(exp_ratio_iid - exp_ratio_LD))

        # lrrs = torch.tensor(lrrs)
        # plt.plot(torch.arange(lrrs.shape[1]), lrrs.mean(dim=0))
        # plt.fill_between(torch.arange(lrrs.shape[1]), lrrs.mean(dim=0) + 1.96*lrrs.std(dim=0), lrrs.mean(dim=0) - 1.96*lrrs.std(dim=0), alpha=0.4)
        # plt.show()

    # torch.save(torch.tensor(lrr), './lrr_LD.ts')
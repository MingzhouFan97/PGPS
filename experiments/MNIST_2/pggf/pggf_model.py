import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as D
import torch.optim as optim
from . import util
import time

def divergence_bf(dx, y):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()





# class PGGF_continious():
#     def __init__(self, path, net, optimizer, num_step, forward_step_size=0.01):
#         self.path = path
#         self.net = net
#         self.optim = optimizer
#         self.num_step = num_step
#         self.forward_step_size = forward_step_size
#
#     def phi(self, X, t):
#         phi = self.net(X, t)
#         return phi
#
#     def step(self, X, t, dt, mask=None):
#         if mask is None:
#             mask = torch.ones_like(t)
#         X += mask*dt * self.phi(X, t)
#         return X
#
#     def forward(self, X, t_goal):
#         X = X.detach().clone()
#         t_cur = torch.zeros_like(t_goal)
#         for i in range(int(1/self.forward_step_size)):
#             X = self.step(X, t_cur, self.forward_step_size, mask=(t_cur<t_goal).float())
#             t_cur += self.forward_step_size*torch.ones_like(t_goal)
#         return X.detach()
#
#
#     def train_one_iter_fw_0(self, sample_num, t_train):
#         X0 = self.path.sample_P0(sample_num*t_train.shape[0])
#         X0 = X0.detach().requires_grad_(True)
#         if len(X0.shape) == 1:
#             X0 = X0.unsqueeze(-1)
#         t = (t_train.unsqueeze(-1)*torch.ones([1, sample_num])).reshape([-1, 1])
#         St = self.net(X0, t)
#         l2 = divergence_bf(St, X0)
#         logt = self.path.log_prob(t, X0)
#         score_func = torch.autograd.grad(logt.sum(), X0)[0].detach()
#         Xt = self.forward(X0, t)
#         logtdt = self.path.derivative_t_log_prob(t, Xt).unsqueeze(1)
#         means = torch.mean(logtdt.view(sample_num, -1), dim=0, keepdim=True).T
#         acg_st = (means * torch.ones([1, sample_num])).reshape([-1, 1])
#         self.net.train()
#         self.optim.zero_grad()
#
#         l1 = logtdt.detach() - acg_st.detach()
#
#         l3 = torch.sum(score_func * St, dim=1)
#         loss_temp = l1.squeeze()+l2+l3
#         loss = torch.mean(torch.abs(loss_temp))
#         loss.backward()
#         loss_v = loss.detach().item()
#         self.optim.step()
#         # print(loss_v, self.optim.param_groups[0]['lr'])
#         return loss_v
#
#     def train_one_iter_is_0(self, sample_num, t_train):
#         X0 = self.path.sample_P0(sample_num*t_train.shape[0])
#         X0 = X0.detach().requires_grad_(True)
#         if len(X0.shape) == 1:
#             X0 = X0.unsqueeze(-1)
#         t = (t_train.unsqueeze(-1)*torch.ones([1, sample_num])).reshape([-1, 1])
#         St = self.net(X0, t)
#         l2 = divergence_bf(St, X0)
#         logt = self.path.log_prob(t, X0)
#         score_func = torch.autograd.grad(logt.sum(), X0)[0].detach()
#         logtdt = self.path.derivative_t_log_prob(t, X0).unsqueeze(1)
#         w = torch.exp(self.path.log_prob(t, X0).squeeze() - self.path.P0.log_prob(X0).squeeze()).unsqueeze(1)
#         sum_weighted = torch.sum((logtdt * w).view(sample_num, -1), dim=0, keepdim=True).T
#         sum_w = torch.sum(w.view(sample_num, -1), dim=0, keepdim=True).T
#         acg_st = (sum_weighted/sum_w*torch.ones([1, sample_num])).reshape([-1, 1])
#         self.net.train()
#         self.optim.zero_grad()
#
#         l1 = logtdt.detach() - acg_st.detach()
#         # l1 = logtdt.detach() + (4.5*(1/(9*t+1) + (10*t/(9*t+1))**2) - 100*t/(9*t+1))
#
#         l3 = torch.sum(score_func * St, dim=1)
#         loss_temp = l1.squeeze()+l2+l3
#         loss = torch.mean(torch.abs(loss_temp))
#         loss.backward()
#         loss_v = loss.detach().item()
#         self.optim.step()
#         # print(loss_v, self.optim.param_groups[0]['lr'])
#         return loss_v
#
#     def train_one_iter_t(self, sample_num, t_train):
#         pass

class PGGF():
    def __init__(self, path, net, optimizer, sample_num, device='cpu'):
        self.path = path
        self.net = net.to(device)
        self.optim = optimizer
        self.t = torch.tensor(0.).to(device)
        self.X = self.path.sample_P0(sample_num).to(device)
        self.done = False

    def step(self, dt):
        X = self.X.detach().requires_grad_(True)
        grad_x = self.net(X)
        self.X += dt * grad_x
        self.t += dt.detach()
        if self.t >= 1.:
            self.done = True

    def prepare_training(self):
        X = self.X.detach().requires_grad_(True)
        logt = self.path.log_prob(self.t, X)
        self.score_func = torch.autograd.grad(logt.sum(), X)[0].detach()
        self.B = self.path.derivative_t_log_prob(self.t, X).unsqueeze(1).detach()
        self.B_mean = torch.mean(self.B).detach()

    def train_one_iter(self):
        X = self.X.detach().requires_grad_(True)
        St = self.net(X)
        div = divergence_bf(St, X)
        A = torch.sum(self.score_func * St, dim=1) + div
        A = A.unsqueeze(1)
        self.net.train()
        self.optim.zero_grad()
        loss = torch.mean((A + self.B - self.B_mean)**2)
        loss.backward()
        loss_v = loss.detach().item()
        self.optim.step()
        return loss_v

    def step_size_decider(self, epsilon):
        return epsilon/torch.mean(torch.norm(self.net(self.X), p=2, dim=1))

    def adaptive_step(self, epsilon, maximum_step=torch.tensor(0.02)):
        proposed_dt = self.step_size_decider(epsilon)
        proposed_dt = torch.min(torch.tensor([1 - self.t, proposed_dt, maximum_step]))
        self.step(proposed_dt)


    def Langevin_step(self, step_size):
        self.X += self.score_func * step_size + torch.randn_like(self.X) * np.sqrt(2*step_size)

    def Langevin_adjust(self, step_size):
        self.prepare_training()
        self.Langevin_step(step_size)




class PGGF_2():
    def __init__(self, path, net, optimizer, sample_num, device='cpu'):
        self.path = path
        self.net = net.to(device)
        self.optim = optimizer
        self.t = torch.tensor(0.).to(device)
        self.X = self.path.sample_P0(sample_num).to(device)
        self.device = device
        self.done = False

    def step(self, dt):
        X = self.X.detach().requires_grad_(True)
        grad_x = self.net(X)[0]
        self.X += dt * grad_x
        self.t += dt.detach()
        if self.t >= 1.:
            self.done = True

    def prepare_training(self, NN=False):
        if not NN:
            X = self.X.detach().requires_grad_(True)
            logt = self.path.log_prob(self.t, X)
            self.score_func = torch.autograd.grad(logt.sum(), X)[0].detach()
            self.B = self.path.derivative_t_log_prob(self.t, X).unsqueeze(1).detach()
            self.B_mean = torch.mean(self.B).detach()
        else:
            X = self.X.detach().requires_grad_(True)
            self.score_func = self.path.score_function(self.t, X)
            self.B = self.path.derivative_t_log_prob(self.t, X).unsqueeze(1).detach()
            self.B_mean = torch.mean(self.B).detach()

    def train_one_iter(self):
        X = self.X.detach().requires_grad_(True)
        St, div = self.net(X)
        # div = divergence_bf(St, X)
        A = torch.sum(self.score_func * St, dim=1) + div
        A = A.unsqueeze(1)
        self.net.train()
        self.optim.zero_grad()
        loss = torch.mean((A + self.B - self.B_mean)**2)
        loss.backward()
        loss_v = loss.detach().item()
        self.optim.step()
        return loss_v

    def step_size_decider(self, epsilon):
        return epsilon/torch.mean(torch.norm(self.net(self.X)[0], p=2, dim=1))

    def adaptive_step(self, epsilon, maximum_step=torch.tensor(0.02)):
        proposed_dt = self.step_size_decider(epsilon)
        proposed_dt = torch.min(torch.tensor([1 - self.t, proposed_dt, maximum_step]))
        self.step(proposed_dt)


    def Langevin_step(self, step_size):
        self.X += self.score_func * step_size + torch.randn_like(self.X) * np.sqrt(2*step_size)

    def Langevin_adjust(self, step_size, NN=False):
        self.prepare_training(NN)
        self.Langevin_step(step_size)



# class PGGFNN(PGGF):
#     pass
#
# # class PGGF_inference(PGGF):
# #     def __init__(self, model_type, net, likelihood, prior, path_type, optimizer, sample_num, train_x, train_y, prior_std=1.):
# #
# #         self.model = model_type
# #         para_vec = util.get_para_vector(model_type())
# #         if prior == 'Gaussian':
# #             prior = D.MultivariateNormal(torch.zeros(para_vec.shape[0]), prior_std * torch.eye(para_vec.shape[0]))
# #         self.posterior = util.ModelPosterior(prior, likelihood)
# #         super().__init__(path_type(prior, self.posterior), net, optimizer, sample_num)
# #         self.train_x = train_x
# #         self.train_y = train_y
# #     def prepare_training(self):
# #         logt = self.path.log_prob(self.t, X)
# #         self.score_func = torch.autograd.grad(logt.sum(), X)[0].detach()
# #         self.B = self.path.derivative_t_log_prob(self.t, X).unsqueeze(1).detach()
# #         self.B_mean = torch.mean(self.B).detach()
# #
# #
# #
#
#
# #
# class PGGF_inference():
#     def __init__(self, model, phi, likelihood, prior, path_type, optimizer, sample_num, train_x, train_y, prior_std=1., device='cpu'):
#         self.model = model
#         self.phi = phi
#         if prior == 'Gaussian':
#             prior = D.MultivariateNormal(torch.zeros(model.para_dim).to(device), prior_std * torch.eye(model.para_dim).to(device))
#         self.posterior = util.ModelPosterior(prior, likelihood)
#         self.path = path_type(prior, self.posterior)
#         self.device = device
#         self.pggf = PGGF(self.path, self.phi, optimizer, sample_num, device=device)
#         self.train_x = train_x
#         self.train_y = train_y
#         self.t_trajec = [0.]
#
#     def train_phi(self, training_iters, scilent):
#         self.pggf.prepare_training()
#         original_lr = self.pggf.optim.param_groups[0]['lr']
#         prev_v_best = torch.tensor(float('inf'))
#         unchanged = 0
#         for k in range(training_iters):
#             v = self.pggf.train_one_iter()
#             if v < prev_v_best:
#                 prev_v_best = v
#                 unchanged = 0
#             else:
#                 unchanged += 1
#             if unchanged > 50:
#                 self.pggf.optim.param_groups[0]['lr'] *= 0.1
#                 prev_v_best = v
#                 unchanged = 0
#             if (v < 1e-4) or (self.pggf.optim.param_groups[0]['lr'] < 1e-6):
#                 self.pggf.optim.param_groups[0]['lr'] = original_lr
#                 break
#             if not scilent:
#                 if not k % 10:
#                     print(k, v, unchanged, self.pggf.optim.param_groups[0]['lr'])
#         self.pggf.optim.param_groups[0]['lr'] = original_lr
#         if k == training_iters - 1:
#             print(f'Warning: training did not converge after {training_iters} iterations!')
#         return prev_v_best
#
#     def perform_one_step(self, training_iters, epsilon, adaptive=True, adjust_iter=10, scilent=False):
#         best_v = self.train_phi(training_iters, scilent)
#         if adaptive:
#             self.pggf.adaptive_step(epsilon)
#         else:
#             self.pggf.step(torch.tensor(epsilon))
#         for _ in range(adjust_iter):
#             self.pggf.Langevin_adjust(0.0001)
#         return best_v
#
#     def perform_one_step_free_langevian(self, training_iters, epsilon, adaptive=True, scilent=False):
#         best_v = self.train_phi(training_iters, scilent)
#         self.pggf.Langevin_step(0.0001)
#         if adaptive:
#             self.pggf.adaptive_step(epsilon)
#         else:
#             self.pggf.step(torch.tensor(epsilon))
#         return best_v
#
#     def perform(self, max_steps, training_iters, epsilon, adjust_iter=5, scilent=False, adaptive=True):
#         t0 = time.time()
#         for j in range(max_steps):
#             self.perform_one_step(training_iters, epsilon, adjust_iter=adjust_iter, scilent=scilent, adaptive=adaptive)
#             print(f'After iter {j+1},\t current time {self.pggf.t:.5f}, \t {time.time()-t0:.1f} seconds used.')
#             self.t_trajec.append(self.pggf.t.detach().item())
#             if self.pggf.done or (self.pggf.t >= 1. - 1e-6):
#                 break
#
#     def perform_free_langevian(self, max_steps, training_iters, epsilon, scilent=False, adaptive=True):
#         for j in range(max_steps):
#             self.perform_one_step_free_langevian(training_iters, epsilon, scilent=scilent, adaptive=adaptive)
#             print(f'After iter {j+1},\t current time {self.pggf.t:.5f}')
#             self.t_trajec.append(self.pggf.t.detach().item())
#             if self.pggf.done or (self.pggf.t >= 1. - 1e-6):
#                 break
#
#
# class PGGF_inference_NN():
#     def __init__(self, model, phi, likelihood, prior, path_type, optimizer, sample_num, train_x, train_y, prior_std=1., device='cpu'):
#         self.model = model
#         self.phi = phi
#         self.posterior = util.NNPosterior(likelihood)
#         self.path = path_type(prior, self.posterior)
#         self.device = device
#         self.pggf = PGGF(self.path, self.phi, optimizer, sample_num, device=device)
#         self.train_x = train_x
#         self.train_y = train_y
#         self.t_trajec = [0.]
#
#     def train_phi(self, training_iters, scilent):
#         self.pggf.prepare_training()
#         original_lr = self.pggf.optim.param_groups[0]['lr']
#         prev_v_best = torch.tensor(float('inf'))
#         unchanged = 0
#         for k in range(training_iters):
#             v = self.pggf.train_one_iter()
#             if v < prev_v_best:
#                 prev_v_best = v
#                 unchanged = 0
#             else:
#                 unchanged += 1
#             if unchanged > 50:
#                 self.pggf.optim.param_groups[0]['lr'] *= 0.1
#                 prev_v_best = v
#                 unchanged = 0
#             if (v < 1e-4) or (self.pggf.optim.param_groups[0]['lr'] < 1e-6):
#                 self.pggf.optim.param_groups[0]['lr'] = original_lr
#                 break
#             if not scilent:
#                 if not k % 10:
#                     print(k, v, unchanged, self.pggf.optim.param_groups[0]['lr'])
#         self.pggf.optim.param_groups[0]['lr'] = original_lr
#         if k == training_iters - 1:
#             print(f'Warning: training did not converge after {training_iters} iterations!')
#         return prev_v_best
#
#     def perform_one_step(self, training_iters, epsilon, adaptive=True, adjust_iter=10, scilent=False):
#         best_v = self.train_phi(training_iters, scilent)
#         if adaptive:
#             self.pggf.adaptive_step(epsilon)
#         else:
#             self.pggf.step(torch.tensor(epsilon))
#         for _ in range(adjust_iter):
#             self.pggf.Langevin_adjust(0.0001)
#         return best_v
#
#     def perform_one_step_free_langevian(self, training_iters, epsilon, adaptive=True, scilent=False):
#         best_v = self.train_phi(training_iters, scilent)
#         self.pggf.Langevin_step(0.0001)
#         if adaptive:
#             self.pggf.adaptive_step(epsilon)
#         else:
#             self.pggf.step(torch.tensor(epsilon))
#         return best_v
#
#     def perform(self, max_steps, training_iters, epsilon, adjust_iter=5, scilent=False, adaptive=True):
#         t0 = time.time()
#         for j in range(max_steps):
#             self.perform_one_step(training_iters, epsilon, adjust_iter=adjust_iter, scilent=scilent, adaptive=adaptive)
#             print(f'After iter {j+1},\t current time {self.pggf.t:.5f}, \t {time.time()-t0:.1f} seconds used.')
#             self.t_trajec.append(self.pggf.t.detach().item())
#             if self.pggf.done or (self.pggf.t >= 1. - 1e-6):
#                 break
#
#     def perform_free_langevian(self, max_steps, training_iters, epsilon, scilent=False, adaptive=True):
#         for j in range(max_steps):
#             self.perform_one_step_free_langevian(training_iters, epsilon, scilent=scilent, adaptive=adaptive)
#             print(f'After iter {j+1},\t current time {self.pggf.t:.5f}')
#             self.t_trajec.append(self.pggf.t.detach().item())
#             if self.pggf.done or (self.pggf.t >= 1. - 1e-6):
#                 break
#
#
# class PGGF_grad():
#     def __init__(self, path, sample_num, device='cpu'):
#         self.path = path
#         self.t = torch.tensor(0.)
#         self.X = self.path.sample_P0(sample_num).to(device)
#         self.done = False
#
#     def step(self, dt):
#         grad_x = self.score_func*self.grad_k()
#         self.X += dt * grad_x
#         self.t += dt.detach()
#
#     def prepare_training(self):
#         X = self.X.detach().requires_grad_(True)
#         logt = self.path.log_prob(self.t, X)
#         def logt_x(X):
#             return torch.sum(self.path.log_prob(self.t, X))
#         self.score_func = torch.autograd.grad(logt.sum(), X)[0]
#
#         self.laplace_func = torch.cat([torch.sum(torch.diag(torch.autograd.functional.hessian(logt_x, X[k:k+1, :]).squeeze())).unsqueeze(0) for k in range(X.shape[0])], dim=0).unsqueeze(1)
#         self.B = self.path.derivative_t_log_prob(self.t, X).unsqueeze(1).detach()
#         self.B_mean = torch.mean(self.B).detach()
#
#     def grad_k(self):
#         aaa = self.B_mean - self.B
#         bbb = torch.sum(self.score_func**2, dim=1, keepdim=True) + self.laplace_func
#         return aaa/bbb
#
#     def Langevin_step(self, step_size):
#         self.X += self.score_func * step_size + torch.randn_like(self.X) * np.sqrt(step_size)
#
#     def Langevin_adjust(self, step_size):
#         self.prepare_training()
#         self.Langevin_step(step_size)

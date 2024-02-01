import torch
from . import pggf_model as pggfm
# import matplotlib.pyplot as plt
import torch.nn as nn
from . import pfg

init_var = .5

def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def set_grads(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.grad = vector[offset:offset + param.numel()].reshape(param.size()).clone().to(device)
        offset += param.numel()

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def get_par_vec(base_model):
    para_list = []
    for para in base_model.parameters():
            para_list.append(para)
    return flatten(para_list)

def get_grad_vec(base_model):
    para_list = []
    for para in base_model.parameters():
            para_list.append(para.grad)
    return flatten(para_list)


class BinaryLikelihood():
    def __init__(self, train_x, train_y, base_model, base_model_para, num, device=None, stochastic=True):
        self.train_x = train_x
        self.train_y = train_y
        self.model_list = torch.nn.ModuleList([base_model(*base_model_para) for _ in range(num)])
        self.device = device
        self.CEL = torch.nn.CrossEntropyLoss(reduction='sum')
        self.stochastic = stochastic
    def log_prob(self, X):
        likelihoods = []
        # print("@@@@@@@@@@@@@@@@@@", X.device)
        for i, model in enumerate(self.model_list):
            # print(self.device)
            set_weights(self.model_list[i], X[i,:], device=self.device)
            # print("$$$$$$$$$$$$$$$$$$$", self.train_x.device)
            self.model_list[i].to(self.device)
            if self.stochastic and self.train_x.shape[0] > 100:
                indices = torch.randperm(self.train_x.shape[0])[:100]
                train_x = self.train_x[indices]
            else:
                indices = torch.arange(self.train_x.shape[0])
                train_x = self.train_x[indices]
            pred_i = self.model_list[i](train_x)
            likelihoods.append(-self.CEL(pred_i, self.train_y[indices]).unsqueeze(0))
        return torch.cat(likelihoods)
    def score(self, X):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!", X.device)
        log_ps = self.log_prob(X).sum()
        log_ps.backward()
        scores = []
        for i, model in enumerate(self.model_list):
            scores.append(get_grad_vec(model).unsqueeze(0))
        return torch.cat(scores, dim=0)


class GaussianLikelihood():
    def __init__(self, train_x, train_y, base_model, base_model_para, num, device=None, stochastic=True):
        self.train_x = train_x
        self.train_y = train_y
        self.model_list = torch.nn.ModuleList([base_model(*base_model_para) for _ in range(num)])
        self.device = device
        self.MSE = torch.nn.MSELoss(reduction='sum')
        self.stochastic = stochastic
    def log_prob(self, X):
        likelihoods = []
        # print("@@@@@@@@@@@@@@@@@@", X.device)
        for i, model in enumerate(self.model_list):
            # print(self.device)
            set_weights(self.model_list[i], X[i,:], device=self.device)
            # print("$$$$$$$$$$$$$$$$$$$", self.train_x.device)
            self.model_list[i].to(self.device)
            if self.stochastic and self.train_x.shape[0] > 100:
                indices = torch.randperm(self.train_x.shape[0])[:100]
                train_x = self.train_x[indices]
            else:
                indices = torch.arange(self.train_x.shape[0])
                train_x = self.train_x[indices]
            pred_i = self.model_list[i](train_x)
            likelihoods.append(-self.MSE(pred_i, self.train_y[indices]).unsqueeze(0))
        return torch.cat(likelihoods)
    def score(self, X):
        # print("!!!!!!!!!!!!!!!!!!!!!!!!", X.device)
        log_ps = self.log_prob(X).sum()
        log_ps.backward()
        scores = []
        for i, model in enumerate(self.model_list):
            scores.append(get_grad_vec(model).unsqueeze(0))
        return torch.cat(scores, dim=0)

class Posterior():
    def __init__(self, P0, Likelihood):
        self.P0 = P0
        self.Likelihood = Likelihood
    def log_prob(self, X):
        # print("!!!!!!!!!!!!!!!!!!!", X.device)
        return self.P0.log_prob(X) + self.Likelihood.log_prob(X)

        # return self.Likelihood.log_prob(X)

    def score(self, X):
        X = X.detach().requires_grad_(True)
        # print("!!!!!!!!!!!!!!!!!!!", X.device, self.P0.loc.device)
        logt = self.P0.log_prob(X)
        score_0 = torch.autograd.grad(logt.sum(), X)[0].detach()
        return score_0 + self.Likelihood.score(X)

        # return self.Likelihood.score(X)


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

class Net_X_plain(torch.nn.Module):
    def __init__(self, x_dim, width):
        super().__init__()
        self.il = nn.Linear(x_dim, width)
        self.act = nn.Sigmoid()
        self.ol = nn.Linear(width, x_dim)

    def forward(self, X):
        x_1 = self.act(self.il(X))
        return self.ol(x_1)


class Inference():
    def __init__(self, base_model, model_paras, train_x, train_y, num, device=None, classification=True):
        model_illu = base_model(*model_paras)
        vec = get_par_vec(model_illu)
        self.dim = vec.shape[0]
        self.P0 = torch.distributions.MultivariateNormal(torch.zeros_like(vec).to(device), init_var*torch.eye(vec.shape[0]).to(device))
        if classification:
            self.Likelihood = BinaryLikelihood(train_x, train_y, base_model, model_paras, num, device)
        else:
            self.Likelihood = GaussianLikelihood(train_x, train_y, base_model, model_paras, num, device)
        self.P1 = Posterior(self.P0, self.Likelihood)
        self.num = num
        self.device=device

    def performe(self, size, path, adj_size=1e-3, adjust_steps=0, adj_delta=0.01, device='cpu', max_iter=100000, final_adj=0, max_train_iter=10000):
            adj_t = adj_delta
            path_i = path(self.P0, self.P1, device=self.device)
            net = Net_X(self.dim, 128)
            net.to(device)
            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-3)
            pggf = pggfm.PGGF_2(path_i, net, optimizer, self.num, device=self.device)
            iteration = 0
            for j in range(max_iter):
                # print(f'Iteration {j + 1}, start at t={pggf.t.item():.5f}')
                pggf.prepare_training(NN=True)
                for k in range(max_train_iter):
                    v = pggf.train_one_iter()
                    # if not k%10:
                    #     print(k, v)
                    if v < .01:
                        break
                # print(self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                pggf.adaptive_step(size)
                # pggf.step(size)
                # print(self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                iteration += 1
                if pggf.t >= adj_t and adjust_steps > 0:
                    for k in range(adjust_steps):
                        pggf.Langevin_adjust(adj_size, NN=True)
                        # print(k, '!!!!!!!!', self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                        iteration += 1
                    adj_t += adj_delta
                    # for k in range(adjust_steps):
                    #     pggf.t = torch.tensor(1.)
                    #     pggf.Langevin_adjust(adj_size, NN=True)
                    #     print(k, self.P1.log_prob(pggf.X).mean(), self.P1.log_prob(pggf.X).std())
                    #     iteration += 1
                #     print('ADJ!')
                #     print(self.P1.log_prob(pggf.X))
                # print(self.P1.log_prob(pggf.X))
                if pggf.done:
                    for k in range(final_adj):
                        pggf.Langevin_adjust(adj_size, NN=True)
                        # print(k, '!!!!!!!!', self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                        iteration += 1
                    break


            for i, model in enumerate(self.P1.Likelihood.model_list):
                set_weights(self.P1.Likelihood.model_list[i], pggf.X[i, :], device=device)

            return self.P1.Likelihood.model_list




class Inference_SVGD():
    def __init__(self, base_model, model_paras, train_x, train_y, num, device=None, classification=True):
        model_illu = base_model(*model_paras)
        vec = get_par_vec(model_illu)
        self.dim = vec.shape[0]
        self.P0 = torch.distributions.MultivariateNormal(torch.zeros_like(vec).to(device), init_var*torch.eye(vec.shape[0]).to(device))
        if classification:
            self.Likelihood = BinaryLikelihood(train_x, train_y, base_model, model_paras, num, device)
        else:
            self.Likelihood = GaussianLikelihood(train_x, train_y, base_model, model_paras, num, device)
        self.P1 = Posterior(self.P0, self.Likelihood)
        self.num = num
        self.device=device

    def performe(self, path, adj_size=1e-3, device='cpu', max_iter=100000):
            path_i = path(self.P0, self.P1, device=self.device)
            Ker = pfg.RBF()
            pggf = pfg.SVGD_2(path_i, Ker, self.num, device=device)
            iteration = 0
            for j in range(max_iter):
                pggf.step(adj_size)
                # print(j, self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                iteration += 1
            for i, model in enumerate(self.P1.Likelihood.model_list):
                set_weights(self.P1.Likelihood.model_list[i], pggf.X[i, :], device=device)

            return self.P1.Likelihood.model_list


class Inference_PFG():
    def __init__(self, base_model, model_paras, train_x, train_y, num, device=None, classification=True):
        model_illu = base_model(*model_paras)
        vec = get_par_vec(model_illu)
        self.dim = vec.shape[0]
        # print(device)
        self.P0 = torch.distributions.MultivariateNormal(torch.zeros_like(vec).to(device), init_var*torch.eye(vec.shape[0]).to(device))
        if classification:
            self.Likelihood = BinaryLikelihood(train_x, train_y, base_model, model_paras, num, device)
        else:
            self.Likelihood = GaussianLikelihood(train_x, train_y, base_model, model_paras, num, device)
        self.P1 = Posterior(self.P0, self.Likelihood)
        self.num = num
        self.device=device

    def performe(self, path, adj_size=1e-3, device='cpu', max_iter=100000):
            path_i = path(self.P0, self.P1, device=self.device)
            net = Net_X(self.dim, 128)
            net.to(device)
            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-3)
            pggf = pfg.PFG_2(path_i, net, optimizer, self.num, device=device)
            iteration = 0
            for j in range(max_iter):
                # print(f'Iteration {j + 1}, start at t={pggf.t.item():.5f}')
                pggf.prepare_training()
                if not j:
                    for k in range(1000):
                        aaa = pggf.train_one_iter()
                        # print(aaa)
                for k in range(5):
                    pggf.prepare_training()
                    v = pggf.train_one_iter()
                # print(self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                pggf.step(adj_size)
                # pggf.step(size)
                # print(self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                iteration += 1
            for i, model in enumerate(self.P1.Likelihood.model_list):
                set_weights(self.P1.Likelihood.model_list[i], pggf.X[i, :], device=device)

            return self.P1.Likelihood.model_list


class Inference_guided_LD():
    def __init__(self, base_model, model_paras, train_x, train_y, num, device=None, classification=True):
        model_illu = base_model(*model_paras)
        vec = get_par_vec(model_illu)
        self.dim = vec.shape[0]
        self.P0 = torch.distributions.MultivariateNormal(torch.zeros_like(vec).to(device), init_var*torch.eye(vec.shape[0]).to(device))
        if classification:
            self.Likelihood = BinaryLikelihood(train_x, train_y, base_model, model_paras, num, device)
        else:
            self.Likelihood = GaussianLikelihood(train_x, train_y, base_model, model_paras, num, device)
        self.P1 = Posterior(self.P0, self.Likelihood)
        self.num = num
        self.device=device

    def performe(self, size, path, adj_size=1e-3, adjust_steps=0, adj_delta=0.01, device='cpu', max_iter=100000, final_adj=0, max_train_iter=10000):
            adj_t = adj_delta
            path_i = path(self.P0, self.P1, device=self.device)
            net = Net_X(self.dim, 128)
            net.to(device)
            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-3)
            pggf = pggfm.PGGF_2(path_i, net, optimizer, self.num, device=self.device)
            iteration = 0
            for j in range(max_iter):
                pggf.t += size
                # print(pggf.t, self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                iteration += 1
                for k in range(adjust_steps):
                    pggf.Langevin_adjust(adj_size, NN=True)
                    # print(k, '!!!!!!!!', self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                    iteration += 1
                adj_t += adj_delta

                if pggf.t >= .999:
                    for k in range(final_adj):
                        pggf.Langevin_adjust(adj_size, NN=True)
                        # print(k, '!!!!!!!!', self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                        iteration += 1
                    break


            for i, model in enumerate(self.P1.Likelihood.model_list):
                set_weights(self.P1.Likelihood.model_list[i], pggf.X[i, :], device=device)

            return self.P1.Likelihood.model_list


class Inference_LD():
    def __init__(self, base_model, model_paras, train_x, train_y, num, device=None, classification=True):
        model_illu = base_model(*model_paras)
        vec = get_par_vec(model_illu)
        self.dim = vec.shape[0]
        self.P0 = torch.distributions.MultivariateNormal(torch.zeros_like(vec).to(device), init_var*torch.eye(vec.shape[0]).to(device))
        if classification:
            self.Likelihood = BinaryLikelihood(train_x, train_y, base_model, model_paras, num, device)
        else:
            self.Likelihood = GaussianLikelihood(train_x, train_y, base_model, model_paras, num, device)
        self.P1 = Posterior(self.P0, self.Likelihood)
        self.num = num

    def performe(self, size, path, adj_size=1e-3, adjust_steps=0, adj_delta=0.01, device='cpu', final_adj=0):
            adj_t = adj_delta
            path_i = path(self.P0, self.P1)
            net = Net_X(self.dim, 128)
            net.to(device)
            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-4)
            pggf = pggfm.PGGF_2(path_i, net, optimizer, self.num, device=device)
            iteration = 0

            for i in range(adjust_steps):
                pggf.t = torch.tensor(1.)
                pggf.Langevin_adjust(adj_size, NN=True)
                # print(i, self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                iteration += 1


            for i, model in enumerate(self.P1.Likelihood.model_list):
                set_weights(self.P1.Likelihood.model_list[i], pggf.X[i, :], device=device)

            return self.P1.Likelihood.model_list

class Inference_SGLD():
    def __init__(self, base_model, model_paras, train_x, train_y, num, device=None, classification=True):
        model_illu = base_model(*model_paras)
        vec = get_par_vec(model_illu)
        self.dim = vec.shape[0]
        self.P0 = torch.distributions.MultivariateNormal(torch.zeros_like(vec).to(device), init_var*torch.eye(vec.shape[0]).to(device))
        if classification:
            self.Likelihood = BinaryLikelihood(train_x, train_y, base_model, model_paras, 1, device)
        else:
            self.Likelihood = GaussianLikelihood(train_x, train_y, base_model, model_paras, 1, device)
        self.P1 = Posterior(self.P0, self.Likelihood)
        self.num = num
        self.base_model = base_model
        self.model_paras = model_paras

    def performe(self, size, path, adj_size=1e-3, adjust_steps=0, adj_delta=0.01, device='cpu', final_adj=0):
            adj_t = adj_delta
            path_i = path(self.P0, self.P1)
            net = Net_X(self.dim, 128)
            net.to(device)
            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-4)
            pggf = pggfm.PGGF_2(path_i, net, optimizer, 1, device=device)
            iteration = 0
            pggf.t = torch.tensor(1.)
            for i in range(adjust_steps):
                pggf.Langevin_adjust(adj_size, NN=True)
                # print(i, self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                iteration += 1
            final_X = []
            for i in range(self.num):
                pggf.Langevin_adjust(adj_size, NN=True)
                # print(i, self.P1.Likelihood.log_prob(pggf.X).mean(), self.P1.Likelihood.log_prob(pggf.X).std())
                final_X.append(pggf.X.detach().clone())
                iteration += 1
            Xs = torch.cat(final_X, dim=0)
            out_models = torch.nn.ModuleList([self.base_model(*self.model_paras) for _ in range(self.num)])
            for out_model in out_models:
                out_model.to(device)
            for i, model in enumerate(out_models):
                set_weights(out_models[i], Xs[i, :], device=device)

            return out_models


import torch

class Path():
    def __init__(self, P0, P1):
        self.P0 = P0
        self.P1 = P1
    def log_prob(self, t, X):
        raise NotImplementedError

    def derivative_t_log_prob(self, t, X):
        raise NotImplementedError

    def sample_P0(self, size):
        p0s = self.P0.sample(torch.Size([size, ]))
        if len(p0s.shape) == 1:
            p0s = p0s.unsqueeze(-1)
        return p0s

class PathNN():
    def __init__(self, P1):
        self.P1 = P1
    def log_prob(self, t, X):
        raise NotImplementedError

    def derivative_t_log_prob(self, t, X):
        raise NotImplementedError


class ExpPath(Path):
    def __init__(self, P0, P1, device='cpu'):
        super().__init__(P0, P1)
    def log_prob(self, t, X):
        return t.squeeze()*self.P1.log_prob(X).squeeze() + (1-t).squeeze()*self.P0.log_prob(X).squeeze()

    def derivative_t_log_prob(self, t, X):
        return self.P1.log_prob(X).squeeze() - self.P0.log_prob(X).squeeze()

    def score_function(self, t, X):
        X = X.detach().requires_grad_(True)
        P0_score = torch.autograd.grad(self.P0.log_prob(X).sum(), X)[0].detach()
        P1_score = torch.autograd.grad(self.P1.log_prob(X).sum(), X)[0].detach()
        return t.squeeze() * P1_score + (1 - t).squeeze() * P0_score

    def derivative_t_log_prob_NN(self, t, X):
        logp1 = self.P1.log_prob(X)
        logp0 = self.P0.log_prob(X)
        if logp1.shape[0] == 1:
            return logp1 - logp0
        else:
            return logp1.squeeze() - logp0.squeeze()

class ExpPathNN(Path):
    def __init__(self, P0, P1, device='cpu'):
        super().__init__(P0, P1)
    def log_prob(self, t, X):
        return t.squeeze()*self.P1.log_prob(X).squeeze() + (1-t).squeeze()*self.P0.log_prob(X).squeeze()
    def score_function(self, t, X):
        X = X.detach().requires_grad_(True)
        P0_score = torch.autograd.grad(self.P0.log_prob(X).sum(), X)[0].detach()
        P1_score = self.P1.score(X)
        return t.squeeze() * P1_score + (1 - t).squeeze() * P0_score
    def derivative_t_log_prob(self, t, X):
        logp1 = self.P1.log_prob(X)
        logp0 = self.P0.log_prob(X)
        if logp1.shape[0] == 1:
            return logp1 - logp0
        else:
            return logp1.squeeze() - logp0.squeeze()
    def sample_P0(self, size):
        p0s = self.P0.sample(torch.Size([size, ]))
        if len(p0s.shape) == 1:
            p0s = p0s.unsqueeze(-1)
        return p0s


class ExpTelePath(Path):
    def __init__(self, P0, P1, device='cpu', base=0.8, center=0., alpha=0):
        super().__init__(P0, P1)
        # self.base = torch.tensor(base)
        # self.center = torch.tensor(center)
        self.base = base
        self.center = center
        self.device = device
        self.alpha=alpha
        # self.base.to(device)
        # self.center.to(device)
    def log_prob(self, t, X):
        return (1 - t).squeeze() * self.P0.log_prob(X * (1 - self.alpha*t)).squeeze() \
               + t.squeeze() * self.P1.log_prob(X / (self.base + (1 - self.base) * t)).squeeze()

    def derivative_t_log_prob(self, t, X):
        if t.shape == torch.Size([]):
            t = t.item()*torch.ones([X.shape[0], 1], device=self.device)
        t = t.detach().requires_grad_(True)
        logpt = self.log_prob(t, X)
        dlogdt = torch.autograd.grad(logpt.sum(), t)[0].detach()
        return dlogdt.squeeze()

    def derivative_t_log_prob_NN(self, t, X):
        # if t.shape == torch.Size([]):
        #     t = t.item()*torch.ones([X.shape[0], 1], device=self.device)
        X_0 = ((1 - self.alpha * t) * X).detach().requires_grad_(True)
        X_1 = (X / (self.base + (1 - self.base) * t)).detach().requires_grad_(True)
        P0_score = torch.autograd.grad(self.P0.log_prob(X_0).sum(), X_0)[0].detach()
        P1_score = torch.autograd.grad(self.P1.log_prob(X_1).sum(), X_1)[0].detach()
        dlogdt = -self.P0.log_prob(X_0) + self.P1.log_prob(X_1) - self.alpha*(1-t)*torch.sum(X*P0_score, dim=1) - (1-self.base)*t/(self.base + (1-self.base)*t)**2*torch.sum(X*P1_score, dim=1)
        return dlogdt.squeeze()

    def score_function(self, t, X):
        # X = X.detach().requires_grad_(True)
        # print(X.device)
        X_0 = ((1-self.alpha*t)*X).detach().requires_grad_(True)
        X_1 = (X/(self.base + (1-self.base)*t)).detach().requires_grad_(True)
        # print(X_0.device)
        # print('!!!!!!!!!!!!!!!!', X_1.device)
        # print("ZZZZZZZZZZZZZZZ", self.P0.log_prob(X_0).sum().device)
        P0_score = torch.autograd.grad(self.P0.log_prob(X_0).sum(), X_0)[0].detach()
        P1_score = torch.autograd.grad(self.P1.log_prob(X_1).sum(), X_1)[0].detach()
        return (t/(self.base + (1-self.base) * t)).squeeze() * P1_score + ((1-self.alpha*t)*(1 - t)).squeeze() * P0_score


class ExpTelePathNN(Path):
    def __init__(self, P0, P1, device='cpu', base=0.5, center=0., alpha=0.):
        super().__init__(P0, P1)
        # self.base = torch.tensor(base)
        # self.center = torch.tensor(center)
        self.base = base
        self.center = center
        self.device = device
        self.alpha=alpha
        # self.base.to(device)
        # self.center.to(device)
    def log_prob(self, t, X):
        return (1 - t).squeeze() * self.P0.log_prob(X * (1 - self.alpha*t)).squeeze() \
               + t.squeeze() * self.P1.log_prob(X / (self.base + (1 - self.base) * t)).squeeze()
    def derivative_t_log_prob(self, t, X):
        # if t.shape == torch.Size([]):
        #     t = t.item()*torch.ones([X.shape[0], 1], device=self.device)
        X_0 = ((1 - self.alpha * t) * X).detach().requires_grad_(True)
        X_1 = (X / (self.base + (1 - self.base) * t)).detach().requires_grad_(True)
        P0_score = torch.autograd.grad(self.P0.log_prob(X_0).sum(), X_0)[0].detach()
        P1_score = self.P1.score(X_1)
        dlogdt = -self.P0.log_prob(X_0) + self.P1.log_prob(X_1) - self.alpha*(1-t)*torch.sum(X*P0_score, dim=1) - (1-self.base)*t/(self.base + (1-self.base)*t)**2*torch.sum(X*P1_score, dim=1)
        return dlogdt.squeeze()

    def score_function(self, t, X):
        # X = X.detach().requires_grad_(True)
        # print(X.device)
        X_0 = ((1-self.alpha*t)*X).detach().requires_grad_(True)
        X_1 = (X/(self.base + (1-self.base)*t)).detach().requires_grad_(True)
        # print(X_0.device)
        # print('!!!!!!!!!!!!!!!!', X_1.device)
        # print("ZZZZZZZZZZZZZZZ", self.P0.log_prob(X_0).sum().device)
        P0_score = torch.autograd.grad(self.P0.log_prob(X_0).sum(), X_0)[0].detach()
        P1_score = self.P1.score(X_1)
        return (t/(self.base + (1-self.base) * t)).squeeze() * P1_score + ((1-self.alpha*t)*(1 - t)).squeeze() * P0_score


# class SumPath(Path):
#     def __init__(self, P0, P1, base=0.2):
#         super().__init__(P0, P1)
#         self.base = base
#
#     def log_prob(self, t, X):
#         return torch.log(t.squeeze() * torch.exp(self.P1.log_prob(X).squeeze()) + (1 - t).squeeze() * torch.exp(self.P0.log_prob(X).squeeze()))
#
#     def derivative_t_log_prob(self, t, X):
#         if t.shape == torch.Size([]):
#             t = t.item()*torch.ones([X.shape[0], 1])
#         t = t.detach().requires_grad_(True)
#         logpt = self.log_prob(t, X)
#         dlogdt = torch.autograd.grad(logpt.sum(), t)[0].detach()
#         return dlogdt.squeeze()
#
#
#
# class ExpPathNN(PathNN):
#     def __init__(self, P1):
#         super().__init__(P1)
#
#     def log_prob(self, t, X):
#         return t.squeeze()*self.P1.log_prob(X).squeeze()
#
#     def derivative_t_log_prob(self, t, X):
#         return self.P1.log_prob(X).squeeze()
#
#
# class ExpTelePathNN(Path):
#     def __init__(self, P1, base=0.2):
#         super().__init__(P1)
#         self.base = base
#     def querys(self, t):
#         t = t.detach().requires_grad_(True)
#         logpt = self.P1.log_prob(1 / (self.base + (1 - self.base) * t)).squeeze()
#         logpt.sum().backward()
#         return

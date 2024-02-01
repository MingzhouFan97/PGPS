import torch
from . import util
class Likelihood(torch.nn.Module):
    def __init__(self, model, data_x, data_y):
        super().__init__()
        self.model = model
        self.data_x = data_x
        self.data_y = data_y

    def forward(self, prediction, observation, fac=1):
        pass

    def log_prob(self, para_vecs):
        log_p = self.forward(self.model(para_vecs, self.data_x), self.data_y, fac=1)
        return log_p

class LogBinaryBernoulliLikelihood(Likelihood):
    def __init__(self, model, data_x, data_y):
        super().__init__(model, data_x, data_y)

    def forward(self, prediction, observation, fac=1):
        p1 = torch.sigmoid(prediction)
        p_true = p1 * observation + (1 - p1) * (1 - observation)
        p_true = torch.clamp(p_true, min=1e-10, max=1-1e-10)
        return fac*torch.sum(torch.log(p_true), dim=-2).squeeze()


class LogGaussianLikelihood(Likelihood):
    def __init__(self, model, data_x, data_y, observation_noise):
        super().__init__(model, data_x, data_y)
        self.obs_n = observation_noise

    def forward(self, prediction, observation, fac=1):
        diff = prediction - observation
        return -fac*torch.sum(1/2*diff**2/self.obs_n, dim=-2).squeeze()

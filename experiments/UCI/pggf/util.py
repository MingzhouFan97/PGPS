import torch

def get_para_vector(model):
    para_list = []
    for para in model.parameters():
        para_list.append(para)
    return flatten(para_list)
def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        # param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        param = (vector[offset:offset + param.numel()].reshape(param.size()))
        offset += param.numel()

class ModelPosterior():
    def __init__(self, prior, likelihood):
        self.prior = prior
        self.likelihood = likelihood
    def log_prob(self, para_vec):
        return self.prior.log_prob(para_vec) + self.likelihood.log_prob(para_vec)

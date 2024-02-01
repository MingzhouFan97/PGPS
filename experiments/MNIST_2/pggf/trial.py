import torch


def divergence_parr(dx, y):
    def divergence_one_dim(dx_i, i):
        return torch.autograd.grad(dx_i.sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()

    batched_divergence_one_dim = torch.vmap(divergence_one_dim, in_dims=1)
    sum_diag = 0.
    i_s = torch.arange(y.shape[1]).unsqueeze(0)
    diag = batched_divergence_one_dim(dx, i_s)
    return diag


def divergence_bf(dx, y):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


if __name__ == "__main__":
    y = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True).unsqueeze(1)
    x = torch.cat([y, y**2, 2 * y**3], dim=-1)
    print(divergence_bf(x, y))
    # print(divergence_parr(x, y))
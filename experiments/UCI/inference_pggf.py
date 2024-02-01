import torch
import os
import sys
print(os.getcwd())
sys.path.append('./')
import numpy as np
import pggf.pggf_model
import pggf.pggf_inference as inference
import pggf.path as ppath
# import matplotlib.pyplot as plt
import pggf.datasets as d
import argparse
from sklearn.calibration import calibration_curve

def ECE(pred, truth):
    with torch.no_grad():
        ece = 0
        for p_l in range(pred.shape[1]):
            label = (truth == p_l).int()
            prob_true, prob_pred = calibration_curve(label.cpu(), pred[:, p_l].cpu(), n_bins=10)
            ece += np.mean(np.abs(prob_true - prob_pred))
        ece /= pred.shape[1]
    return torch.tensor(ece).unsqueeze(0).to(device)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('name', type=str)
    parser.add_argument('alpha', type=float)
    parser.add_argument('beta', type=float)
    args = parser.parse_args()
    name = args.name
    alpha = args.alpha
    beta = args.beta


    torch.manual_seed(0)
    np.random.seed(0)

    class model(torch.nn.Module):
        def __init__(self, i_d, o_d, h_d):
            super().__init__()
            self.l1 = torch.nn.Linear(i_d, h_d)
            self.nonl = torch.nn.Sigmoid()
            self.o1 = torch.nn.Linear(h_d, o_d)
            self.onl = torch.nn.Softmax(dim=-1)
        def forward(self, x):
            return self.onl(self.o1(self.nonl(self.l1(x))))

    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    print(device)


    # optimizer = torch.optim.SGD(base_model.parameters(), lr=1e-2)
    # for i in range(5000):
    #
    #     pred_y = base_model(train_x)
    #     loss = loss_f(pred_y, train_y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     print(loss)
    #     optimizer.step()
    # # with torch.no_grad():
    # #     pred_y = base_model(train_x)
    # #     plt.scatter(train_x, pred_y)
    # #     plt.show()

    rec_data = [[], [], [], [], [], []]
    for runs in range(5):
        if device == 'cpu':
            dataset = d.UCIDatasets(name, '../../dataset', device=device)
        else:
            dataset = d.UCIDatasets(name, './dataset', device=device)
        # base_model = model(dataset.in_dim, dataset.out_dim, 50)

        train_x = dataset.train_x.float()
        train_y = dataset.train_y
        loss_f = torch.nn.CrossEntropyLoss(reduction='sum')
        num_samples = 100
        path = lambda p0, p1, device: ppath.ExpTelePathNN(p0, p1, device, base=beta, alpha=alpha)
        base_model = model
        model_para = (dataset.in_dim, dataset.out_dim, 32)
        inf = inference.Inference(base_model, model_para, train_x, train_y, num_samples, device=device)
        models = inf.performe(torch.tensor(1e5), path, adj_size=1e-1, adjust_steps=300, adj_delta=0.1, device=device, final_adj=1000, max_train_iter=100)
        preds = torch.cat([model(dataset.test_x.float()).unsqueeze(0) for model in models], dim=0)
        _, pred_label = torch.max(torch.mean(preds, dim=0), dim=1)
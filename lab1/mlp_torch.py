import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_torch_model(params: dict[str, np.ndarray]) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
    with torch.no_grad():
        # numpy weights are (in, out), nn.Linear stores (out, in)
        model[0].weight.copy_(torch.from_numpy(params["W1"].T.copy()))
        model[0].bias.copy_(torch.from_numpy(params["b1"].copy()))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T.copy()))
        model[2].bias.copy_(torch.from_numpy(params["b2"].copy()))
    return model


def torch_loss_and_grads(params, X, y):
    model = build_torch_model(params)
    Xt = torch.from_numpy(X.copy())
    yt = torch.from_numpy(y.copy()).long()
    loss = F.cross_entropy(model(Xt), yt)
    model.zero_grad()
    loss.backward()
    grads = {
        "W1": model[0].weight.grad.numpy().T.copy(),
        "b1": model[0].bias.grad.numpy().copy(),
        "W2": model[2].weight.grad.numpy().T.copy(),
        "b2": model[2].bias.grad.numpy().copy(),
    }
    return float(loss.item()), grads

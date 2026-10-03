import numpy as np

from mlp_numpy import backward, forward, loss_fn
from mlp_torch import torch_loss_and_grads

TOL_TORCH = 1e-12
EPS = 1e-6
TOL_NUM = 1e-7
NUM_PARAMS = [("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))]


def manual_grads(params, X, y, divide_by_n=True):
    _, cache = forward(params, X)
    return backward(params, cache, y, divide_by_n=divide_by_n)


def compare_with_torch(params, X, y, divide_by_n=True):
    loss_np = loss_fn(params, X, y)
    loss_t, grads_t = torch_loss_and_grads(params, X, y)
    g_np = manual_grads(params, X, y, divide_by_n)

    rows = []
    d = abs(loss_np - loss_t)
    ok = bool(np.isfinite(loss_np) and np.isfinite(loss_t) and d <= TOL_TORCH)
    rows.append(("Втрата", d, ok))
    for k in ("W1", "b1", "W2", "b2"):
        finite = bool(np.isfinite(g_np[k]).all() and np.isfinite(grads_t[k]).all())
        md = float(np.max(np.abs(g_np[k] - grads_t[k])))
        rows.append((f"Градієнт {k}", md, finite and md <= TOL_TORCH))
    return loss_np, loss_t, rows


def numeric_check(params, X, y, divide_by_n=True):
    g_man = manual_grads(params, X, y, divide_by_n)
    rows = []
    for name, idx in NUM_PARAMS:
        arr = params[name]
        orig = arr[idx]
        arr[idx] = orig + EPS
        l_plus = loss_fn(params, X, y)
        arr[idx] = orig - EPS
        l_minus = loss_fn(params, X, y)
        arr[idx] = orig
        g_num = (l_plus - l_minus) / (2 * EPS)
        g_m = float(g_man[name][idx])
        diff = abs(g_num - g_m)
        label = f"{name}[{', '.join(map(str, idx))}]"
        rows.append((label, g_m, float(g_num), float(diff),
                     bool(np.isfinite(diff) and diff <= TOL_NUM)))
    return rows

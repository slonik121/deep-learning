import numpy as np
import torch
import torch.nn.functional as F

from mlp_numpy import cross_entropy, log_softmax


def naive_ce(z, y):
    with np.errstate(all="ignore"):
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(len(y)), y]).mean())


def run():
    rng = np.random.default_rng(1)

    # large logits: naive formula breaks, stable one does not
    z = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
    y = np.array([2, 0])
    naive, stable = naive_ce(z, y), cross_entropy(z, y)
    ref = F.cross_entropy(torch.from_numpy(z), torch.from_numpy(y)).item()
    assert not np.isfinite(naive)
    assert np.isfinite(stable) and abs(stable - ref) <= 1e-12

    # example from the lecture
    s = cross_entropy(np.array([[90.0, 91.0, 92.0]]), np.array([2]))
    assert abs(s - 0.40760596444438) < 1e-9

    # random logits
    for _ in range(20):
        z = rng.normal(0, 30, size=(17, 3))
        y = rng.integers(0, 3, 17)
        ref = F.cross_entropy(torch.from_numpy(z), torch.from_numpy(y)).item()
        assert abs(cross_entropy(z, y) - ref) <= 1e-12
        assert np.allclose(np.exp(log_softmax(z)).sum(1), 1.0, atol=1e-14)
    return naive, stable


if __name__ == "__main__":
    naive, stable = run()
    print(f"naive cross-entropy: {naive}, stable cross-entropy: {stable:.6f}")
    print("stability tests passed")

import numpy as np

DTYPE = np.float64


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # he for w1, xavier for w2, generated in this order
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8)).astype(DTYPE)
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3)).astype(DTYPE)
    return {
        "W1": W1,
        "b1": np.zeros(8, dtype=DTYPE),
        "W2": W2,
        "b2": np.zeros(3, dtype=DTYPE),
    }


def log_softmax(Z: np.ndarray) -> np.ndarray:
    # shift by the row maximum for stability
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    logp = log_softmax(logits)
    return float(-logp[np.arange(logits.shape[0]), y].mean())


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
    return Z2, cache


def loss_fn(params, X, y) -> float:
    logits, _ = forward(params, X)
    return cross_entropy(logits, y)


def backward(params, cache, y, divide_by_n: bool = True):
    X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
    n = X.shape[0]

    dZ2 = np.exp(log_softmax(Z2))
    dZ2[np.arange(n), y] -= 1.0
    if divide_by_n:  # set to False for the deliberate error experiment
        dZ2 /= n

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

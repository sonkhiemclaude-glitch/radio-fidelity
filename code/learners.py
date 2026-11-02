"""Downstream learners for the localization task.

The evaluation protocol is the one the guest editors' own measurement study
uses: a fingerprint database is built entirely from *simulated* reference
points, and positioning error is then measured on *real* messages.  Nothing
real enters the training side, which is what makes the number a sim-to-real
number rather than a fine-tuning number.

`FLOOR` is the value written into a fingerprint entry for a gateway that did
not hear the message.  Its choice is a nuisance parameter of the protocol, not
of the physics, so every experiment reports sensitivity to it.
"""
from __future__ import annotations
import numpy as np

FLOOR = -140.0            # dBm written for "not received"


def to_fingerprint(rssi, gw, msg, n_msg, n_gw, floor=FLOOR):
    """Scatter sparse (message, gateway, rssi) observations into a dense
    (n_msg, n_gw) fingerprint matrix."""
    F = np.full((n_msg, n_gw), floor, np.float32)
    F[np.asarray(msg, int), np.asarray(gw, int)] = rssi
    return F


def knn_localize(db_fp, db_xy, q_fp, k=5, weighted=True, block=1024):
    """k-nearest-neighbour fingerprint positioning.

    Euclidean distance in dB space over the floor-filled fingerprint vectors --
    the standard baseline, and the one used by the study this protocol follows.
    Returns predicted (x, y) for every query.
    """
    db = np.asarray(db_fp, np.float32); q = np.asarray(q_fp, np.float32)
    xy = np.asarray(db_xy, np.float64)
    out = np.empty((len(q), 2))
    db2 = (db ** 2).sum(1)
    for a in range(0, len(q), block):
        Q = q[a: a + block]
        # d2 = |q|^2 - 2 q.db + |db|^2, built in place so that the
        # (block x database) matrix exists once rather than four times
        d2 = Q @ db.T
        d2 *= -2.0
        d2 += db2[None, :]
        d2 += (Q ** 2).sum(1)[:, None]
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        dd = np.take_along_axis(d2, idx, 1)
        if weighted:
            w = 1.0 / np.maximum(dd, 1e-3)
            w /= w.sum(1, keepdims=True)
        else:
            w = np.full_like(dd, 1.0 / k)
        out[a: a + block, 0] = (w * xy[idx, 0]).sum(1)
        out[a: a + block, 1] = (w * xy[idx, 1]).sum(1)
    return out


def positioning_errors(pred_xy, true_xy):
    return np.hypot(pred_xy[:, 0] - true_xy[:, 0], pred_xy[:, 1] - true_xy[:, 1])


# ---------------------------------------------------------------------------
# a small representation learner, for the "foundation-model" arm
# ---------------------------------------------------------------------------
def pretrain_encoder(X, Y, dim=32, hidden=128, epochs=40, batch=512, lr=1e-3,
                     seed=0, verbose=False):
    """Train a small MLP encoder + head on simulated fingerprints, and return
    the frozen encoder.  Deliberately tiny: the claim under test is about which
    simulator produced the pretraining data, not about model scale."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    Xt = torch.tensor(np.asarray(X, np.float32))
    Yt = torch.tensor(np.asarray(Y, np.float32))
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
    ymu, ysd = Yt.mean(0, keepdim=True), Yt.std(0, keepdim=True) + 1e-6
    enc = nn.Sequential(nn.Linear(Xt.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, dim))
    head = nn.Linear(dim, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for a in range(0, n, batch):
            b = perm[a: a + batch]
            xb = (Xt[b] - mu) / sd
            yb = (Yt[b] - ymu) / ysd
            loss = ((head(enc(xb)) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if verbose and ep % 10 == 0:
            print(f"    epoch {ep} loss {tot/n:.4f}")
    return {"enc": enc, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd}


def embed(model, X):
    import torch
    with torch.no_grad():
        Xt = torch.tensor(np.asarray(X, np.float32))
        return model["enc"]((Xt - model["mu"]) / model["sd"]).numpy()


def linear_probe(Z_tr, Y_tr, Z_te, ridge=1.0):
    """Ridge probe on frozen features: the standard foundation-model readout."""
    Z = np.c_[Z_tr, np.ones(len(Z_tr))]
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ Y_tr)
    return np.c_[Z_te, np.ones(len(Z_te))] @ W


def knn_indices(db_fp, q_fp, k=5, block=1024):
    """Neighbour indices and inverse-distance weights, so that several targets
    (position, floor, building) can be read off one search."""
    db = np.asarray(db_fp, np.float32); q = np.asarray(q_fp, np.float32)
    I = np.empty((len(q), k), np.int64); W = np.empty((len(q), k), np.float64)
    db2 = (db ** 2).sum(1)
    for a in range(0, len(q), block):
        Q = q[a: a + block]
        d2 = Q @ db.T
        d2 *= -2.0
        d2 += db2[None, :]
        d2 += (Q ** 2).sum(1)[:, None]
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        dd = np.take_along_axis(d2, idx, 1)
        w = 1.0 / np.maximum(dd, 1e-3)
        W[a: a + block] = w / w.sum(1, keepdims=True)
        I[a: a + block] = idx
    return I, W

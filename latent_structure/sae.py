"""A small sparse autoencoder (SAE) for unsupervised feature discovery.

Standard tied-bias SAE: x -> (x - b_dec) -> ReLU(W_enc x + b_enc) = f (sparse) ->
f W_dec + b_dec = x_hat. Loss = reconstruction MSE + L1 sparsity. Decoder columns
are renormalized to unit norm each step (the usual SAE convention, so the L1 acts
on feature magnitudes, not decoder scale).

Used in two modes (scripts/16):
  - CALIBRATION (D-1): train on a PLANTED model's activations and check whether some
    sparse feature recovers the known latent (depth) -- validating the method at our
    scale before trusting it on unlabeled structure.
  - DISCOVERY (D-2): train on a capable model over natural text, then flag features
    that do NOT correlate with KNOWN properties (POS / depth / position / frequency)
    as candidate non-obvious structure.
"""

from __future__ import annotations

import numpy as np


def train_sae(X, expansion=8, l1=1e-3, steps=4000, batch=4096, lr=1e-3,
              device="cuda", seed=0, log=None):
    """Train an SAE on activations X (n, d). Returns (model, stats)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    n, d = X.shape
    m = expansion * d
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    mu = Xt.mean(0)
    Xc = Xt - mu                                   # center (b_dec absorbs the mean)
    # normalize input scale so mean ||x|| = sqrt(d) (standard SAE convention) --
    # makes the L1 sparsity coefficient scale-consistent across models/layers.
    scale = float(np.sqrt(d)) / (Xc.norm(dim=1).mean().item() + 1e-8)
    Xc = Xc * scale

    W_enc = nn.Parameter((torch.randn(d, m, device=device) / np.sqrt(d)))
    b_enc = nn.Parameter(torch.zeros(m, device=device))
    W_dec = nn.Parameter((torch.randn(m, d, device=device) / np.sqrt(m)))
    opt = torch.optim.Adam([W_enc, b_enc, W_dec], lr=lr)
    rng = np.random.default_rng(seed)

    def normalize_dec():
        with torch.no_grad():
            W_dec.data /= (W_dec.data.norm(dim=1, keepdim=True) + 1e-8)

    normalize_dec()
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        x = Xc[idx]
        f = torch.relu(x @ W_enc + b_enc)
        x_hat = f @ W_dec
        recon = ((x_hat - x) ** 2).sum(1).mean()
        sparsity = f.abs().sum(1).mean()
        loss = recon + l1 * sparsity
        opt.zero_grad(); loss.backward(); opt.step()
        normalize_dec()
        if log and (step % 1000 == 0 or step == steps):
            with torch.no_grad():
                fvu = (((x_hat - x) ** 2).sum() / (x ** 2).sum()).item()
                l0 = (f > 1e-6).float().sum(1).mean().item()
            log(f"[sae] step {step} loss {loss.item():.3f} FVU {fvu:.3f} L0 {l0:.1f}")

    # final stats on a held-out-ish sample
    with torch.no_grad():
        x = Xc
        f_all = torch.relu(x @ W_enc + b_enc)
        x_hat = f_all @ W_dec
        fvu = float(((x_hat - x) ** 2).sum() / (x ** 2).sum())
        l0 = float((f_all > 1e-6).float().sum(1).mean())
    model = {"W_enc": W_enc.detach(), "b_enc": b_enc.detach(),
             "W_dec": W_dec.detach(), "mu": mu, "scale": scale, "m": m}
    return model, {"fvu": fvu, "l0": l0, "n_features": m}


def encode(model, X, device="cuda", batch=8192):
    """Feature activations f (n, m) for X (n, d)."""
    import torch
    out = []
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            x = (Xt[s:s + batch] - model["mu"]) * model["scale"]
            out.append(torch.relu(x @ model["W_enc"] + model["b_enc"]).cpu().numpy())
    return np.concatenate(out)


def feature_label_corr(F, labels):
    """Per-feature |Pearson corr| with a scalar label, ignoring dead features.
    Returns (best_feature_idx, best_abs_corr, corr_array)."""
    labels = np.asarray(labels, dtype=np.float64)
    lab = (labels - labels.mean()) / (labels.std() + 1e-9)
    corrs = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        f = F[:, j]
        if f.std() < 1e-9:
            continue
        fz = (f - f.mean()) / (f.std() + 1e-9)
        corrs[j] = float(np.mean(fz * lab))
    a = np.abs(corrs)
    return int(a.argmax()), float(a.max()), corrs

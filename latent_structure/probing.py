"""Probing the planted latent structure: does the trained model REPRESENT the
hidden depth, and is that representation genuine (not a probe artifact)?

Per the lit review, probe accuracy alone is insufficient (it is similar for real
and random labels). So every probe is gated on:
  - MDL / online (prequential) codelength (Voita & Titov 2020): genuine structure
    compresses; a control task does not.
  - a CONTROL TASK (shuffled labels of matched cardinality): codelength/accuracy
    on real labels must beat it.
  - a RANDOM-INIT model baseline: a trained model must encode depth far better
    than its untrained counterpart.

The probe target at position i is `depth_after[i]` (see synthetic_corpus.py): the
latent depth the model's position-i hidden state should encode to predict the
marker at i+1. Linear (logistic-regression) probes by default, per the lit review.
"""

from __future__ import annotations

import numpy as np


def collect_position_activations(model, corpus, rows, layer, device, *,
                                 max_positions=25000, batch=64, seed=0):
    """Run the model over `rows` and collect (residual-stream activation, depth)
    pairs at `layer`. Subsamples to `max_positions`. Skips position 0 (no
    predictive context) and the sequence boundaries are fine since depth_after is
    defined everywhere."""
    import torch
    model.eval()
    rng = np.random.default_rng(seed)
    feats, depths = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            idx = rows[i:i + batch]
            x = torch.from_numpy(corpus.tokens[idx].astype(np.int64)).to(device)
            out = model(input_ids=x, output_hidden_states=True)
            h = out.hidden_states[layer].float().cpu().numpy()   # (b, L, d)
            d = corpus.depth_after[idx]                          # (b, L)
            b, L, dim = h.shape
            feats.append(h.reshape(b * L, dim))
            depths.append(d.reshape(b * L))
    X = np.concatenate(feats); y = np.concatenate(depths).astype(np.int64)
    if len(X) > max_positions:
        sel = rng.choice(len(X), max_positions, replace=False)
        X, y = X[sel], y[sel]
    return X, y


def probe_accuracy(X, y, seed=0):
    """Linear-probe test accuracy for real labels and for a control task
    (shuffled labels). A gap is necessary (not sufficient) evidence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    clf = LogisticRegression(max_iter=200, multi_class="multinomial")
    clf.fit(Xtr, ytr)
    real = clf.score(Xte, yte)
    rng = np.random.default_rng(seed)
    yshuf = rng.permutation(ytr)
    ctrl = LogisticRegression(max_iter=200, multi_class="multinomial")
    ctrl.fit(Xtr, yshuf)
    control = ctrl.score(Xte, rng.permutation(yte))
    return {"real": real, "control_task": control, "n_classes": int(y.max() + 1)}


def online_codelength(X, y, seed=0,
                      fractions=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Prequential (online) codelength in bits: code each data block under a
    probe trained on all earlier blocks; block 0 under a uniform prior. Returns
    total bits, the uniform-code baseline, and the compression ratio
    (uniform/total). Lower total bits == more genuine structure (Voita & Titov)."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    K = int(y.max() + 1)
    uniform_bits = n * np.log2(K)
    cuts = [int(n * f) for f in fractions]
    bits = 0.0
    # first block: uniform prior
    bits += cuts[0] * np.log2(K)
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        clf = LogisticRegression(max_iter=200, multi_class="multinomial")
        # guard: need >=2 classes present in the training prefix
        if len(np.unique(y[:a])) < 2:
            bits += (b - a) * np.log2(K)
            continue
        clf.fit(X[:a], y[:a])
        proba = clf.predict_proba(X[a:b])
        classes = clf.classes_
        col = {c: j for j, c in enumerate(classes)}
        p = np.array([proba[r, col[y[a + r]]] if y[a + r] in col else 1e-12
                      for r in range(b - a)])
        p = np.clip(p, 1e-12, 1.0)
        bits += float(-np.log2(p).sum())
    return {"total_bits": bits, "uniform_bits": float(uniform_bits),
            "compression": float(uniform_bits / bits) if bits > 0 else float("nan"),
            "bits_per_token": bits / n}


def probe_report(model, random_init_model, corpus, rows, layer, device, seed=0):
    """Full probe report at `layer`: accuracy (real vs control task), MDL for the
    trained model, and MDL for a random-init model -- the two controls that
    distinguish genuine structure from a probe artifact."""
    X, y = collect_position_activations(model, corpus, rows, layer, device, seed=seed)
    acc = probe_accuracy(X, y, seed=seed)
    mdl = online_codelength(X, y, seed=seed)
    Xr, yr = collect_position_activations(random_init_model, corpus, rows, layer,
                                          device, seed=seed)
    mdl_rand = online_codelength(Xr, yr, seed=seed)
    return {"layer": layer, "accuracy": acc, "mdl_trained": mdl,
            "mdl_random_init": mdl_rand,
            "mdl_ratio_rand_over_trained": mdl_rand["total_bits"] / mdl["total_bits"]}

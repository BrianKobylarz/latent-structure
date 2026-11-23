"""Shared structural-probe machinery (Hewitt & Manning norm probe) + subspace
helpers for the causal-use test.

The norm probe predicts a per-token non-negative scalar (tree depth, or linear
head-distance) as ||B h||^2 under a learned linear map B, fit by L1 regression,
with per-dimension activation STANDARDIZATION (essential at LLM scale). Scored by
norm-Spearman (rank-corr of ||Bh||^2 vs the true scalar) and root accuracy (for
depth: argmin ||Bh||^2 == the depth-0 token).

`orthonormal_rowspace` / `random_rowspace` + `project_out` support the causal test
in scripts/14: ablate the depth subspace from the residual stream and measure the
LM-loss impact vs a random subspace of equal rank (decodability != use).
"""

from __future__ import annotations

import numpy as np


def collect_sentences(model, tok, sents, layer, device, max_sents=1500):
    """Per-sentence (word_activations (n_words, d), labels (n_words,)) using the
    LAST subtoken of each word. `sents` = [(text, [(cs, ce, label), ...]), ...]."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for text, spans in sents:
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) < 2 or len(ids) > 1024:
                continue
            acts, labs = [], []
            for cs, ce, lab in spans:
                toks = [j for j, (a, b) in enumerate(offs)
                        if a != b and not (b <= cs or a >= ce)]
                if toks:
                    acts.append(toks[-1]); labs.append(lab)
            if len(acts) < 3:
                continue
            x = torch.tensor([ids], device=device)
            h = model(input_ids=x, output_hidden_states=True).hidden_states[layer][0]
            out.append((h[acts].float().cpu().numpy(), np.asarray(labs, dtype=np.float32)))
            if len(out) >= max_sents:
                break
    return out


def train_norm_probe(train, d_model, device, rank=128, steps=3000, seed=0):
    import torch
    torch.manual_seed(seed)
    Hraw = np.concatenate([a for a, _ in train])
    mu = Hraw.mean(0); sigma = Hraw.std(0) + 1e-6
    H = torch.tensor((Hraw - mu) / sigma, device=device)
    D = torch.tensor(np.concatenate([d for _, d in train]), device=device)
    B = torch.nn.Parameter(torch.randn(rank, d_model, device=device) / np.sqrt(rank * d_model))
    opt = torch.optim.Adam([B], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        pred = (H @ B.t()).pow(2).sum(1)
        loss = (pred - D).abs().mean()
        loss.backward(); opt.step(); sched.step()
    return (B.detach(), torch.tensor(mu, device=device), torch.tensor(sigma, device=device))


def eval_norm_probe(probe, test, device, root=True):
    import torch
    from scipy.stats import spearmanr
    B, mu, sigma = probe
    nsprs, roots = [], []
    with torch.no_grad():
        for a, d in test:
            if len(d) < 3 or np.ptp(d) == 0:
                continue
            h = (torch.tensor(a, device=device) - mu) / sigma
            pred = (h @ B.t()).pow(2).sum(1).cpu().numpy()
            rho = spearmanr(pred, d).statistic
            if rho == rho:
                nsprs.append(float(rho))
            if root:
                roots.append(int(np.argmin(pred) == np.argmin(d)))
    out = {"norm_spearman": float(np.mean(nsprs)), "n_sentences": len(nsprs)}
    if root:
        out["root_acc"] = float(np.mean(roots)) if roots else float("nan")
    return out


# --- subspace helpers for the causal-use test (scripts/14) ---

def orthonormal_rowspace(B):
    """Orthonormal basis (r, d) spanning the row space of B (the depth subspace
    in standardized activation space)."""
    import torch
    Q, _ = torch.linalg.qr(B.t())      # (d, r)
    return Q.t()                        # (r, d) orthonormal rows


def random_rowspace(rank, d, device, seed=0):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(d, rank, generator=g).to(device)
    Q, _ = torch.linalg.qr(M)
    return Q.t()[:rank]


def project_out(h_std, Q):
    """Remove the subspace spanned by orthonormal rows Q from standardized acts
    h_std (..., d): h - (h Q^T) Q."""
    return h_std - (h_std @ Q.t()) @ Q


def decoder_layers(model):
    """Return the list of decoder blocks (arch-agnostic: GPT-2 / Llama / Qwen /
    Phi / GPT-NeoX). hidden_states[L] is the output of layers[L-1]."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError("could not locate decoder layers")

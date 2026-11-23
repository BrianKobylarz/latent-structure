"""Interchange restricted to the subspace the counter is read from.

`scripts/07_intervene_planted.py` swaps the whole residual vector at the
prediction position. That shows where the deciding information is, but not which
part of the vector carries it. Here only a low-dimensional part is swapped:

    h_new = h_recipient + P (h_donor - h_recipient)

with P the orthogonal projector onto

  * the subspace spanned by the mean activation of each counter value (rank 8:
    nine counter values);
  * the single direction joining the mean activations of the two sides of the
    threshold (rank 1);
  * the discriminant directions of a linear probe for the counter (linear
    discriminant analysis: rank 8). These are the directions a probe reads the
    counter from; they need not be the directions along which the counter
    moves the activation;

and, as comparisons, onto

  * random subspaces of the same rank;
  * the leading principal directions of the activations once the class-mean
    and probe subspaces have been projected out (a high-variance subspace
    unrelated to the counter);

and onto the complement of the first three (everything swapped except that
subspace).

Every subspace is fit on training sequences; the swaps are run on held-out
sequences. A swap "flips" when the marker the model prefers moves to the donor's
side of the threshold. No GPU is needed: the planted models have about three
million parameters.

Usage (after scripts/05_run_planted.py has trained the models):
    python scripts/19_subspace_interchange.py --config config/pilot.yaml \
        --conditions rule_p_1.0 rule_p_0.05 needle_rule_p_1.0 natural_rule_p_1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml
from transformers.utils.logging import disable_progress_bar, set_verbosity_error

disable_progress_bar()
set_verbosity_error()

from latent_structure import corpus as C
from latent_structure.intervention import collect_pred_positions
from latent_structure.synthetic_corpus import Corpus

RANK = 8          # nine counter values -> eight independent probe directions
N_RANDOM = 5
BATCH = 64


def orthonormal_rows(m, k=None):
    """Orthonormal basis (rows) of the row space of m, leading directions first."""
    _, s, vt = np.linalg.svd(np.asarray(m, dtype=np.float64), full_matrices=False)
    keep = s > 1e-8 * s[0]
    q = vt[keep]
    return (q[:k] if k else q).astype(np.float32)


def random_rows(rank, d, seed):
    g = np.random.default_rng(seed).standard_normal((d, rank))
    q, _ = np.linalg.qr(g)
    return q.T.astype(np.float32)


def collect_activations(model, corpus, rows, layer, max_positions, seed):
    import torch
    feats, depth = [], []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            idx = rows[i:i + BATCH]
            x = torch.from_numpy(corpus.tokens[idx].astype(np.int64)).to(model.device)
            h = model(input_ids=x, output_hidden_states=True).hidden_states[layer]
            feats.append(h.reshape(-1, h.shape[-1]).float().cpu().numpy())
            depth.append(corpus.depth_after[idx].reshape(-1))
    X = np.concatenate(feats)
    y = np.concatenate(depth).astype(np.int64)
    sel = np.random.default_rng(seed).choice(len(X), min(max_positions, len(X)), replace=False)
    return X[sel], y[sel]


def fit_subspaces(model, corpus, train_rows, test_rows, layer, seed):
    """Probe subspace, side direction and the leftover principal directions at
    one layer, all from training sequences; probe accuracy on held-out ones."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    thresh = corpus.config.effective_thresh
    X, y = collect_activations(model, corpus, train_rows[:320], layer, 30000, seed)
    Xt, yt = collect_activations(model, corpus, test_rows[:120], layer, 15000, seed + 1)
    mu = X.mean(0)
    # A linear discriminant probe: closed form, and it reads the counter only
    # through its discriminant directions, so swapping their span hands the
    # probe exactly the donor's reading.
    probe = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
    acc = float(probe.score(Xt, yt))
    q_probe = orthonormal_rows(probe.scalings_[:, :RANK].T, RANK)
    # the subspace spanned by the mean activation of each counter value
    means = np.stack([X[y == c].mean(0) for c in np.unique(y)])
    q_means = orthonormal_rows(means - means.mean(0, keepdims=True), RANK)
    side = y >= thresh
    u = X[side].mean(0) - X[~side].mean(0)
    q_side = (u / np.linalg.norm(u))[None, :].astype(np.float32)
    xc = X - mu
    # leading principal directions of what is left once both subspaces are removed
    q_both = orthonormal_rows(np.concatenate([q_probe, q_means]))
    xr = xc - (xc @ q_both.T) @ q_both
    evals, evecs = np.linalg.eigh((xr.T @ xr).astype(np.float64))
    q_pc = evecs[:, ::-1][:, :RANK].T.astype(np.float32)
    total = float((xc ** 2).sum())
    share = lambda q: float(((xc @ q.T) ** 2).sum() / total)
    # how much of the class-mean subspace lies inside the probe subspace (1 = all of it)
    overlap = float(((q_means @ q_probe.T) ** 2).sum() / q_means.shape[0])
    info = {"probe_accuracy_heldout": acc, "probe_rank": int(q_probe.shape[0]),
            "class_mean_rank": int(q_means.shape[0]),
            "class_mean_subspace_inside_probe_subspace": overlap,
            "variance_share_probe_subspace": share(q_probe),
            "variance_share_class_mean_subspace": share(q_means),
            "variance_share_side_direction": share(q_side),
            "variance_share_leftover_principal": share(q_pc)}
    return q_probe, q_means, q_side, q_pc, info


def flip_rates(model, corpus, layer, pairs, transforms):
    """For each named transform, the share of pairs whose preferred marker ends
    on the donor's side. `pairs` = list of (recipient, donor), each (row, pos, side)."""
    import torch
    cfg = corpus.config
    block = model.transformer.h[layer - 1]
    state = {}

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out   # tuple or tensor, by library version
        pos = state["pos"]
        idx = torch.arange(len(pos), device=hs.device)
        r = hs[idx, pos, :]
        d = state["donor"]
        kind, q = state["transform"]
        if kind == "full":
            new = d
        elif kind == "none":
            new = r
        elif kind == "sub":                          # swap only the subspace
            new = r + ((d - r) @ q.T) @ q
        else:                                        # "comp": swap everything else
            new = d + ((r - d) @ q.T) @ q
        hs[idx, pos, :] = new.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    hits = {name: 0 for name in transforms}
    n = 0
    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, len(pairs), BATCH):
                chunk = pairs[s:s + BATCH]
                dev = model.device
                rtok = torch.from_numpy(np.stack([corpus.tokens[r[0]] for r, _ in chunk]).astype(np.int64)).to(dev)
                dtok = torch.from_numpy(np.stack([corpus.tokens[d[0]] for _, d in chunk]).astype(np.int64)).to(dev)
                rpos = torch.tensor([r[1] for r, _ in chunk], device=dev)
                dpos = torch.tensor([d[1] for _, d in chunk], device=dev)
                donor_side = torch.tensor([d[2] for _, d in chunk], device=dev)
                rows = torch.arange(len(chunk), device=dev)
                state.update(pos=rpos, transform=("none", None), donor=None)
                dact = model(input_ids=dtok, output_hidden_states=True).hidden_states[layer]
                state["donor"] = dact[rows, dpos, :]
                for name, tr in transforms.items():
                    state["transform"] = tr
                    logits = model(input_ids=rtok).logits[rows, rpos, :]
                    pred_high = (logits[:, cfg.b_id] > logits[:, cfg.a_id]).long()
                    hits[name] += int((pred_high == donor_side).sum())
                n += len(chunk)
    finally:
        handle.remove()
    return {name: h / n for name, h in hits.items()}


def run_condition(run_dir, n_pairs, seed):
    import torch
    from transformers import GPT2LMHeadModel
    corpus = Corpus.load(run_dir / "corpus.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(run_dir).eval().to(device)
    n = corpus.tokens.shape[0]
    cut = int(n * 0.9)
    train_rows, test_rows = np.arange(cut), np.arange(cut, n)

    pos = collect_pred_positions(corpus, test_rows)
    rng = np.random.default_rng(seed)
    low = [p for p in pos if p[2] == 0]
    high = [p for p in pos if p[2] == 1]
    rng.shuffle(low)
    rng.shuffle(high)
    k = min(n_pairs, len(low) // 2, len(high) // 2)
    opposite = list(zip(low[:k], high[:k])) + list(zip(high[:k], low[:k]))
    same = list(zip(low[:k], low[k:2 * k])) + list(zip(high[:k], high[k:2 * k]))

    d = model.config.n_embd
    out = {"pairs_per_direction": k, "layers": {}}
    for layer in range(1, model.config.n_layer + 1):
        q_probe, q_means, q_side, q_pc, info = fit_subspaces(model, corpus, train_rows, test_rows, layer, seed)
        t = lambda a: torch.from_numpy(a).to(device)
        transforms = {
            "no_swap": ("none", None),
            "whole_vector": ("full", None),
            "class_mean_subspace": ("sub", t(q_means)),
            "all_but_class_mean_subspace": ("comp", t(q_means)),
            "side_direction": ("sub", t(q_side)),
            "all_but_side_direction": ("comp", t(q_side)),
            "probe_subspace": ("sub", t(q_probe)),
            "all_but_probe_subspace": ("comp", t(q_probe)),
            "leftover_principal_subspace": ("sub", t(q_pc)),
        }
        for s in range(N_RANDOM):
            transforms[f"random_rank{RANK}_{s}"] = ("sub", t(random_rows(RANK, d, 1000 + s)))
            transforms[f"random_rank1_{s}"] = ("sub", t(random_rows(1, d, 2000 + s)))
        opp = flip_rates(model, corpus, layer, opposite, transforms)
        ctl = flip_rates(model, corpus, layer, same,
                         {k: transforms[k] for k in ("class_mean_subspace", "side_direction", "whole_vector")})
        r8 = [v for name, v in opp.items() if name.startswith(f"random_rank{RANK}_")]
        r1 = [v for name, v in opp.items() if name.startswith("random_rank1_")]
        rec = dict(info)
        rec.update({
            "flip_no_swap": opp["no_swap"],
            "flip_whole_vector": opp["whole_vector"],
            "flip_class_mean_subspace": opp["class_mean_subspace"],
            "flip_all_but_class_mean_subspace": opp["all_but_class_mean_subspace"],
            "flip_side_direction": opp["side_direction"],
            "flip_all_but_side_direction": opp["all_but_side_direction"],
            "flip_probe_subspace": opp["probe_subspace"],
            "flip_all_but_probe_subspace": opp["all_but_probe_subspace"],
            "flip_leftover_principal_subspace": opp["leftover_principal_subspace"],
            "flip_random_same_rank_mean": float(np.mean(r8)), "flip_random_same_rank_max": float(np.max(r8)),
            "flip_random_rank1_mean": float(np.mean(r1)), "flip_random_rank1_max": float(np.max(r1)),
            # donors from the recipient's own side: share that leave their side (should be near 0)
            "same_side_class_mean_subspace_left_side": 1.0 - ctl["class_mean_subspace"],
            "same_side_side_direction_left_side": 1.0 - ctl["side_direction"],
            "same_side_whole_vector_left_side": 1.0 - ctl["whole_vector"],
        })
        out["layers"][str(layer)] = rec
        print(f"  layer {layer}: probe acc {info['probe_accuracy_heldout']:.3f} | flips: whole {opp['whole_vector']:.3f}  "
              f"class means {opp['class_mean_subspace']:.3f}  all but them {opp['all_but_class_mean_subspace']:.3f}  "
              f"side direction {opp['side_direction']:.3f}  all but it {opp['all_but_side_direction']:.3f}  "
              f"probe subspace {opp['probe_subspace']:.3f}  all but it {opp['all_but_probe_subspace']:.3f}  "
              f"leftover principal {opp['leftover_principal_subspace']:.3f}  "
              f"random rank {RANK} {np.mean(r8):.3f} (max {np.max(r8):.3f})  no swap {opp['no_swap']:.3f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--conditions", nargs="+", default=["rule_p_1.0"])
    ap.add_argument("--n-pairs", type=int, default=800)
    ap.add_argument("--out", default=None,
                    help="output file (default: <storage_root>/runs/planted/subspace_interchange.json)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    base = runs_dir / "planted"
    out_path = Path(args.out) if args.out else base / "subspace_interchange.json"

    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for cond in args.conditions:
        run_dir = base / cond
        if not (run_dir / "corpus.npz").exists():
            sys.exit(f"{run_dir} has no corpus.npz; run 05_run_planted.py first")
        print(f"=== subspace interchange: {cond} ===", flush=True)
        results[cond] = run_condition(run_dir, args.n_pairs, seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[subspace] wrote {out_path}")


if __name__ == "__main__":
    main()

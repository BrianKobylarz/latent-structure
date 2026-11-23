"""Local-context baselines for the planted counter, scored against the hidden rule.

The n-gram baseline stored with each training run (`ngram_null` in
`history.json`) counts whole contexts of raw tokens and is scored against the
marker that was emitted. The model's marker accuracy is scored against the hidden
rule. Below rule_p = 1.0 those are different targets, and with eight raw tokens
the context table is sparse. This script computes baselines that are scored the
same way as the model:

  * the same count-based n-gram over raw tokens;
  * an n-gram over a reduced alphabet in which every filler token is one symbol
    (filler, open, close, A, B), which is far less sparse;
  * copying the most recent earlier marker of the sequence;
  * always naming the more common side of the threshold.

Every predictor is fit on the first half of the sequences from the emitted
markers only (what a learner sees) and scored on the second half, both against
the emitted marker and against the rule. No model and no GPU are needed. The
corpora are regenerated from the config and the seed exactly as
`scripts/05_run_planted.py` generates them.

Usage:
    python scripts/18_local_baselines.py --config config/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml

from latent_structure.synthetic_corpus import GenConfig, generate

# The recorded sweeps (results/planted/), by run-name prefix.
SWEEPS = {
    "": [1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0],
    "needle_": [1.0, 0.5, 0.3, 0.1, 0.0],
    "natural_": [1.0, 0.3, 0.0],
}
KS = (1, 2, 4, 8)


def reduced_alphabet(corpus):
    cfg = corpus.config
    t = corpus.tokens.astype(np.int64)
    r = np.zeros_like(t)                      # 0 = any filler token
    r[t == cfg.open_id] = 1
    r[t == cfg.close_id] = 2
    r[t == cfg.a_id] = 3
    r[t == cfg.b_id] = 4
    return r


def ngram_scores(seqs, corpus, k):
    """Count-based n-gram over the symbol sequences `seqs`. Returns the share of
    held-out markers predicted correctly against the emitted marker and against
    the rule, and the share of held-out contexts never seen in the fit half."""
    is_marker = corpus.is_marker
    emitted_b = corpus.tokens == corpus.config.b_id
    rule_b = corpus.governing_depth() >= corpus.config.effective_thresh
    n = seqs.shape[0]
    cut = n // 2

    def key(r, t):
        return seqs[r, t - k:t].tobytes()

    counts = defaultdict(lambda: [0, 0])
    for r in range(cut):
        for t in np.flatnonzero(is_marker[r]):
            if t >= k:
                counts[key(r, t)][int(emitted_b[r, t])] += 1
    pred = {c: (0 if v[0] >= v[1] else 1) for c, v in counts.items()}
    tot_a = sum(v[0] for v in counts.values())
    tot_b = sum(v[1] for v in counts.values())
    fallback = 0 if tot_a >= tot_b else 1

    hit_emitted = hit_rule = total = unseen = 0
    for r in range(cut, n):
        for t in np.flatnonzero(is_marker[r]):
            if t < k:
                continue
            c = key(r, t)
            if c not in pred:
                unseen += 1
            g = pred.get(c, fallback)
            hit_emitted += g == int(emitted_b[r, t])
            hit_rule += g == int(rule_b[r, t])
            total += 1
    return {"vs_emitted": hit_emitted / total, "vs_rule": hit_rule / total,
            "unseen_contexts": unseen / total, "n_eval": total}


def copy_previous_marker(corpus):
    is_marker = corpus.is_marker
    emitted_b = corpus.tokens == corpus.config.b_id
    rule_b = corpus.governing_depth() >= corpus.config.effective_thresh
    n = corpus.tokens.shape[0]
    hit_emitted = hit_rule = total = 0
    for r in range(n // 2, n):
        last = None
        for t in np.flatnonzero(is_marker[r]):
            if last is not None:
                hit_emitted += last == int(emitted_b[r, t])
                hit_rule += last == int(rule_b[r, t])
                total += 1
            last = int(emitted_b[r, t])
    return {"vs_emitted": hit_emitted / total, "vs_rule": hit_rule / total, "n_eval": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--out", default="results/planted/local_baselines.json")
    ap.add_argument("--sweeps", nargs="+", default=["synthetic", "needle"],
                    choices=["synthetic", "needle", "natural"],
                    help="'natural' needs WikiText-103 through the datasets library")
    ap.add_argument("--rule-p", nargs="+", type=float, default=None,
                    help="restrict to these values of rule_p")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed; entries then carry a _seed<N> suffix")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pc = cfg["planted"]
    seed = int(cfg.get("seed", 0)) if args.seed is None else args.seed
    suffix = "" if args.seed is None else f"_seed{seed}"
    prefixes = {"synthetic": "", "needle": "needle_", "natural": "natural_"}

    out = Path(args.out)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    natural_stream = None
    for sweep_name in args.sweeps:
        prefix = prefixes[sweep_name]
        natural = sweep_name == "natural"
        vocab = pc.get("natural_vocab_k", 512) if natural else pc["carrier_vocab"]
        if natural and natural_stream is None:
            from latent_structure.synthetic_corpus import build_natural_stream
            natural_stream = build_natural_stream(vocab, split=cfg["corpus"]["train_split"])
        for rule_p in SWEEPS[prefix]:
            if args.rule_p is not None and rule_p not in args.rule_p:
                continue
            gen = GenConfig(carrier_vocab=vocab, d_max=pc["d_max"],
                            rule_p=rule_p, seq_len=pc["seq_len"], n_seqs=pc["n_seqs"],
                            seed=seed, needle_mode=sweep_name == "needle",
                            carrier_mode="natural" if natural else "synthetic",
                            natural_vocab_k=vocab)
            corpus = generate(gen, natural_stream=natural_stream if natural else None)
            raw = corpus.tokens.astype(np.int64)
            red = reduced_alphabet(corpus)
            m = corpus.is_marker
            half = corpus.tokens.shape[0] // 2
            rule_b = (corpus.governing_depth() >= corpus.config.effective_thresh)[half:][m[half:]]
            rec = {
                "rule_p": rule_p, "needle_mode": sweep_name == "needle",
                "carrier": "natural" if natural else "synthetic",
                "emitted_equals_rule": float(corpus.followed_rule[m].mean()),
                # a predictor that always names the more common side of the threshold
                "constant_vs_rule": float(max(rule_b.mean(), 1 - rule_b.mean())),
                "raw_ngram": {str(k): ngram_scores(raw, corpus, k) for k in KS},
                "reduced_ngram": {str(k): ngram_scores(red, corpus, k) for k in KS},
                "copy_previous_marker": copy_previous_marker(corpus),
            }
            rec["recorded_style_ngram_null"] = max(v["vs_emitted"] for v in rec["raw_ngram"].values())
            rec["best_local_vs_rule"] = max(
                [v["vs_rule"] for v in rec["raw_ngram"].values()]
                + [v["vs_rule"] for v in rec["reduced_ngram"].values()]
                + [rec["copy_previous_marker"]["vs_rule"], rec["constant_vs_rule"]])
            name = f"{prefix}rule_p_{rule_p}{suffix}"
            results[name] = rec
            print(f"[local] {name:22s} recorded-style null {rec['recorded_style_ngram_null']:.3f} | "
                  f"best local predictor against the rule {rec['best_local_vs_rule']:.3f} "
                  f"(copy previous marker {rec['copy_previous_marker']['vs_rule']:.3f})", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[local] wrote {args.out}")


if __name__ == "__main__":
    main()

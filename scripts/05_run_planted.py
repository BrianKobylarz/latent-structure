"""Phase A1: planted-structure reliability (rule_p) sweep.

For each rule_p, generate a synthetic corpus (hidden bounded counter governs a
non-salient A/B marker), train a small from-scratch Transformer, and report
behavioral marker accuracy vs the n-gram null and the true-depth oracle. Also
runs a shuffle control (train on token-shuffled sequences, evaluate on clean
markers) which should NOT beat the null. Traces the sensitivity frontier.

Usage:
    python scripts/05_run_planted.py --config config/pilot.yaml
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

from latent_structure import corpus as C
from latent_structure.synthetic_corpus import (
    GenConfig, generate, build_natural_stream, ngram_marker_accuracy,
    oracle_marker_accuracy)
from latent_structure.planted import train_planted


def shuffle_train_rows(corpus, seed):
    """Return a copy of `corpus` with each TRAIN-row token sequence permuted
    (order destroyed, unigram preserved); test rows and their depth labels are
    left clean so behavioral eval remains valid."""
    import copy
    rng = np.random.default_rng(seed)
    cut = int(corpus.tokens.shape[0] * 0.9)
    toks = corpus.tokens.copy()
    for r in range(cut):
        rng.shuffle(toks[r])
    c2 = copy.copy(corpus)
    c2.tokens = toks
    return c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed (corpus and model); run folders then "
                         "carry a _seed<N> suffix")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pc = cfg["planted"]
    seed = int(cfg.get("seed", 0)) if args.seed is None else args.seed
    suffix = "" if args.seed is None else f"_seed{seed}"
    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    base = runs_dir / "planted"
    base.mkdir(parents=True, exist_ok=True)

    needle = bool(pc.get("needle_mode", False))
    carrier_mode = pc.get("carrier_mode", "synthetic")
    natural = carrier_mode == "natural"
    prefix = ("natural_" if natural else "") + ("needle_" if needle else "")

    natural_stream = None
    carrier_vocab = pc["carrier_vocab"]
    if natural:
        carrier_vocab = pc.get("natural_vocab_k", 512)
        print(f"[planted] building natural carrier stream (k={carrier_vocab}) ...", flush=True)
        natural_stream = build_natural_stream(carrier_vocab,
                                              split=cfg["corpus"]["train_split"])
        print(f"[planted] natural stream tokens={len(natural_stream):,}", flush=True)

    results = {}
    for rule_p in pc["rule_p_sweep"]:
        gen = GenConfig(carrier_vocab=carrier_vocab, d_max=pc["d_max"],
                        rule_p=rule_p, seq_len=pc["seq_len"], n_seqs=pc["n_seqs"],
                        seed=seed, needle_mode=needle, carrier_mode=carrier_mode,
                        natural_vocab_k=carrier_vocab)
        corpus = generate(gen, natural_stream=natural_stream)
        out = base / f"{prefix}rule_p_{rule_p}{suffix}"
        _, hist = train_planted(
            corpus, preset=pc["preset"], steps=pc["steps"],
            batch_size=pc["batch_size"], eval_every=pc["eval_every"],
            out_dir=out, seed=seed)
        corpus.save(out / "corpus.npz")
        results[f"{prefix}rule_p={rule_p}{suffix}"] = hist["evals"][-1] | {
            "ngram_null": hist["ngram_null"], "oracle": hist["oracle"]}

    if pc.get("shuffle_control"):
        gen = GenConfig(carrier_vocab=pc["carrier_vocab"], d_max=pc["d_max"],
                        rule_p=1.0, seq_len=pc["seq_len"], n_seqs=pc["n_seqs"],
                        seed=seed)
        clean = generate(gen)
        ngram = max(ngram_marker_accuracy(clean, k)["accuracy"] for k in (1, 2, 4, 8))
        oracle = oracle_marker_accuracy(clean)
        ctrl = shuffle_train_rows(clean, seed)
        out = base / "shuffle_control"
        _, hist = train_planted(
            ctrl, preset=pc["preset"], steps=pc["steps"],
            batch_size=pc["batch_size"], eval_every=pc["eval_every"],
            out_dir=out, seed=seed, ngram_null=ngram, oracle=oracle)
        results["shuffle_control"] = hist["evals"][-1] | {
            "ngram_null": ngram, "oracle": oracle}

    tag = (prefix.rstrip("_") or "synthetic") + suffix
    (base / f"summary_{tag}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== planted-structure frontier: marker accuracy vs null/oracle ===")
    head = f"{'condition':18s}{'marker_acc':>12s}{'ngram_null':>12s}{'oracle':>9s}{'margin':>9s}"
    print(head); print("-" * len(head))
    for k, r in results.items():
        print(f"{k:18s}{r['marker_acc']:>12.3f}{r['ngram_null']:>12.3f}"
              f"{r['oracle']:>9.3f}{r['margin_over_ngram']:>+9.3f}")
    print(f"\n[planted] wrote {base/'summary.json'}")


if __name__ == "__main__":
    main()

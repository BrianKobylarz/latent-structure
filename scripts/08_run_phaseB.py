"""Phase B: does a tiny model trained on NATURAL text represent syntactic
dependency depth -- a genuinely natural, non-salient structural variable?

Builds a parsed WikiText corpus (word tokens + per-token dependency depth),
trains a small from-scratch LM, and probes the residual stream for depth with the
same controls as the planted phase: probe accuracy vs control task, and MDL vs a
random-init model. (Causal "use" for a natural feature -- amnesic/patching -- is
the next sub-step; this establishes "represents".)

Usage:
    python scripts/08_run_phaseB.py --config config/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml
from transformers.utils.logging import disable_progress_bar, set_verbosity_error

disable_progress_bar(); set_verbosity_error()

from latent_structure import corpus as C
from latent_structure import model as M
from latent_structure.probing import (collect_position_activations,
                                       online_codelength, probe_accuracy)
from latent_structure.syntactic_corpus import SyntacticCorpus, build_syntactic_corpus


def train_lm(corpus, preset, steps, batch_size, lr, device, seed, log=print):
    import torch
    from torch.optim.lr_scheduler import LambdaLR
    ctx = corpus.seq_len
    rng = np.random.default_rng(seed)
    cut = int(corpus.tokens.shape[0] * 0.9)
    model = M.build_raw_model(preset, corpus.vocab_size, ctx=ctx).to(device)
    pc = M.param_counts(model)
    log(f"[phaseB] vocab={corpus.vocab_size} ctx={ctx} params total={pc['total']:,} "
        f"non_emb={pc['non_embedding']:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = LambdaLR(opt, lambda s: s / 100 if s < 100 else
                     0.5 * (1 + np.cos(np.pi * (s - 100) / max(1, steps - 100))))
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    run = 0.0
    for step in range(1, steps + 1):
        idx = rng.integers(0, cut, size=batch_size)
        x = torch.from_numpy(corpus.tokens[idx].astype(np.int64)).to(device)
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
        else:
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); run += loss.item()
        if step % 500 == 0 or step == steps:
            log(f"[phaseB] step {step:5d} lm_loss={run/500:.3f}"); run = 0.0
    return model, pc, cut


def probe_depth(model, rand, corpus, rows, layer, device, seed):
    X, y = collect_position_activations(model, corpus, rows, layer, device, seed=seed)
    keep = y >= 0
    X, y = X[keep], y[keep]
    acc = probe_accuracy(X, y, seed=seed)
    mdl = online_codelength(X, y, seed=seed)
    Xr, yr = collect_position_activations(rand, corpus, rows, layer, device, seed=seed)
    keep = yr >= 0
    mdl_r = online_codelength(Xr[keep], yr[keep], seed=seed)
    return {"layer": layer, "accuracy": acc, "mdl_trained": mdl,
            "mdl_random_init": mdl_r,
            "mdl_ratio_rand_over_trained": mdl_r["total_bits"] / mdl["total_bits"],
            "n_classes": int(y.max() + 1), "majority": float(np.bincount(y).max() / len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    bcfg = cfg.get("phaseB", {})

    import torch
    from transformers import GPT2LMHeadModel
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    out = runs_dir / "phaseB"; out.mkdir(parents=True, exist_ok=True)
    cache = out / "syntactic_corpus.npz"
    if cache.exists():
        corpus = SyntacticCorpus.load(cache); nsent = "cached"
    else:
        print("[phaseB] parsing WikiText for dependency depth ...", flush=True)
        t0 = time.time()
        corpus, nsent = build_syntactic_corpus(
            n_sentences=bcfg.get("n_sentences", 30000),
            vocab_k=bcfg.get("vocab_k", 512),
            seq_len=bcfg.get("seq_len", 256),
            max_documents=bcfg.get("max_documents", 2000))
        corpus.save(cache)
        print(f"[phaseB] parsed {nsent} sentences -> {corpus.tokens.shape} in "
              f"{time.time()-t0:.0f}s", flush=True)

    # depth distribution (for context)
    d = corpus.depth_after[corpus.depth_after >= 0]
    print("[phaseB] depth dist:", np.round(np.bincount(d) / d.size, 3).tolist())

    model, pc, cut = train_lm(corpus, bcfg.get("preset", "tiny"),
                              bcfg.get("steps", 3000), bcfg.get("batch_size", 64),
                              bcfg.get("lr", 6e-4), device, seed)
    rand = GPT2LMHeadModel(model.config).to(device)
    test_rows = np.arange(cut, corpus.tokens.shape[0])

    results = {"params": pc, "layers": {}}
    for layer in range(1, model.config.n_layer + 1):
        rep = probe_depth(model, rand, corpus, test_rows, layer, device, seed)
        results["layers"][f"layer{layer}"] = rep
        print(f"[phaseB] L{layer}: depth-probe acc real={rep['accuracy']['real']:.3f} "
              f"ctrl={rep['accuracy']['control_task']:.3f} majority={rep['majority']:.3f} | "
              f"MDL trained={rep['mdl_trained']['total_bits']:.0f}b "
              f"rand={rep['mdl_random_init']['total_bits']:.0f}b "
              f"(rand/trained={rep['mdl_ratio_rand_over_trained']:.2f})", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    model.save_pretrained(out / "model")
    print(f"\n[phaseB] wrote {out/'results.json'}")


if __name__ == "__main__":
    main()

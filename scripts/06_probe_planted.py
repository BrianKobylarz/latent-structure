"""Phase A2: probe the planted-structure models for the latent depth.

For each trained planted model, fit a linear probe for latent depth at each
layer and report: probe accuracy (real vs control task), MDL/codelength for the
trained model, and MDL for a random-init model of identical architecture. The
trustworthy evidence is MDL (trained << random-init, real << control), NOT
accuracy alone.

Usage:
    python scripts/06_probe_planted.py --config config/pilot.yaml
    python scripts/06_probe_planted.py --config config/pilot.yaml --conditions rule_p_1.0
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
from latent_structure.probing import probe_report
from latent_structure.synthetic_corpus import Corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of planted run dirs (default: all with a corpus.npz)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import GPT2LMHeadModel
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    base = runs_dir / "planted"
    dirs = sorted(d for d in base.glob("*") if (d / "corpus.npz").exists())
    if args.conditions:
        dirs = [d for d in dirs if d.name in args.conditions]
    if not dirs:
        sys.exit(f"no planted runs with corpus.npz in {base}; run 05_run_planted.py first")

    results = {}
    for d in dirs:
        corpus = Corpus.load(d / "corpus.npz")
        n = corpus.tokens.shape[0]
        test_rows = np.arange(int(n * 0.9), n)
        model = GPT2LMHeadModel.from_pretrained(d).to(device)
        rand = GPT2LMHeadModel(model.config).to(device)   # random-init, same arch
        n_layer = model.config.n_layer
        # probe every layer for the deterministic run; last layer for others
        layers = range(1, n_layer + 1) if d.name == "rule_p_1.0" else [n_layer]
        cond = {}
        for layer in layers:
            rep = probe_report(model, rand, corpus, test_rows, layer, device, seed=seed)
            cond[f"layer{layer}"] = rep
            print(f"[probe {d.name} L{layer}] acc real={rep['accuracy']['real']:.3f} "
                  f"ctrl={rep['accuracy']['control_task']:.3f} | MDL trained="
                  f"{rep['mdl_trained']['total_bits']:.0f}b rand="
                  f"{rep['mdl_random_init']['total_bits']:.0f}b "
                  f"(rand/trained={rep['mdl_ratio_rand_over_trained']:.2f})", flush=True)
        results[d.name] = cond
        del model, rand
        if device == "cuda":
            torch.cuda.empty_cache()

    (base / "probe_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[probe] wrote {base/'probe_results.json'}")


if __name__ == "__main__":
    main()

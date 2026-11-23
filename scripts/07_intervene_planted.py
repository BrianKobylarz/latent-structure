"""Phase A3: causal interchange intervention on the planted-structure model.

Patches the residual at each layer with a donor from the opposite depth side and
reports the flip rate vs a same-side control. Opposite-side flip >> same-side
control = the residual causally carries the depth used by the marker decision.

Usage:
    python scripts/07_intervene_planted.py --config config/pilot.yaml
    python scripts/07_intervene_planted.py --config config/pilot.yaml --condition rule_p_1.0
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
from latent_structure.intervention import interchange_flip_rate
from latent_structure.synthetic_corpus import Corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--condition", default="rule_p_1.0")
    ap.add_argument("--n-pairs", type=int, default=800)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import GPT2LMHeadModel
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    d = runs_dir / "planted" / args.condition
    if not (d / "corpus.npz").exists():
        sys.exit(f"{d} has no corpus.npz; run 05_run_planted.py first")
    corpus = Corpus.load(d / "corpus.npz")
    model = GPT2LMHeadModel.from_pretrained(d).to(device)

    print(f"=== causal interchange intervention: {args.condition} "
          f"(thresh={corpus.config.effective_thresh}) ===")
    print(f"{'layer':>6s}{'opp_flip':>10s}{'low->high':>11s}{'high->low':>11s}"
          f"{'same_ctrl':>11s}")
    results = {}
    for layer in range(1, model.config.n_layer + 1):
        r = interchange_flip_rate(model, corpus, layer, device,
                                  n_pairs=args.n_pairs, seed=seed)
        results[f"layer{layer}"] = r
        print(f"{layer:>6d}{r['opposite_side_flip']:>10.3f}{r['flip_low_to_high']:>11.3f}"
              f"{r['flip_high_to_low']:>11.3f}{r['same_side_flip_control']:>11.3f}",
              flush=True)

    (d / "intervention_results.json").write_text(json.dumps(results, indent=2),
                                                 encoding="utf-8")
    print(f"\n[intervene] wrote {d/'intervention_results.json'}")


if __name__ == "__main__":
    main()

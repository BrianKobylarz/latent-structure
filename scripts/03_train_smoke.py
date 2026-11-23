"""Phase 3 smoke run: train one small model per variant on its perturbed TRAIN
data, evaluate every model on the SAME clean validation set, and collate the
position-decomposed NLL into one comparison table.

The eval set (and tokenizer) are built once and shared across all models so the
numbers are directly comparable.

Usage:
    python scripts/03_train_smoke.py --config config/pilot.yaml
    python scripts/03_train_smoke.py --config config/pilot.yaml --variants clean modal_polarity control_mask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from latent_structure import corpus as C
from latent_structure import model as M
from latent_structure.evaluate import CATEGORIES, build_eval_set
from latent_structure.train import train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of variants to train (default: clean + all configured)")
    ap.add_argument("--size", default=None, help="override training.size (tiny/small/medium)")
    ap.add_argument("--steps", type=int, default=None, help="override training.steps")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    variants = args.variants or (["clean"] + list(cfg["interventions"]))

    tcfg = cfg["training"]
    if args.size:
        tcfg["size"] = args.size
    if args.steps:
        tcfg["steps"] = args.steps
    tokenizer = M.build_tokenizer()
    print("[smoke] building shared clean eval set ...", flush=True)
    eval_set = build_eval_set(cfg, tokenizer, split=cfg["corpus"]["eval_split"],
                              max_documents=tcfg["eval_max_documents"],
                              window=tcfg["post_modal_window"])
    print(f"[smoke] eval docs={len(eval_set)}", flush=True)

    results = {}
    for v in variants:
        hist = train_one(cfg, v, eval_set=eval_set, tokenizer=tokenizer)
        results[v] = hist["evals"][-1]  # final eval

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    summary_path = runs_dir / f"smoke_summary_{tcfg['size']}.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # comparison table: clean-validation NLL per category, per training variant
    cats = [c for c in CATEGORIES]
    print("\n=== clean-validation NLL by training variant (lower = better) ===")
    head = f"{'train_variant':20s}" + "".join(f"{c:>12s}" for c in cats)
    print(head)
    print("-" * len(head))
    clean_row = results.get("clean", {})
    for v in variants:
        r = results[v]
        cells = "".join(f"{r[c]['nll']:>12.4f}" for c in cats)
        print(f"{v:20s}{cells}")
    # deltas vs clean-trained model, on the modal-sensitive buckets
    if clean_row:
        print("\n=== delta vs clean-trained (positive = worse than clean training) ===")
        focus = ["global", "core_modal", "post_modal", "control"]
        head = f"{'train_variant':20s}" + "".join(f"{c:>12s}" for c in focus)
        print(head)
        print("-" * len(head))
        for v in variants:
            r = results[v]
            cells = "".join(f"{r[c]['nll'] - clean_row[c]['nll']:>+12.4f}" for c in focus)
            print(f"{v:20s}{cells}")
    print(f"\n[smoke] wrote {summary_path}")


if __name__ == "__main__":
    main()

"""Phase 4: score every trained model on the modal minimal-pair set.

Builds (or loads) a custom modal minimal-pair set from CLEAN held-out text, then
for each runs/<variant>_<size> checkpoint reports minimal-pair accuracy per
contrast type (cross / polarity / within). The clean-trained model is the
validity/ceiling check: if it cannot beat chance, the items are uninformative.

Usage:
    python scripts/04_eval_minimal_pairs.py --config config/pilot.yaml
    python scripts/04_eval_minimal_pairs.py --config config/pilot.yaml --size tiny
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
from latent_structure.minimal_pairs import build_minimal_pairs, score_model_on_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--size", default=None, help="model size suffix (default: training.size)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    size = args.size or cfg["training"]["size"]
    mp_cfg = cfg.get("minimal_pairs", {})
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[mp] building/loading minimal-pair set ...", flush=True)
    items = build_minimal_pairs(
        cfg, seed=seed,
        max_sentences=mp_cfg.get("max_sentences", 4000),
        max_items_per_contrast=mp_cfg.get("max_items_per_contrast", 800))
    from collections import Counter
    by_type = Counter(it["contrast_type"] for it in items)
    print(f"[mp] {len(items)} items: " + ", ".join(f"{k}={v}" for k, v in by_type.items()))

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    run_dirs = sorted(p for p in runs_dir.glob(f"*_{size}") if (p / "config.json").exists())
    if not run_dirs:
        sys.exit(f"no trained {size} models in {runs_dir}; run 03_train_smoke.py first")

    results = {}
    contrasts = ["cross", "polarity", "within"]
    for rd in run_dirs:
        variant = rd.name[: -(len(size) + 1)]
        model = AutoModelForCausalLM.from_pretrained(rd).to(device)
        tok = AutoTokenizer.from_pretrained(rd)
        res = score_model_on_pairs(model, tok, items, device)
        results[variant] = res
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        cells = "  ".join(
            f"{c}={res['by_contrast'].get(c, {}).get('accuracy', float('nan')):.3f}"
            for c in contrasts)
        print(f"[mp] {variant:20s} {cells}", flush=True)

    out = runs_dir / f"minimal_pairs_results_{size}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # comparison table: accuracy (prefers original) per contrast, per training variant
    print("\n=== minimal-pair accuracy (prefers ORIGINAL over contrast; "
          "0.5 = chance) ===")
    head = f"{'train_variant':20s}" + "".join(f"{c:>11s}" for c in contrasts)
    print(head)
    print("-" * len(head))
    for v in sorted(results):
        r = results[v]["by_contrast"]
        cells = "".join(
            f"{r.get(c, {}).get('accuracy', float('nan')):>11.3f}" for c in contrasts)
        print(f"{v:20s}{cells}")

    # cross-swap broken down by force axis, for the clean-trained ceiling model
    if "clean" in results and results["clean"]["by_cross_axis"]:
        print("\n=== clean-trained model, cross-swap accuracy by force axis ===")
        for axis, a in results["clean"]["by_cross_axis"].items():
            print(f"  {axis:34s} acc={a['accuracy']:.3f}  (n={a['n']})")
    print(f"\n[mp] wrote {out}")


if __name__ == "__main__":
    main()

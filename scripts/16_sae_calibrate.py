"""D-1: SAE calibration on the PLANTED model.

Before trusting an SAE to surface unlabeled structure, check it recovers the
KNOWN planted latent (depth). We train an SAE on the planted model's residual
stream and ask: (a) does it reconstruct (low FVU) sparsely (low L0)? (b) does some
sparse FEATURE track the ground-truth depth (|corr|)? (c) does the sparse code
preserve depth under a linear readout (Spearman) ~ as well as raw activations?

If yes -> SAEs are usable at our scale and D-2 (discovery on natural text) is
warranted. If no -> honest negative; SAE-based discovery is unreliable here.

Usage:
    python scripts/16_sae_calibrate.py --config config/pilot.yaml --condition rule_p_1.0
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

disable_progress_bar(); set_verbosity_error()

from latent_structure import corpus as C
from latent_structure import sae as SAE
from latent_structure.probing import collect_position_activations
from latent_structure.synthetic_corpus import Corpus


def linear_readout_spearman(F, y, seed=0):
    """Spearman corr of a linear regression F->y on held-out (does the sparse
    code preserve the scalar?)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from scipy.stats import spearmanr
    Ftr, Fte, ytr, yte = train_test_split(F, y, test_size=0.3, random_state=seed)
    pred = Ridge(alpha=1.0).fit(Ftr, ytr).predict(Fte)
    return float(spearmanr(pred, yte).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--condition", default="rule_p_1.0")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--l1", type=float, default=1e-3)
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
    n = corpus.tokens.shape[0]
    rows = np.arange(int(n * 0.9))

    print(f"[d1] collecting planted activations (layer {args.layer}) ...", flush=True)
    X, depth = collect_position_activations(model, corpus, rows, args.layer, device,
                                            max_positions=40000, seed=seed)
    print(f"[d1] X={X.shape}, training SAE (expansion {args.expansion}, l1 {args.l1}) ...",
          flush=True)
    sae, stats = SAE.train_sae(X, expansion=args.expansion, l1=args.l1, device=device,
                               seed=seed, log=print)

    F = SAE.encode(sae, X, device=device)
    best_j, best_corr, corrs = SAE.feature_label_corr(F, depth)
    n_corr = int((np.abs(corrs) > 0.3).sum())
    readout_F = linear_readout_spearman(F, depth, seed=seed)
    readout_X = linear_readout_spearman(X, depth, seed=seed)

    res = {"condition": args.condition, "layer": args.layer, **stats,
           "best_feature": best_j, "best_feature_depth_corr": best_corr,
           "n_features_corr_gt_0.3": n_corr,
           "depth_spearman_from_sae_code": readout_F,
           "depth_spearman_from_raw_acts": readout_X}
    print(f"\n[d1] FVU={stats['fvu']:.3f} L0={stats['l0']:.1f} "
          f"({stats['n_features']} features)")
    print(f"[d1] best single feature |corr| with depth = {best_corr:.3f} "
          f"(feature {best_j}); {n_corr} features |corr|>0.3")
    print(f"[d1] depth Spearman: from SAE code {readout_F:.3f} vs from raw acts {readout_X:.3f}")
    verdict = ("SAE RECOVERS the planted latent" if best_corr > 0.5 and readout_F > 0.7 * readout_X
               else "SAE does NOT clearly recover the planted latent")
    print(f"[d1] VERDICT: {verdict}")
    res["verdict"] = verdict

    (runs_dir / f"sae_calibrate_{args.condition}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"[d1] wrote sae_calibrate_{args.condition}.json")


if __name__ == "__main__":
    main()

"""#3 (force geometry): does a clean-trained modal model represent modal FORCE as
an ORDERED axis (POSSIBILITY < WEAK_NECESSITY < NECESSITY)?

We collect residual-stream activations at core-modal token positions on clean
held-out text, labeled by force class, and test:
  - representation: linear probe accuracy + MDL vs control task vs random-init
    (per Phase B, MDL is the trustworthy signal, not accuracy);
  - ordering: (a) a 1-D regression axis whose per-class means should be monotonic
    POSSIBILITY < WEAK_NECESSITY < NECESSITY, with Spearman corr vs the ordinal
    label; (b) confusion structure -- if force is ordered, the DISTANT pair
    (POSSIBILITY vs NECESSITY) should be confused LESS than adjacent pairs.

Usage:
    python scripts/09_force_geometry.py --config config/pilot.yaml --size small
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
from latent_structure.modals import (NECESSITY, POSSIBILITY, WEAK_NECESSITY,
                                      detect_spans, get_nlp, is_core)
from latent_structure.probing import online_codelength, probe_accuracy

FORCE3 = {POSSIBILITY: 0, WEAK_NECESSITY: 1, NECESSITY: 2}
NAMES = ["POSSIBILITY", "WEAK_NECESSITY", "NECESSITY"]


def _harvest_sents(docs):
    nlp = get_nlp()
    sents = []
    for doc in nlp.pipe(docs, batch_size=32):
        for s in doc.sents:
            if 6 <= len(s) <= 80:
                sents.append(s.text.strip())
    return sents


def collect_force_activations(model, tok, sents, layer, device, max_positions=8000):
    """Collect activations + force labels at TWO positions per modal:
    'modal' = the modal token's own position (token-identity-confounded), and
    'pred'  = the position just BEFORE the modal (hasn't seen the token, so any
    force signal is genuinely contextual/anticipatory -- the clean test)."""
    import torch
    feats = {"modal": [], "pred": []}
    labels = {"modal": [], "pred": []}
    model.eval()
    with torch.no_grad():
        for sent in sents:
            spans = [sp for sp in detect_spans(sent)
                     if is_core(sp.lemma) and sp.force in FORCE3]
            if not spans:
                continue
            enc = tok(sent, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) < 2 or len(ids) > model.config.n_positions:
                continue
            x = torch.tensor([ids], device=device)
            h = model(input_ids=x, output_hidden_states=True).hidden_states[layer][0]
            for sp in spans:
                idxs = [j for j, (a, b) in enumerate(offs)
                        if a != b and not (b <= sp.char_start or a >= sp.char_end)]
                if not idxs:
                    continue
                lab = FORCE3[sp.force]
                feats["modal"].append(h[idxs[-1]].float().cpu().numpy())
                labels["modal"].append(lab)
                if idxs[0] - 1 >= 0:                     # prediction position
                    feats["pred"].append(h[idxs[0] - 1].float().cpu().numpy())
                    labels["pred"].append(lab)
            if len(feats["modal"]) >= max_positions:
                break
    return ({k: np.array(v) for k, v in feats.items()},
            {k: np.array(v) for k, v in labels.items()})


def ordering_analysis(X, y, seed=0):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from scipy.stats import spearmanr
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    # 1-D ordinal axis
    reg = LinearRegression().fit(Xtr, ytr)
    proj = reg.predict(Xte)
    means = [float(proj[yte == c].mean()) for c in (0, 1, 2)]
    rho = float(spearmanr(proj, yte).statistic)
    # confusion structure
    clf = LogisticRegression(max_iter=300, multi_class="multinomial").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    cm = np.zeros((3, 3))
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    cm = cm / cm.sum(1, keepdims=True)
    distant = (cm[0, 2] + cm[2, 0]) / 2          # POSSIBILITY <-> NECESSITY
    adjacent = (cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]) / 4
    return {"axis_means": means, "monotonic": means[0] < means[1] < means[2],
            "spearman_proj_vs_ordinal": rho,
            "distant_confusion": float(distant), "adjacent_confusion": float(adjacent),
            "confusion_matrix": cm.round(3).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--size", default="small")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    rd = runs_dir / f"clean_{args.size}"
    if not (rd / "config.json").exists():
        sys.exit(f"no clean_{args.size} model; train it first")
    model = AutoModelForCausalLM.from_pretrained(rd).to(device)
    tok = AutoTokenizer.from_pretrained(rd)
    rand = GPT2LMHeadModel(model.config).to(device)

    docs = C.build_corpus(split=cfg["corpus"]["eval_split"], max_documents=300,
                          min_doc_chars=cfg["corpus"].get("min_doc_chars", 500))
    sents = _harvest_sents(docs)

    results = {}
    print(f"=== force geometry (redesigned): clean_{args.size} ===")
    print("  pred-pos MDL ratio = clean (token-unconfounded) test; "
          "modal-pos ordering compared trained-vs-random")
    for layer in range(1, model.config.n_layer + 1):
        Xt, yt = collect_force_activations(model, tok, sents, layer, device)
        Xr, yr = collect_force_activations(rand, tok, sents, layer, device)
        if len(yt["modal"]) < 200:
            print(f"  L{layer}: too few modal positions"); continue
        rec = {"n_modal": int(len(yt["modal"])), "n_pred": int(len(yt["pred"]))}
        # (a) clean token-unconfounded test: prediction position
        accp = probe_accuracy(Xt["pred"], yt["pred"], seed=seed)
        mdlp = online_codelength(Xt["pred"], yt["pred"], seed=seed)
        mdlp_r = online_codelength(Xr["pred"], yr["pred"], seed=seed)
        rec["pred_pos"] = {"acc": accp, "mdl_ratio": mdlp_r["total_bits"] / mdlp["total_bits"]}
        # (b) modal-position ordering: trained vs random-init
        ot = ordering_analysis(Xt["modal"], yt["modal"], seed=seed)
        orr = ordering_analysis(Xr["modal"], yr["modal"], seed=seed)
        mdm = online_codelength(Xt["modal"], yt["modal"], seed=seed)
        mdm_r = online_codelength(Xr["modal"], yr["modal"], seed=seed)
        rec["modal_pos"] = {"mdl_ratio": mdm_r["total_bits"] / mdm["total_bits"],
                            "ordered_trained": ot, "ordered_random": orr}
        results[f"layer{layer}"] = rec
        print(f"  L{layer}: PRED-pos acc={accp['real']:.3f} MDL_ratio={rec['pred_pos']['mdl_ratio']:.2f}"
              f"  | MODAL-pos MDL_ratio={rec['modal_pos']['mdl_ratio']:.2f}"
              f" ordered(trained rho={ot['spearman_proj_vs_ordinal']:+.2f} dist-adj="
              f"{ot['distant_confusion']-ot['adjacent_confusion']:+.3f}; "
              f"random rho={orr['spearman_proj_vs_ordinal']:+.2f} dist-adj="
              f"{orr['distant_confusion']-orr['adjacent_confusion']:+.3f})", flush=True)

    (runs_dir / f"force_geometry_{args.size}_v2.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[force-geometry] wrote force_geometry_{args.size}_v2.json")


if __name__ == "__main__":
    main()

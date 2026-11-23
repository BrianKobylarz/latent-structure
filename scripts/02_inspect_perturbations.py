"""Inspect and compare clean vs perturbed corpus variants -- NO GPU, NO API.

This is the first checkpoint: before spending any GPU time, quantify what each
perturbation actually did to the text. For every variant it reports:

  - coverage: how many target tokens were perturbed (applied / targets)
  - size deltas: token-count change vs clean (deletions shrink, "must not"-style
    polarity flips grow)
  - unigram divergence from clean: Jensen-Shannon divergence over word types.
    The frequency-matched control should sit near the modal-mask variant on a
    pure-frequency basis -- if a modal intervention diverges much MORE than the
    control, that excess is attributable to modal-specific structure rather than
    to perturbing frequent tokens.
  - a few worked before/after examples per variant.

Usage:
    python scripts/02_inspect_perturbations.py --config config/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# WikiText contains non-cp1252 characters; force UTF-8 so printing examples on
# the Windows console does not crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from latent_structure import corpus as C

_WORD_RE = C._WORD_RE


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def unigram_counts(records) -> Counter:
    freq: Counter = Counter()
    for r in records:
        freq.update(_WORD_RE.findall(r["text"].lower()))
    return freq


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two count dists."""
    vocab = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    js = 0.0
    for w in vocab:
        pi, qi = p.get(w, 0) / tp, q.get(w, 0) / tq
        mi = 0.5 * (pi + qi)
        if pi > 0:
            js += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            js += 0.5 * qi * math.log2(qi / mi)
    return js


def token_total(records) -> int:
    return sum(len(_WORD_RE.findall(r["text"].lower())) for r in records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--examples", type=int, default=3,
                    help="worked before/after examples per variant")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ccfg = cfg["corpus"]
    split = args.split or ccfg["train_split"]
    data_dir, _ = C.resolve_storage(cfg["storage_root"])
    var_dir = data_dir / ccfg["name"] / split
    if not (var_dir / "clean.jsonl").exists():
        sys.exit(f"no variants found in {var_dir}; run 01_build_datasets.py first")

    clean = read_jsonl(var_dir / "clean.jsonl")
    clean_freq = unigram_counts(clean)
    clean_tokens = token_total(clean)
    manifest = json.loads((var_dir / "manifest.json").read_text(encoding="utf-8"))

    print(f"\nCorpus: {ccfg['name']} / {split}")
    print(f"Documents: {len(clean)}   Clean word tokens: {clean_tokens:,}")
    print(f"Modal occurrences: {manifest['total_modal_occurrences']:,}   "
          f"Control-matched: {manifest['total_control_occurrences']:,}")
    print("\nTop modal lemmas: " + ", ".join(
        f"{k}={v}" for k, v in list(manifest["modal_lemma_counts"].items())[:12]))

    header = f"\n{'variant':18s} {'applied/targets':>16s} {'tok_delta%':>11s} {'JSD_vs_clean':>13s}"
    print(header)
    print("-" * len(header))
    rows = []
    for iv in cfg["interventions"]:
        path = var_dir / f"{iv}.jsonl"
        if not path.exists():
            continue
        recs = read_jsonl(path)
        freq = unigram_counts(recs)
        toks = token_total(recs)
        jsd = js_divergence(clean_freq, freq)
        st = manifest["variant_stats"].get(iv, {})
        cov = f"{st.get('n_applied','?')}/{st.get('n_targets','?')}"
        delta = 100.0 * (toks - clean_tokens) / max(clean_tokens, 1)
        rows.append((iv, jsd))
        print(f"{iv:18s} {cov:>16s} {delta:>10.2f}% {jsd:>13.5f}")

    # Highlight the confound check explicitly.
    jsd_map = dict(rows)
    if "modal_mask" in jsd_map and "control_mask" in jsd_map:
        mm, cm = jsd_map["modal_mask"], jsd_map["control_mask"]
        print(f"\nConfound check (unigram level): "
              f"modal_mask JSD={mm:.5f} vs control_mask JSD={cm:.5f}")
        print("  (These should be similar at the unigram level since both just "
              "replace tokens with the sentinel. Divergence in DOWNSTREAM model\n"
              "  loss between them -- once trained -- is the modal-specific signal.)")

    print("\n--- worked examples (window around first edit) ---")
    for iv in cfg["interventions"]:
        path = var_dir / f"{iv}.jsonl"
        if not path.exists():
            continue
        recs = read_jsonl(path)
        shown = 0
        print(f"\n[{iv}]")
        for cr, pr in zip(clean, recs):
            c, p = cr["text"], pr["text"]
            if c == p:
                continue
            # locate the first character where the two texts diverge
            i = 0
            lim = min(len(c), len(p))
            while i < lim and c[i] == p[i]:
                i += 1
            a, b = max(0, i - 50), i + 60
            ct = c[a:b].replace("\n", " ")
            pt = p[a:b].replace("\n", " ")
            print(f"  CLEAN: ...{ct}...")
            print(f"  PERT : ...{pt}...")
            shown += 1
            if shown >= args.examples:
                break


if __name__ == "__main__":
    main()

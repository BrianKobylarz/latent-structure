"""Build the clean corpus and all perturbed variants.

One spaCy detection pass over the corpus is cached and reused by every modal
intervention. Each variant is written as JSONL (one document per line) to
<storage_root>/data/<corpus>/<intervention>.jsonl, alongside a manifest with
build provenance and per-variant edit statistics.

Usage:
    python scripts/01_build_datasets.py --config config/pilot.yaml
    python scripts/01_build_datasets.py --config config/pilot.yaml --split validation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# allow running as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from latent_structure import corpus as C
from latent_structure import perturb as P
from latent_structure.modals import detect_spans_iter


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--split", default=None,
                    help="override which corpus split to build (train/validation)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    ccfg = cfg["corpus"]
    split = args.split or ccfg["train_split"]

    data_dir, _ = C.resolve_storage(cfg["storage_root"])
    out_dir = data_dir / ccfg["name"] / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] loading {ccfg['name']} split={split} ...", flush=True)
    docs = C.build_corpus(split=split,
                          max_documents=ccfg.get("max_documents"),
                          min_doc_chars=ccfg.get("min_doc_chars", 500))
    print(f"[build] {len(docs)} documents", flush=True)

    # one detection pass, reused by all modal interventions
    t0 = time.time()
    print("[build] detecting modal spans (spaCy, batched) ...", flush=True)
    spans_per_doc = list(detect_spans_iter(docs))
    print(f"[build] detection done in {time.time()-t0:.1f}s", flush=True)

    freq = C.word_frequencies(docs)
    modal_counts = C.modal_lemma_counts(spans_per_doc)
    control_vocab_map = C.build_control_vocab(modal_counts, freq)
    control_vocab = set(control_vocab_map)
    rng = random.Random(seed)
    random_vocab = C.build_random_vocab(rng, size=int(cfg.get("random_vocab_size", 10000)))

    total_modals = sum(modal_counts.values())
    total_control = sum(control_vocab_map.values())
    print(f"[build] modal occurrences={total_modals} across {len(modal_counts)} lemmas; "
          f"control set matches {total_control} occurrences across {len(control_vocab_map)} types",
          flush=True)

    # always write the clean corpus
    write_jsonl(out_dir / "clean.jsonl",
                ({"id": i, "text": d} for i, d in enumerate(docs)))

    variant_stats: dict[str, dict] = {}
    for iv in cfg["interventions"]:
        t1 = time.time()
        records = []
        n_targets = n_applied = 0
        for i, d in enumerate(docs):
            kwargs = {"seed": seed + i}
            if iv in P.MODAL_INTERVENTIONS:
                kwargs["spans"] = spans_per_doc[i]
            elif iv == P.CONTROL_MASK:
                kwargs["control_vocab"] = control_vocab
            elif iv == P.RANDOM_STRINGS:
                kwargs["random_vocab"] = random_vocab
            pd = P.perturb_document(d, iv, **kwargs)
            n_targets += pd.n_targets
            n_applied += pd.n_applied
            records.append({"id": i, "text": pd.text,
                            "n_targets": pd.n_targets, "n_applied": pd.n_applied})
        write_jsonl(out_dir / f"{iv}.jsonl", records)
        variant_stats[iv] = {"n_targets": n_targets, "n_applied": n_applied}
        print(f"[build] {iv:18s} applied {n_applied}/{n_targets} in {time.time()-t1:.1f}s",
              flush=True)

    manifest = {
        "config": cfg,
        "split": split,
        "n_documents": len(docs),
        "seed": seed,
        "modal_lemma_counts": dict(modal_counts.most_common()),
        "total_modal_occurrences": total_modals,
        "control_vocab": control_vocab_map,
        "total_control_occurrences": total_control,
        "variant_stats": variant_stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] wrote variants + manifest to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

"""The depth test: Hewitt & Manning (2019) structural NORM probe.

A multiclass-integer-depth logistic probe scored by codelength (scripts/10) reads
depth as only weakly represented. The canonical test predicts tree depth as the
squared L2 norm under a LEARNED linear map B:

    depth_hat(word) = || B h_word ||^2 ,  B fit by L1 regression to integer depth.

Scored per sentence by:
  - root accuracy: argmin_word ||Bh||^2 == the true depth-0 (root) word
  - norm Spearman (NSpr): rank-corr between ||Bh||^2 and true depth

Reference points (Hewitt & Manning, BERT/ELMo encoders): strong ~0.85-0.89 NSpr,
~86-90% root; LINEAR baseline ~0.27 / 2.9%. We report trained vs random-init
(GPT-2 family) and absolute for 4-bit local models.

Word representation = last subtoken of each word (offset-aligned to spaCy words).

Usage:
    python scripts/13_structural_probe.py --config config/pilot.yaml --model gpt2
    python scripts/13_structural_probe.py --config config/pilot.yaml --model Qwen/Qwen2.5-7B-Instruct --load-4bit --no-random-init
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
from latent_structure.structural import (collect_sentences, eval_norm_probe,
                                         train_norm_probe)
from latent_structure.syntactic_corpus import parse_depth_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-sentences", type=int, default=1500)
    ap.add_argument("--label", default="depth", choices=["depth", "head_dist"])
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-random-init", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(name):
        if args.load_4bit:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            return AutoModelForCausalLM.from_pretrained(name, quantization_config=qc,
                                                        device_map="cuda")
        return AutoModelForCausalLM.from_pretrained(name).to(device)

    root = args.label == "depth"
    print(f"[struct] parsing {args.label}-labeled sentences ...", flush=True)
    sents = parse_depth_sentences(n_sentences=args.n_sentences,
                                  split=cfg["corpus"]["eval_split"], label=args.label)
    cut = int(len(sents) * 0.7)
    print(f"[struct] {len(sents)} sentences (train {cut} / test {len(sents)-cut})", flush=True)

    print(f"[struct] loading {args.model} (4bit={args.load_4bit}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load(args.model)
    d_model = model.config.hidden_size
    n_layer = getattr(model.config, "num_hidden_layers", None) or model.config.n_layer
    layers = sorted(set([round(n_layer * 0.5), round(n_layer * 0.75)]))

    results = {"model": args.model, "label": args.label, "n_layer": n_layer, "layers": {}}
    for L in layers:
        data = collect_sentences(model, tok, sents, L, device)
        tr, te = data[:int(len(data) * 0.7)], data[int(len(data) * 0.7):]
        B = train_norm_probe(tr, d_model, device, seed=seed)
        trained = eval_norm_probe(B, te, device, root=root)
        rec = {"trained": trained}
        msg = f"[struct] {args.model} L{L} ({args.label}): trained NSpr={trained['norm_spearman']:.3f}"
        if not args.no_random_init:
            rand = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model)).to(device)
            rdata = collect_sentences(rand, tok, sents, L, device)
            rtr, rte = rdata[:int(len(rdata) * 0.7)], rdata[int(len(rdata) * 0.7):]
            Br = train_norm_probe(rtr, d_model, device, seed=seed)
            rec["random_init"] = eval_norm_probe(Br, rte, device, root=root)
            msg += f" | random NSpr={rec['random_init']['norm_spearman']:.3f}"
            del rand
            if device == "cuda":
                torch.cuda.empty_cache()
        results["layers"][f"layer{L}"] = rec
        print(msg, flush=True)

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    suffix = "" if args.label == "depth" else f"_{args.label}"
    out = runs_dir / f"structural_probe_{args.model.replace('/', '_')}{suffix}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[struct] wrote {out}")


if __name__ == "__main__":
    main()

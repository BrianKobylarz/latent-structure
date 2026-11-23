"""Phase B-2: does a CAPABLE PRETRAINED model represent syntactic dependency
depth, where our tiny from-scratch model did not (MDL ~1.0)?

Probes a pretrained model (default GPT-2 124M) for per-token dependency depth on
parsed WikiText sentences (BPE tokens aligned to spaCy word depth via offsets),
with the same MDL + control-task + random-init controls. If the pretrained model
shows MDL >> 1 where our tiny model showed ~1.0, the Phase B negative localizes to
SCALE, not method.

Usage:
    python scripts/10_probe_pretrained_depth.py --config config/pilot.yaml --model gpt2
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
from latent_structure.probing import online_codelength, probe_accuracy
from latent_structure.syntactic_corpus import parse_depth_sentences


def collect(model, tok, sents, layers, device, max_positions=10000):
    import torch
    model.eval()
    feats = {L: [] for L in layers}
    labels = []
    with torch.no_grad():
        for text, spans in sents:
            char_depth = np.full(len(text), -1, dtype=np.int64)
            for cs, ce, d in spans:
                char_depth[cs:ce] = d
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) < 2 or len(ids) > 1024:
                continue
            pos, lab = [], []
            for j, (a, b) in enumerate(offs):
                if a < len(char_depth) and char_depth[a] >= 0:
                    pos.append(j); lab.append(int(char_depth[a]))
            if not pos:
                continue
            x = torch.tensor([ids], device=device)
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
            for L in layers:
                h = hs[L][0].float().cpu().numpy()
                feats[L].append(h[pos])
            labels.extend(lab)
            if len(labels) >= max_positions:
                break
    Xs = {L: np.concatenate(v) for L, v in feats.items()}
    y = np.asarray(labels)
    return Xs, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-sentences", type=int, default=3000)
    ap.add_argument("--label", default="depth", choices=["depth", "head_dist", "dep"])
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the model in 4-bit (bitsandbytes) -- for 7B+ on 16GB")
    ap.add_argument("--no-random-init", action="store_true",
                    help="skip the random-init twin (won't fit for 7B+); use the "
                         "self-contained MDL COMPRESSION metric (uniform/trained) instead")
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

    print(f"[B2] parsing {args.label}-labeled sentences ...", flush=True)
    sents = parse_depth_sentences(n_sentences=args.n_sentences,
                                  split=cfg["corpus"]["eval_split"], label=args.label)
    print(f"[B2] {len(sents)} sentences", flush=True)

    print(f"[B2] loading {args.model} (4bit={args.load_4bit}, "
          f"random_init={'no' if args.no_random_init else 'yes'}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load(args.model)
    cfg_m = model.config
    n_layer = getattr(cfg_m, "num_hidden_layers", None) or cfg_m.n_layer
    layers = sorted(set([max(1, n_layer // 4), n_layer // 2,
                         3 * n_layer // 4, n_layer]))

    Xs, y = collect(model, tok, sents, layers, device)
    Xr = yr = None
    if not args.no_random_init:
        rand = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model)).to(device)
        Xr, yr = collect(rand, tok, sents, layers, device)
    print(f"[B2] positions={len(y)} {args.label} classes={int(y.max()+1)}", flush=True)

    results = {"model": args.model, "label": args.label, "n_layer": n_layer,
               "n_positions": int(len(y)), "metric":
               "compression(uniform/trained)" if args.no_random_init else "rand/trained",
               "layers": {}}
    for L in layers:
        acc = probe_accuracy(Xs[L], y, seed=seed)
        mdl = online_codelength(Xs[L], y, seed=seed)
        rec = {"accuracy": acc, "mdl_trained": mdl["total_bits"],
               "uniform_bits": mdl["uniform_bits"], "compression": mdl["compression"]}
        if args.no_random_init:
            print(f"[B2] {args.model} L{L}: {args.label} acc real={acc['real']:.3f} "
                  f"ctrl={acc['control_task']:.3f} | compression(uniform/trained)="
                  f"{mdl['compression']:.2f}", flush=True)
        else:
            mdl_r = online_codelength(Xr[L], yr, seed=seed)
            rec["mdl_random"] = mdl_r["total_bits"]
            rec["mdl_ratio"] = mdl_r["total_bits"] / mdl["total_bits"]
            print(f"[B2] {args.model} L{L}: {args.label} acc real={acc['real']:.3f} "
                  f"ctrl={acc['control_task']:.3f} | MDL ratio (rand/trained)="
                  f"{rec['mdl_ratio']:.2f} | compression={mdl['compression']:.2f}",
                  flush=True)
        results["layers"][f"layer{L}"] = rec

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    suffix = "" if args.label == "depth" else f"_{args.label}"
    out = runs_dir / f"phaseB2_{args.model.replace('/', '_')}{suffix}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[B2] wrote {out}")


if __name__ == "__main__":
    main()

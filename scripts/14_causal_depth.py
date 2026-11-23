"""Causal test: does the model USE the (decodable) depth representation?

Decodability != use. We learn the norm-probe map B (depth ~ ||Bh||^2) at a layer,
take the orthonormal basis Q of its row space (the linearly-depth-encoding
subspace, in standardized activation space), and ABLATE that subspace from the
residual stream via a forward hook. We then measure the increase in clean LM loss
vs ablating a RANDOM subspace of equal rank. If depth-ablation hurts LM prediction
substantially more than random-ablation, the model causally uses the depth signal
for prediction (amnesic-probing logic, Elazar et al.).

Usage:
    python scripts/14_causal_depth.py --config config/pilot.yaml --model gpt2
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
                                         orthonormal_rowspace, project_out,
                                         random_rowspace, train_norm_probe)
from latent_structure.syntactic_corpus import parse_depth_sentences


def decoder_layers(model):
    """Return the list of decoder blocks (arch-agnostic)."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError("could not locate decoder layers")


def lm_loss(model, tok, texts, device, hook_module=None, hook_fn=None):
    import torch
    model.eval()
    handle = hook_module.register_forward_hook(hook_fn) if hook_fn else None
    tot, n = 0.0, 0
    try:
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
                tot += float(model(input_ids=ids, labels=ids).loss); n += 1
    finally:
        if handle:
            handle.remove()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-sentences", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=128)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sents = parse_depth_sentences(n_sentences=args.n_sentences,
                                  split=cfg["corpus"]["eval_split"], label="depth")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    d = model.config.hidden_size
    n_layer = getattr(model.config, "num_hidden_layers", None) or model.config.n_layer
    layer = round(n_layer * 0.5)                       # hidden_states[layer] = block[layer-1] out

    # split SENTENCES first (so we keep test texts for the LM-loss eval)
    cut = int(len(sents) * 0.7)
    train_sents, test_sents = sents[:cut], sents[cut:]
    tr = collect_sentences(model, tok, train_sents, layer, device)
    te = collect_sentences(model, tok, test_sents, layer, device)
    B, mu, sigma = train_norm_probe(tr, d, device, rank=args.rank, seed=seed)
    nspr = eval_norm_probe((B, mu, sigma), te, device)["norm_spearman"]
    Qd = orthonormal_rowspace(B)                       # (r, d)
    Qr = random_rowspace(args.rank, d, device, seed=seed)
    print(f"[causal] {args.model} layer {layer}: depth NSpr={nspr:.3f}, ablating rank {args.rank}/{d}",
          flush=True)

    block = decoder_layers(model)[layer - 1]

    def make_hook(Q):
        def hook(mod, inp, out):
            # the block returns a tuple or a tensor, depending on the library version
            h = out[0] if isinstance(out, tuple) else out
            h_new = project_out((h - mu) / sigma, Q) * sigma + mu
            return (h_new,) + tuple(out[1:]) if isinstance(out, tuple) else h_new
        return hook

    texts = [t for t, _ in test_sents]
    base = lm_loss(model, tok, texts, device)
    abl_depth = lm_loss(model, tok, texts, device, block, make_hook(Qd))
    abl_rand = lm_loss(model, tok, texts, device, block, make_hook(Qr))

    res = {"model": args.model, "layer": layer, "rank": args.rank, "depth_nspr": nspr,
           "lm_loss_baseline": base, "lm_loss_depth_ablated": abl_depth,
           "lm_loss_random_ablated": abl_rand,
           "delta_depth": abl_depth - base, "delta_random": abl_rand - base,
           "depth_over_random": (abl_depth - base) / (abl_rand - base + 1e-9)}
    print(f"[causal] baseline={base:.4f} depth_abl={abl_depth:.4f} (+{res['delta_depth']:.4f}) "
          f"rand_abl={abl_rand:.4f} (+{res['delta_random']:.4f}) | "
          f"depth/random={res['depth_over_random']:.2f}", flush=True)

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    out = runs_dir / f"causal_depth_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[causal] wrote {out}")


if __name__ == "__main__":
    main()

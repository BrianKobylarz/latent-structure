"""Does removing the subspace from which depth is read hurt prediction, and does
it hurt more than removing a comparison subspace of the same size?

A norm probe for dependency depth (rank 128) is fit at the middle layer. Its row
space is projected out of the residual stream (in standardized coordinates, at
every position), and the change in next-token loss on held-out sentences is
compared with two kinds of comparison subspace of the same rank:

  * the row space of the same probe fit to SHUFFLED depth labels (same
    procedure, same activations, a meaningless target);
  * random subspaces.

The loss is reported over all predicted tokens and by the parser depth of the
predicted token (shallow: 2 or less; middle: 3 or 4; deep: 5 or more). The depth
of a token is looked up at the LAST character of its offset, so that tokens
that begin with a space are covered.

Output: <storage_root>/runs/causal_depth_by_depth_<model>.json.

The files results/causal_depth_buckets_*.json were written in June 2026 by an
earlier version that looked the depth up at the first character of the offset
(a space for most tokens) and used one random subspace; they cover punctuation
and word-continuation tokens only.

Usage:
    python scripts/15_causal_depth_buckets.py --config config/pilot.yaml --model gpt2
    python scripts/15_causal_depth_buckets.py --config config/pilot.yaml --model Qwen/Qwen2.5-7B-Instruct --load-4bit
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
from latent_structure.structural import (collect_sentences, decoder_layers,
                                         eval_norm_probe, orthonormal_rowspace,
                                         project_out, random_rowspace, train_norm_probe)
from latent_structure.syntactic_corpus import parse_depth_sentences

BUCKETS = ("all", "shallow", "middle", "deep")


def bucketed_ce(model, tok, test_sents, block, hook_fn, device, shallow_max, deep_min):
    """Mean next-token cross-entropy over all predicted tokens and by the depth
    of the predicted token. Returns ({bucket: mean loss}, {bucket: token count})."""
    import torch
    import torch.nn.functional as F
    model.eval()
    handle = block.register_forward_hook(hook_fn) if hook_fn else None
    tot = {b: 0.0 for b in BUCKETS}
    cnt = {b: 0 for b in BUCKETS}
    try:
        with torch.no_grad():
            for text, spans in test_sents:
                char_depth = np.full(len(text), -1, dtype=np.int64)
                for cs, ce, dep in spans:
                    char_depth[cs:ce] = dep
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                if len(ids) < 3:
                    continue
                x = torch.tensor([ids], device=device)
                logits = model(input_ids=x).logits[0]            # (L, V)
                lp = F.log_softmax(logits[:-1].float(), dim=-1)
                tgt = torch.tensor(ids[1:], device=device)
                ce = (-lp[torch.arange(tgt.shape[0]), tgt]).cpu().numpy()  # ce[t] predicts ids[t+1]
                for t in range(len(ce)):
                    tot["all"] += ce[t]; cnt["all"] += 1
                    a, b = offs[t + 1]
                    if b <= a or b - 1 >= len(char_depth):
                        continue
                    dep = char_depth[b - 1]                      # last character of the token
                    if dep < 0:
                        continue
                    name = "shallow" if dep <= shallow_max else "deep" if dep >= deep_min else "middle"
                    tot[name] += ce[t]; cnt[name] += 1
    finally:
        if handle:
            handle.remove()
    return {b: tot[b] / max(1, cnt[b]) for b in BUCKETS}, cnt


def shuffled_labels(train, seed):
    """The training set with its depth labels permuted across all words."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([d for _, d in train])
    labels = labels[rng.permutation(len(labels))]
    out, i = [], 0
    for a, d in train:
        out.append((a, labels[i:i + len(d)]))
        i += len(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-sentences", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--shallow-max", type=int, default=2)
    ap.add_argument("--deep-min", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--n-shuffled", type=int, default=3)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(name):
        if args.load_4bit:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            return AutoModelForCausalLM.from_pretrained(name, quantization_config=qc, device_map="cuda")
        if device == "cuda":                             # load straight onto the card
            return AutoModelForCausalLM.from_pretrained(name, device_map="cuda")
        return AutoModelForCausalLM.from_pretrained(name)

    sents = parse_depth_sentences(n_sentences=args.n_sentences,
                                  split=cfg["corpus"]["eval_split"], label="depth")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load(args.model)
    d = model.config.hidden_size
    n_layer = getattr(model.config, "num_hidden_layers", None) or model.config.n_layer
    layer = round(n_layer * 0.5)

    cut = int(len(sents) * 0.7)
    train_sents, test_sents = sents[:cut], sents[cut:]
    tr = collect_sentences(model, tok, train_sents, layer, device)
    te = collect_sentences(model, tok, test_sents, layer, device)
    B, mu, sigma = train_norm_probe(tr, d, device, rank=args.rank, seed=seed)
    nspr = eval_norm_probe((B, mu, sigma), te, device)["norm_spearman"]
    block = decoder_layers(model)[layer - 1]
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"[removal] {args.model} ({dtype}, {device}) layer {layer}: depth norm Spearman {nspr:.3f}; "
          f"{len(test_sents)} held-out sentences", flush=True)

    def mk(Q):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h32 = h.float()
            new = (project_out((h32 - mu) / sigma, Q) * sigma + mu).to(h.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook

    def measure(Q):
        loss, _ = bucketed_ce(model, tok, test_sents, block, mk(Q), device,
                              args.shallow_max, args.deep_min)
        return loss

    base, counts = bucketed_ce(model, tok, test_sents, block, None, device,
                               args.shallow_max, args.deep_min)
    inc = lambda loss: {b: loss[b] - base[b] for b in BUCKETS}
    depth_inc = inc(measure(orthonormal_rowspace(B)))
    print(f"[removal] loss before {base['all']:.3f}; depth subspace removed {depth_inc['all']:+.3f} "
          f"(deep tokens {depth_inc['deep']:+.3f})", flush=True)

    shuf_runs, shuf_nspr = [], []
    for s in range(args.n_shuffled):
        Bs, _, _ = train_norm_probe(shuffled_labels(tr, 100 + s), d, device, rank=args.rank, seed=seed)
        shuf_nspr.append(eval_norm_probe((Bs, mu, sigma), te, device)["norm_spearman"])
        shuf_runs.append(inc(measure(orthonormal_rowspace(Bs))))
        print(f"[removal] shuffled-label probe {s}: norm Spearman on true depth {shuf_nspr[-1]:+.3f}; "
              f"removed {shuf_runs[-1]['all']:+.3f}", flush=True)
    rand_runs = []
    for s in range(args.n_random):
        rand_runs.append(inc(measure(random_rowspace(args.rank, d, device, seed=seed + s))))
        print(f"[removal] random subspace {s}: removed {rand_runs[-1]['all']:+.3f}", flush=True)

    def summary(runs):
        return {"mean": {b: float(np.mean([r[b] for r in runs])) for b in BUCKETS},
                "min": {b: float(np.min([r[b] for r in runs])) for b in BUCKETS},
                "max": {b: float(np.max([r[b] for r in runs])) for b in BUCKETS},
                "runs": runs} if runs else None

    res = {"model": args.model, "precision": "4bit" if args.load_4bit else dtype, "device": device,
           "layer": layer, "rank": args.rank, "hidden_size": d,
           "depth_nspr": nspr, "held_out_sentences": len(test_sents), "tokens": counts,
           "loss_before": base, "increase_depth_subspace": depth_inc,
           "shuffled_probe_nspr_on_true_depth": shuf_nspr,
           "increase_shuffled_probe_subspace": summary(shuf_runs),
           "increase_random_subspace": summary(rand_runs)}

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    out = runs_dir / f"causal_depth_by_depth_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n[removal] wrote {out}")


if __name__ == "__main__":
    main()

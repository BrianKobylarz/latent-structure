"""Track C: junk-beats-clean crossing trainer (a harness; the full run was not made).

Trains two models -- one on a CLEAN text pool, one on POOL + shuffled-word JUNK --
and tracks clean held-out validation loss over training. The paper's effect: with
enough tokens (~11B on a ~330M model) the pool+junk model's clean-val loss reaches
or undercuts the pool-only model (junk helps). After the crossing, the pool+junk
model is the one to probe (with the validated MDL/causal method) to characterize
what structure it extracted from the junk.

The defaults (config:track_c) target cloud GPUs: ~330M / ~670M-pool
/ many steps; `--smoke` overrides to a tiny fast LOCAL run that only validates the
pipeline end-to-end (it will NOT show the crossing -- far too small).

Usage:
    python scripts/12_track_c_crossing.py --config config/pilot.yaml --smoke   # local harness check
    python scripts/12_track_c_crossing.py --config config/pilot.yaml           # full run (cloud GPUs)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml
from transformers.utils.logging import disable_progress_bar, set_verbosity_error

disable_progress_bar(); set_verbosity_error()

from latent_structure import corpus as C


def tokenize_docs(docs, tok, max_tokens, seed=0):
    """Tokenize clean docs into one int32 stream, eos-separated (pool / val)."""
    eos = tok.eos_token_id
    ids = []
    for d in docs:
        ids.extend(tok.encode(d)); ids.append(eos)
        if len(ids) >= max_tokens:
            break
    return np.asarray(ids[:max_tokens], dtype=np.int32)


def build_junk(docs, tok, max_tokens, mode, seed=0):
    """Construct a junk token stream under one arm:
    'shuffle'  -- shuffle word order within each doc (per-doc multiset and thus
                  DOCUMENT-LEVEL CO-OCCURRENCE preserved; order destroyed). The
                  paper's helpful arm.
    'unigram'  -- words drawn i.i.d. from the GLOBAL unigram distribution over all
                  doc words (global unigram preserved; ALL co-occurrence DESTROYED).
                  The decisive control: if shuffle helps but this does not, the
                  benefit IS co-occurrence.
    """
    rng = random.Random(seed)
    eos = tok.eos_token_id
    ids = []
    if mode == "shuffle":
        for d in docs:
            w = d.split(); rng.shuffle(w)
            ids.extend(tok.encode(" ".join(w))); ids.append(eos)
            if len(ids) >= max_tokens:
                break
    elif mode == "unigram":
        words = []
        for d in docs:                     # global word bag (empirical unigram)
            words.extend(d.split())
            if len(words) >= 4_000_000:
                break
        words = np.asarray(words, dtype=object)
        npr = np.random.default_rng(seed)
        while len(ids) < max_tokens:        # i.i.d. global-unigram pseudo-docs
            chunk = npr.choice(words, size=200, replace=True)
            ids.extend(tok.encode(" ".join(map(str, chunk)))); ids.append(eos)
    else:
        raise ValueError(f"unknown junk mode {mode!r}")
    return np.asarray(ids[:max_tokens], dtype=np.int32)


def get_batch(tokens, batch_size, ctx, rng, device):
    import torch
    ix = rng.integers(0, len(tokens) - ctx - 1, size=batch_size)
    x = np.stack([tokens[i:i + ctx] for i in ix])
    return torch.from_numpy(x.astype(np.int64)).to(device)


def eval_loss(model, tokens, ctx, device, max_chunks=400):
    import torch
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for s in range(0, min(len(tokens) - 1, max_chunks * ctx), ctx):
            chunk = tokens[s:s + ctx]
            if len(chunk) < 2:
                continue
            x = torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(device)
            tot += float(model(input_ids=x, labels=x).loss); n += 1
    return tot / max(1, n)


def train_stream(name, train_tokens, val_tokens, cfg_tc, out_dir, device, seed, log):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from torch.optim.lr_scheduler import LambdaLR
    rng = np.random.default_rng(seed)
    ctx, steps = cfg_tc["ctx"], cfg_tc["steps"]
    config = GPT2Config(vocab_size=50257, n_positions=ctx, n_embd=cfg_tc["n_embd"],
                        n_layer=cfg_tc["n_layer"], n_head=cfg_tc["n_head"])
    model = GPT2LMHeadModel(config).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"[track_c:{name}] params={nparams/1e6:.0f}M tokens={len(train_tokens):,} steps={steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_tc["lr"], weight_decay=0.1, betas=(0.9, 0.95))
    sched = LambdaLR(opt, lambda s: min(1.0, s / 100) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    hist = {"name": name, "params_M": nparams / 1e6, "evals": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    running = 0.0
    model.train()
    for step in range(1, steps + 1):
        x = get_batch(train_tokens, cfg_tc["batch_size"], ctx, rng, device)
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
        else:
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); running += loss.item()
        if step % cfg_tc["eval_every"] == 0 or step == steps:
            vloss = eval_loss(model, val_tokens, ctx, device)
            toks_seen = step * cfg_tc["batch_size"] * ctx
            hist["evals"].append({"step": step, "tokens_seen": toks_seen,
                                  "train_loss": running / cfg_tc["eval_every"],
                                  "clean_val_loss": vloss})
            (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
            log(f"[track_c:{name}] step {step} tok {toks_seen/1e6:.1f}M "
                f"train {running/cfg_tc['eval_every']:.3f} clean_val {vloss:.3f}")
            running = 0.0
    model.save_pretrained(out_dir)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny fast local harness check")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tc = dict(cfg["track_c"]); seed = int(cfg.get("seed", 0))

    if args.smoke:
        tc.update(pool_tokens=2_000_000, n_embd=256, n_layer=4, n_head=4, ctx=256,
                  steps=400, batch_size=16, eval_every=100, eval_tokens=200_000)

    from transformers import AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    base = runs_dir / ("track_c_smoke" if args.smoke else "track_c")
    modes = tc.get("junk_modes", ["shuffle", "unigram"])

    print(f"[track_c] building pool ({tc['pool_tokens']/1e6:.0f}M tokens) + arms {modes} "
          f"@ +{tc['junk_ratio']*100:.0f}% ...", flush=True)
    train_docs = C.build_corpus(split="train", max_documents=None, min_doc_chars=200)
    val_docs = C.build_corpus(split="validation", max_documents=None, min_doc_chars=200)
    pool = tokenize_docs(train_docs, tok, tc["pool_tokens"], seed=seed)
    val = tokenize_docs(val_docs, tok, tc["eval_tokens"], seed=seed)
    print(f"[track_c] pool={len(pool):,} val={len(val):,}", flush=True)

    t0 = time.time()
    finals = {}
    h = train_stream("pool", pool, val, tc, base / "pool", device, seed, print)
    finals["pool"] = h["evals"][-1]["clean_val_loss"]

    junk_tok = int(tc["pool_tokens"] * tc["junk_ratio"])
    for mode in modes:
        junk = build_junk(train_docs, tok, junk_tok, mode, seed=seed + 1)
        stream = np.concatenate([pool, junk])
        print(f"[track_c] arm '{mode}': junk={len(junk):,} -> pool+{mode}={len(stream):,}",
              flush=True)
        h = train_stream(f"pool+{mode}", stream, val, tc, base / f"pool_{mode}",
                         device, seed, print)
        finals[f"pool+{mode}"] = h["evals"][-1]["clean_val_loss"]

    vp = finals["pool"]
    summary = {"final_clean_val": finals,
               "deltas_vs_pool": {k: finals[k] - vp for k in finals if k != "pool"},
               "wall_seconds": time.time() - t0, "smoke": args.smoke}
    (base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[track_c] final clean-val loss (lower = better):")
    for k, v in finals.items():
        d = "" if k == "pool" else f"  (delta vs pool {v - vp:+.3f})"
        print(f"  {k:16s} {v:.3f}{d}")
    print("[track_c] DECISIVE READ: if pool+shuffle beats pool but pool+unigram does NOT, "
          "the junk benefit is co-occurrence (alien falsified).")
    if args.smoke:
        print("[track_c] SMOKE run: harness validated only; the crossing needs the full "
              "330M / ~11B-token run.")


if __name__ == "__main__":
    main()

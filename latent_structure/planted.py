"""Train a small from-scratch Transformer on a synthetic planted-structure
corpus and measure BEHAVIORAL extraction of the latent rule.

Behavioral signal (per the lit review, NOT loss): at each marker position, does
the model assign higher probability to the rule-correct marker (governed by the
latent depth) than to the other? Reported as marker accuracy, compared against
the n-gram null (best over small k) and the true-depth oracle. Genuine
latent-state use shows up as accuracy ABOVE the n-gram null, growing over
training, and approaching the oracle. (Probe + MDL + causal intervention, which
rule out a bigram shortcut, live in probing.py / intervention.py.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import model as M
from .synthetic_corpus import Corpus, ngram_marker_accuracy, oracle_marker_accuracy


def _split(corpus: Corpus, test_frac=0.1):
    n = corpus.tokens.shape[0]
    cut = int(n * (1 - test_frac))
    return cut


def behavioral_marker_eval(model, corpus: Corpus, device, rows, batch=64):
    """Marker accuracy of the model on `rows` of `corpus`: at each marker
    position t, compare the model's predicted logits for A vs B (from position
    t-1) against the rule-correct label (governing depth >= thresh)."""
    import torch
    model.eval()
    cfg = corpus.config
    a_id, b_id, thresh = cfg.a_id, cfg.b_id, cfg.effective_thresh
    gov = corpus.governing_depth()           # (N, L), -1 off markers
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            idx = rows[i:i + batch]
            x = torch.from_numpy(corpus.tokens[idx].astype(np.int64)).to(device)
            logits = model(input_ids=x).logits           # (b, L, V)
            # prediction for token t comes from position t-1
            pred_b = (logits[:, :-1, b_id] > logits[:, :-1, a_id]).cpu().numpy()  # (b, L-1)
            g = gov[idx][:, 1:]                            # governing depth for token t
            mask = g >= 0                                  # marker positions (t>=1)
            true_b = g[mask] >= thresh
            correct += int((pred_b[mask] == true_b).sum())
            total += int(mask.sum())
    return correct / max(1, total)


def train_planted(corpus: Corpus, *, preset="tiny", steps=4000, batch_size=64,
                  lr=6e-4, warmup=100, weight_decay=0.1, grad_clip=1.0,
                  eval_every=500, out_dir=None, seed=0, log=print,
                  ngram_null=None, oracle=None):
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = corpus.config
    ctx = corpus.tokens.shape[1]
    rng = np.random.default_rng(seed)

    cut = _split(corpus)
    train_rows = np.arange(cut)
    test_rows = np.arange(cut, corpus.tokens.shape[0])

    # data-only baselines (independent of the model); may be passed in (e.g. so a
    # shuffle-control run is scored against the CLEAN null/oracle)
    ngram = ngram_null if ngram_null is not None else \
        max(ngram_marker_accuracy(corpus, k)["accuracy"] for k in (1, 2, 4, 8))
    oracle = oracle if oracle is not None else oracle_marker_accuracy(corpus)

    model = M.build_raw_model(preset, cfg.vocab_size, ctx=ctx).to(device)
    pc = M.param_counts(model)
    log(f"[planted rule_p={cfg.rule_p}] vocab={cfg.vocab_size} ctx={ctx} "
        f"params total={pc['total']:,} non_emb={pc['non_embedding']:,} "
        f"ngram_null={ngram:.3f} oracle={oracle:.3f} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = LambdaLR(opt, lr_lambda)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    history = {"rule_p": cfg.rule_p, "preset": preset, "params": pc,
               "ngram_null": ngram, "oracle": oracle, "evals": []}

    def run_eval(step, train_loss):
        acc = behavioral_marker_eval(model, corpus, device, test_rows)
        rec = {"step": step, "train_loss": train_loss, "marker_acc": acc,
               "margin_over_ngram": acc - ngram}
        history["evals"].append(rec)
        if out_dir:
            (Path(out_dir) / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
        log(f"[planted rule_p={cfg.rule_p}] step {step:5d} loss={train_loss:.3f} "
            f"marker_acc={acc:.3f} (null={ngram:.3f} oracle={oracle:.3f} "
            f"margin={acc-ngram:+.3f})")

    model.train()
    running = 0.0
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, cut, size=batch_size)
        x = torch.from_numpy(corpus.tokens[idx].astype(np.int64)).to(device)
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
        else:
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        running += loss.item()
        if step % eval_every == 0 or step == steps:
            run_eval(step, running / eval_every)
            running = 0.0

    history["wall_seconds"] = time.time() - t0
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history

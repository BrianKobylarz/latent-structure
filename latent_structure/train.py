"""Train one small GPT on one corpus variant, evaluating on CLEAN validation.

History (train loss + the position-decomposed clean-validation NLL) is written
to runs/<variant>_<size>/history.json after every eval, so a long run is always
trackable mid-flight. The final model + tokenizer are saved alongside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _load_variant_tokens(jsonl_path: Path, tokenizer, max_tokens: int) -> np.ndarray:
    """Tokenize a variant's documents into one int64 stream, eos-separated."""
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            ids.extend(tokenizer.encode(text))
            ids.append(eos)
            if len(ids) >= max_tokens:
                break
    return np.asarray(ids[:max_tokens], dtype=np.int64)


def _get_batch(tokens, batch_size, ctx, rng, device):
    import torch
    ix = rng.integers(0, len(tokens) - ctx - 1, size=batch_size)
    x = np.stack([tokens[i:i + ctx] for i in ix])
    y = np.stack([tokens[i + 1:i + 1 + ctx] for i in ix])
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


def train_one(cfg, variant, *, eval_set=None, tokenizer=None, log=print):
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    from . import corpus as C
    from . import model as M
    from .evaluate import CATEGORIES, build_eval_set, evaluate_model

    tcfg = cfg["training"]
    size = tcfg["size"]
    ctx = M.context_length(size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    data_dir, runs_dir = C.resolve_storage(cfg["storage_root"])
    train_dir = data_dir / cfg["corpus"]["name"] / cfg["corpus"]["train_split"]
    jsonl = train_dir / f"{variant}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"{jsonl} -- run 01_build_datasets.py first")

    if tokenizer is None:
        tokenizer = M.build_tokenizer()
    if eval_set is None:
        eval_set = build_eval_set(cfg, tokenizer, split=cfg["corpus"]["eval_split"],
                                  max_documents=tcfg["eval_max_documents"],
                                  window=tcfg["post_modal_window"])

    tokens = _load_variant_tokens(jsonl, tokenizer, tcfg["max_train_tokens"])
    model = M.build_model(size, tokenizer).to(device)
    pc = M.param_counts(model)
    log(f"[{variant}] tokens={len(tokens):,} params total={pc['total']:,} "
        f"non_emb={pc['non_embedding']:,} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"], betas=(0.9, 0.95))
    steps = tcfg["steps"]
    warmup = tcfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = LambdaLR(opt, lr_lambda)

    out_dir = runs_dir / f"{variant}_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {"variant": variant, "size": size, "params": pc, "evals": []}
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    def run_eval(step, train_loss):
        res = evaluate_model(model, eval_set, device, ctx)
        rec = {"step": step, "train_loss": train_loss,
               **{c: res[c] for c in CATEGORIES}}
        history["evals"].append(rec)
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        g, cm, pm = res["global"]["nll"], res["core_modal"]["nll"], res["post_modal"]["nll"]
        log(f"[{variant}] step {step:5d} train={train_loss:.3f} "
            f"clean global={g:.3f} core_modal={cm:.3f} post_modal={pm:.3f}")

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        x, y = _get_batch(tokens, tcfg["batch_size"], ctx, rng, device)
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=y).loss
        else:
            loss = model(input_ids=x, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        sched.step()
        running += loss.item()
        if step % tcfg["eval_every"] == 0 or step == steps:
            run_eval(step, running / tcfg["eval_every"])
            running = 0.0

    history["wall_seconds"] = time.time() - t0
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log(f"[{variant}] done in {history['wall_seconds']:.1f}s -> {out_dir}")
    return history

"""Position-decomposed NLL evaluation on CLEAN held-out text.

This is the core instrument. A model trained on some perturbed TRAIN variant is
always evaluated on the CLEAN validation split -- never on the distribution it
was trained on. We report the mean next-token NLL not just globally but broken
down by what kind of token is being predicted:

    global         every predicted token
    core_modal     predicted token is part of an epistemic-deontic modal span
    pred_modal     predicted token is part of a PREDICTION modal (will/would)
    post_modal     predicted token falls within W tokens AFTER a modal span
                   (where modal force propagates -- e.g. the proposition under
                   "it must be <here>")
    control        predicted token is a frequency-matched non-modal control word

The hypothesis structure (H1-H3) lives in the CONTRASTS between these buckets
across models trained on different variants -- e.g. polarity-corrupted training
should hurt core_modal / post_modal NLL more than control NLL.

Span -> token mapping uses the fast tokenizer's offset mapping, so spaCy
character spans are aligned exactly onto BPE token positions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .modals import detect_spans, is_core
from .perturb import _control_pattern

CATEGORIES = ("global", "core_modal", "pred_modal", "post_modal", "control")


def _spans_to_token_mask(offsets, char_spans) -> np.ndarray:
    """Boolean mask over tokens: True where a token overlaps any char span."""
    mask = np.zeros(len(offsets), dtype=bool)
    if not char_spans:
        return mask
    for j, (os, oe) in enumerate(offsets):
        if os == oe:           # special / zero-width token
            continue
        for cs, ce in char_spans:
            if not (oe <= cs or os >= ce):
                mask[j] = True
                break
    return mask


def label_document(text, tokenizer, control_pattern, window: int):
    """Tokenize one document and return (input_ids, masks) where masks is a
    dict of per-token boolean arrays for each category (except 'global')."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n = len(ids)

    spans = detect_spans(text)
    core_spans = [(s.char_start, s.char_end) for s in spans if is_core(s.lemma)]
    pred_spans = [(s.char_start, s.char_end) for s in spans if not is_core(s.lemma)]
    ctrl_spans = [(m.start(), m.end()) for m in control_pattern.finditer(text)] \
        if control_pattern is not None else []

    core = _spans_to_token_mask(offsets, core_spans)
    pred = _spans_to_token_mask(offsets, pred_spans)
    control = _spans_to_token_mask(offsets, ctrl_spans)

    # post-modal window: tokens after any modal token, up to `window` ahead,
    # excluding modal tokens themselves.
    modal_any = core | pred
    post = np.zeros(n, dtype=bool)
    modal_idx = np.flatnonzero(modal_any)
    for j in modal_idx:
        post[j + 1: j + 1 + window] = True
    post &= ~modal_any

    return np.asarray(ids, dtype=np.int64), {
        "core_modal": core, "pred_modal": pred,
        "post_modal": post, "control": control,
    }


def build_eval_set(cfg, tokenizer, split="validation", max_documents=None,
                   window=5, cache=True):
    """Build (and cache) the labeled clean evaluation set. Returns a list of
    {ids, masks} per document. Cached under runs/eval_cache keyed by split +
    doc cap + window so every model reuses the identical labeled data."""
    from . import corpus as C
    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    cache_dir = runs_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{cfg['corpus']['name']}_{split}_n{max_documents}_w{window}"
    cache_path = cache_dir / f"{key}.npz"

    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return list(z["docs"])

    # control vocabulary from the TRAIN manifest (so eval labels match the
    # control tokens the control_mask variant actually perturbed)
    data_dir, _ = C.resolve_storage(cfg["storage_root"])
    manifest_path = data_dir / cfg["corpus"]["name"] / cfg["corpus"]["train_split"] / "manifest.json"
    control_vocab = set()
    if manifest_path.exists():
        control_vocab = set(json.loads(manifest_path.read_text(encoding="utf-8"))["control_vocab"])
    pattern = _control_pattern(frozenset(control_vocab)) if control_vocab else None

    docs = C.build_corpus(split=split, max_documents=max_documents,
                          min_doc_chars=cfg["corpus"].get("min_doc_chars", 500))
    out = []
    for text in docs:
        ids, masks = label_document(text, tokenizer, pattern, window)
        out.append({"ids": ids, "masks": masks})
    if cache:
        np.savez(cache_path, docs=np.array(out, dtype=object))
    return out


def evaluate_model(model, eval_set, device, ctx, batch_docs=8):
    """Compute mean NLL per category over the eval set. Processes each document
    in non-overlapping ctx-length chunks so label alignment is exact. Loss at
    position i is attributed to the category of the predicted token (i+1)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    sums = {c: 0.0 for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}

    with torch.no_grad():
        for doc in eval_set:
            ids = doc["ids"]
            masks = doc["masks"]
            for start in range(0, len(ids), ctx):
                chunk = ids[start:start + ctx]
                if len(chunk) < 2:
                    continue
                x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                logits = model(x).logits[0]            # (L, V)
                # per-position NLL predicting the next token
                lp = F.log_softmax(logits[:-1].float(), dim=-1)
                tgt = x[0, 1:]
                nll = -lp[torch.arange(tgt.shape[0]), tgt]  # (L-1,)
                nll_np = nll.cpu().numpy()

                sums["global"] += float(nll_np.sum())
                counts["global"] += nll_np.shape[0]
                # target position i predicts token at absolute index start+i+1
                for cat in ("core_modal", "pred_modal", "post_modal", "control"):
                    m = masks[cat][start + 1: start + len(chunk)]
                    if m.any():
                        sums[cat] += float(nll_np[m].sum())
                        counts[cat] += int(m.sum())

    result = {}
    for c in CATEGORIES:
        mean = sums[c] / counts[c] if counts[c] else float("nan")
        result[c] = {"nll": mean, "ppl": float(np.exp(mean)) if counts[c] else float("nan"),
                     "tokens": counts[c]}
    return result

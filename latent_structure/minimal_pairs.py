"""Modal minimal-pair evaluation (Phase 4).

Per the literature review, no existing modal benchmark is scorable on a small
from-scratch LM (they all need instruction-following), and perplexity is not
comparable across models trained on different-entropy corpora. So we BUILD a
custom modal minimal-pair set and score by minimal-pair ACCURACY.

Construction (naturalistic, BLiMP-style): harvest CLEAN held-out sentences that
contain exactly one modal, then form contrast variants by applying our own
perturbations to that modal:

    cross    : swap modal force (POSSIBILITY/WEAK_NECESSITY/NECESSITY) -- corrupts
               inferential force while keeping grammaticality
    polarity : flip polarity (can -> cannot) -- teaches the wrong conditional
    within   : swap within force class -- near-control; even a good model need
               not strongly prefer the original (low expected ceiling)

A model "passes" an item if it assigns a higher length-normalized log-probability
to the ORIGINAL than to the contrast. Reported as accuracy per contrast type.
The CLEAN-trained model's accuracy is the validity/ceiling check: if it cannot
distinguish original from contrast, the items (or the metric) are uninformative.

Scoring is PREFIX-CONDITIONED: the original and contrast share a token prefix up
to the point they diverge (the modal region), so we score the mean log-prob of
only the divergent suffix conditioned on that shared prefix. This isolates the
modal change and removes the sentence-length artifact that made naive
full-sentence scoring fail on the polarity contrast (which inserts "not").
"""

from __future__ import annotations

import json
from pathlib import Path

from . import perturb as P
from .modals import detect_spans, get_nlp

CONTRASTS = {
    "cross": P.MODAL_CROSS_SWAP,
    "polarity": P.MODAL_POLARITY,
    "within": P.MODAL_WITHIN_SWAP,
}


def harvest_single_modal_sentences(docs, max_sentences, min_tokens=6,
                                   max_tokens=45):
    """Yield (sentence_text, span) for sentences containing exactly one detected
    modal, within a reasonable length range."""
    nlp = get_nlp()
    out = []
    for doc in nlp.pipe(docs, batch_size=64):
        for sent in doc.sents:
            n = len(sent)
            if n < min_tokens or n > max_tokens:
                continue
            spans = detect_spans(sent.text)
            if len(spans) != 1:
                continue
            out.append((sent.text.strip(), spans[0]))
            if len(out) >= max_sentences:
                return out
    return out


def build_minimal_pairs(cfg, *, split=None, max_sentences=4000,
                        max_items_per_contrast=1000, seed=0, cache=True):
    """Build (and cache) the modal minimal-pair set from CLEAN held-out text.
    Returns a list of items: {original, contrast, contrast_type, lemma, force,
    target_force}."""
    from . import corpus as C
    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    cache_dir = runs_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split = split or cfg["corpus"]["eval_split"]
    cache_path = cache_dir / f"minimal_pairs_{split}_n{max_items_per_contrast}.json"
    if cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    docs = C.build_corpus(split=split,
                          max_documents=cfg["corpus"].get("max_documents"),
                          min_doc_chars=cfg["corpus"].get("min_doc_chars", 500))
    sents = harvest_single_modal_sentences(docs, max_sentences)

    items = []
    counts = {k: 0 for k in CONTRASTS}
    for text, span in sents:
        for ctype, intervention in CONTRASTS.items():
            if counts[ctype] >= max_items_per_contrast:
                continue
            pd = P.perturb_document(text, intervention, seed=seed, spans=[span])
            if pd.n_applied != 1 or pd.text == text:
                continue
            edit = pd.edits[0]
            items.append({
                "original": text,
                "contrast": pd.text,
                "contrast_type": ctype,
                "lemma": span.lemma,
                "force": span.force,
                "target_force": edit.target_force,
            })
            counts[ctype] += 1
    if cache:
        cache_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items


def _token_logprobs(model, tokenizer, text, device):
    """Return (ids, lp) where lp[i] = log P(token_i | token_<i) for i>=1
    (lp[0] is 0.0, unused)."""
    import torch
    import torch.nn.functional as F
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        return None, None
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]          # (L, V)
    logp = F.log_softmax(logits[:-1].float(), dim=-1)     # predicts tokens 1..L-1
    tgt = ids[0, 1:]
    chosen = logp[torch.arange(tgt.shape[0]), tgt]        # (L-1,)
    lp = [0.0] + chosen.tolist()
    return ids[0].tolist(), lp


def _common_prefix_len(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


def _prefix_conditioned_scores(model, tokenizer, original, contrast, device):
    """Mean log-prob of the divergent suffix (from the first differing token to
    the end), conditioned on the shared prefix, for each sentence."""
    ids_o, lp_o = _token_logprobs(model, tokenizer, original, device)
    ids_c, lp_c = _token_logprobs(model, tokenizer, contrast, device)
    if ids_o is None or ids_c is None:
        return None, None
    k = _common_prefix_len(ids_o, ids_c)
    start = max(k, 1)            # need a predecessor token to condition on
    suf_o, suf_c = lp_o[start:], lp_c[start:]
    if not suf_o or not suf_c:
        return None, None
    return sum(suf_o) / len(suf_o), sum(suf_c) / len(suf_c)


def score_model_on_pairs(model, tokenizer, items, device):
    """Return per-contrast and per-force accuracy: fraction of items where the
    model prefers the original to the contrast (prefix-conditioned scoring)."""
    model.eval()
    by_contrast: dict[str, list[bool]] = {}
    by_axis: dict[str, list[bool]] = {}
    for it in items:
        so, sc = _prefix_conditioned_scores(
            model, tokenizer, it["original"], it["contrast"], device)
        if so is None or sc is None:
            continue
        correct = so > sc
        by_contrast.setdefault(it["contrast_type"], []).append(correct)
        if it["contrast_type"] == "cross":
            axis = f"{it['force']}->{it['target_force']}"
            by_axis.setdefault(axis, []).append(correct)

    def acc(lst):
        return {"accuracy": sum(lst) / len(lst), "n": len(lst)} if lst else \
            {"accuracy": float("nan"), "n": 0}

    return {
        "by_contrast": {k: acc(v) for k, v in sorted(by_contrast.items())},
        "by_cross_axis": {k: acc(v) for k, v in sorted(by_axis.items())},
    }

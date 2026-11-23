"""Phase B: a GENUINELY natural non-salient regularity -- syntactic dependency
depth -- as the latent to probe for.

This is the real-language analog of the planted bracket counter: every token in
natural text has a dependency depth (distance to the parse-tree root), which is a
real structural property, plausibly non-salient to human readers (we do not
consciously track parse depth), and -- crucially -- LABELABLE via a parser, so we
keep the rigorous probe + MDL + control methodology from the planted phase.

We build word-level token sequences (reduced vocab, so one token == one word ==
one depth label, avoiding subword-alignment complications), pack them into fixed
length with a SEP token between sentences, and store per-token depth as the probe
target. `depth_after[i]` is the dependency depth of token i (its position-i hidden
state should encode it); SEP/padding positions get -1 and are excluded from probing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntacticCorpus:
    tokens: np.ndarray        # (N, L) int16 word-token ids
    depth_after: np.ndarray   # (N, L) int16 dependency depth (-1 at SEP/excluded)
    vocab_size: int
    seq_len: int

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, tokens=self.tokens, depth_after=self.depth_after,
                 vocab_size=self.vocab_size, seq_len=self.seq_len)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        return SyntacticCorpus(z["tokens"], z["depth_after"],
                               int(z["vocab_size"]), int(z["seq_len"]))


def parse_depth_sentences(n_sentences=3000, depth_clip=12, split="validation",
                          max_documents=400, min_len=6, max_len=80, label="depth"):
    """Return [(sentence_text, [(char_start, char_end, int_label), ...]), ...] with
    a per-spaCy-token structural label + char spans, for aligning to ANY tokenizer
    (e.g. a pretrained model's BPE) via offset overlap. `label` selects the
    structural variable to probe:
        "depth"     dependency-tree depth (distance to root), clipped at depth_clip
        "head_dist" linear distance |i - head.i|, clipped at 20
        "dep"       dependency-relation label, mapped to a contiguous int id
    """
    import spacy
    from . import corpus as C
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    docs = C.build_corpus(split=split, max_documents=max_documents, min_doc_chars=200)

    dep_ids: dict[str, int] = {}

    def lab(t):
        if label == "depth":
            return min(_token_depth(t), depth_clip)
        if label == "head_dist":
            return min(abs(t.i - t.head.i), 20)
        if label == "dep":
            return dep_ids.setdefault(t.dep_, len(dep_ids))
        raise ValueError(f"unknown label {label!r}")

    out = []
    for doc in nlp.pipe(docs, batch_size=32):
        for sent in doc.sents:
            toks = [t for t in sent if not t.is_space]
            if not (min_len <= len(toks) <= max_len):
                continue
            base = sent.start_char
            spans = [(t.idx - base, t.idx - base + len(t), lab(t)) for t in toks]
            out.append((sent.text, spans))
            if len(out) >= n_sentences:
                return out
    return out


def _token_depth(tok, cap=60):
    d = 0
    while tok.head != tok and d < cap:
        tok = tok.head
        d += 1
    return d


def build_syntactic_corpus(n_sentences=30000, vocab_k=512, seq_len=256,
                           depth_clip=12, split="train", max_documents=2000,
                           min_len=4, max_len=80):
    """Parse WikiText, extract per-token dependency depth, and pack into fixed-
    length sequences over a top-(vocab_k-2) word vocabulary (0=<unk>, last=SEP)."""
    import re  # noqa
    from collections import Counter

    import spacy

    from . import corpus as C
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    docs = C.build_corpus(split=split, max_documents=max_documents, min_doc_chars=200)

    sents = []
    for doc in nlp.pipe(docs, batch_size=32):
        for sent in doc.sents:
            ws, ds = [], []
            for t in sent:
                if t.is_space:
                    continue
                ws.append(t.lower_)
                ds.append(min(_token_depth(t), depth_clip))
            if min_len <= len(ws) <= max_len:
                sents.append((ws, ds))
        if len(sents) >= n_sentences:
            break
    sents = sents[:n_sentences]

    freq = Counter(w for ws, _ in sents for w in ws)
    vocab = {w: i + 1 for i, (w, _) in enumerate(freq.most_common(vocab_k - 2))}
    sep = vocab_k - 1

    toks, deps = [], []
    for ws, ds in sents:
        for w, d in zip(ws, ds):
            toks.append(vocab.get(w, 0))
            deps.append(int(d))
        toks.append(sep)
        deps.append(-1)
    toks = np.asarray(toks, dtype=np.int16)
    deps = np.asarray(deps, dtype=np.int16)
    nseq = len(toks) // seq_len
    tokens = toks[:nseq * seq_len].reshape(nseq, seq_len)
    depth = deps[:nseq * seq_len].reshape(nseq, seq_len)
    return SyntacticCorpus(tokens, depth, vocab_k, seq_len), len(sents)

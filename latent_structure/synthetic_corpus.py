"""Synthetic corpus with a planted, non-salient latent structure.

A hidden bounded-depth counter governs a non-salient binary surface choice.

Token id layout (small vocab):
    0 .. carrier_vocab-1   ordinary "carrier" tokens (n-gram distractor surface)
    OPEN  = carrier_vocab          increments latent depth (bounded at d_max)
    CLOSE = carrier_vocab + 1       decrements latent depth (floored at 0)
    A     = carrier_vocab + 2        marker emitted when governing depth <  thresh
    B     = carrier_vocab + 3        marker emitted when governing depth >= thresh
    vocab_size = carrier_vocab + 4

The marker uses a MID-RANGE THRESHOLD on the bounded counter (default
thresh = d_max // 2), not a zero-test. This is deliberate: with a zero-test and a
low-ranging counter, the single nearest bracket token leaks the answer to a 2-gram.
A mid-range threshold means knowing the last +/-1 step tells you nothing about
which side of the threshold the ABSOLUTE depth is on, so the model must integrate
the full bracket history -- making the structure genuinely non-local. (Verified
in A0: the n-gram null sits at the class base rate.)

The latent depth is never written. OPEN/CLOSE/A/B are just tokens the model must
make sense of; their roles are non-salient to a human reading the stream.

Depth convention: `depth_after[t]` is the latent depth AFTER emitting token t.
A marker at position t is governed by the depth that exists when it appears,
i.e. `depth_after[t-1]` (the "governing depth"). To predict that marker, a model
reads tokens 0..t-1, whose position-(t-1) hidden state should encode
`depth_after[t-1]`. So the probe target at position i is `depth_after[i]`, and the
marker at i+1 is governed by it. This alignment is what `probing.py` relies on.

The marker follows the rule with probability `rule_p`; otherwise it is a random
A/B (this is the subtlety knob). `rule_p = 1.0` is fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class GenConfig:
    carrier_vocab: int = 64
    d_max: int = 8
    p_marker: float = 0.15      # P(a position is a marker slot)
    p_bracket: float = 0.30     # P(a non-marker position is a bracket op)
    rule_p: float = 1.0         # marker reliability (the frontier knob)
    thresh: int = -1            # marker threshold; -1 => max(1, d_max // 2)
    needle_mode: bool = False   # A6: emit OPEN/CLOSE 50/50 (saturating), so the
                                # counter is NOT incentivized by token prediction --
                                # only the marker rule can induce its representation
    carrier_mode: str = "synthetic"  # "synthetic" | "natural" (real WikiText carrier)
    natural_vocab_k: int = 512  # carrier vocab size when carrier_mode == "natural"
    seq_len: int = 256
    n_seqs: int = 20000
    seed: int = 0

    @property
    def effective_thresh(self) -> int:
        return self.thresh if self.thresh >= 0 else max(1, self.d_max // 2)

    @property
    def open_id(self) -> int:
        return self.carrier_vocab

    @property
    def close_id(self) -> int:
        return self.carrier_vocab + 1

    @property
    def a_id(self) -> int:
        return self.carrier_vocab + 2

    @property
    def b_id(self) -> int:
        return self.carrier_vocab + 3

    @property
    def vocab_size(self) -> int:
        return self.carrier_vocab + 4


@dataclass
class Corpus:
    tokens: np.ndarray        # (N, L) int16 token ids
    depth_after: np.ndarray   # (N, L) int16 latent depth after each token
    is_marker: np.ndarray     # (N, L) bool
    followed_rule: np.ndarray # (N, L) bool (marker positions only; else False)
    config: GenConfig

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, tokens=self.tokens, depth_after=self.depth_after,
                 is_marker=self.is_marker, followed_rule=self.followed_rule,
                 config=np.array([str(asdict(self.config))], dtype=object))

    @staticmethod
    def load(path) -> "Corpus":
        import ast
        z = np.load(path, allow_pickle=True)
        cfg = GenConfig(**ast.literal_eval(str(z["config"][0])))
        return Corpus(z["tokens"], z["depth_after"], z["is_marker"],
                      z["followed_rule"], cfg)

    def governing_depth(self) -> np.ndarray:
        """Per-position governing depth for marker slots (depth_after shifted by
        one), -1 at non-marker / sequence-start positions."""
        gov = np.full_like(self.depth_after, -1)
        gov[:, 1:] = self.depth_after[:, :-1]
        gov[~self.is_marker] = -1
        return gov


def build_natural_stream(carrier_vocab_k: int, split: str = "train",
                         max_documents: int = 2000,
                         max_tokens: int = 5_000_000) -> np.ndarray:
    """A long stream of real WikiText word-token ids for the natural carrier.
    Top (carrier_vocab_k - 1) words by frequency map to ids 1..k-1; everything
    else maps to 0 (<unk>). Real contiguous order is preserved, so the carrier
    carries genuine n-gram / co-occurrence structure (the interference threat)."""
    import re
    from collections import Counter
    from . import corpus as C
    word = re.compile(r"[a-z]+(?:'[a-z]+)?")
    docs = C.build_corpus(split=split, max_documents=max_documents, min_doc_chars=200)
    per_doc, freq = [], Counter()
    for d in docs:
        ws = word.findall(d.lower())
        per_doc.append(ws)
        freq.update(ws)
    vocab = {w: i + 1 for i, (w, _) in enumerate(freq.most_common(carrier_vocab_k - 1))}
    stream = []
    for ws in per_doc:
        for w in ws:
            stream.append(vocab.get(w, 0))
        if len(stream) >= max_tokens:
            break
    return np.asarray(stream[:max_tokens], dtype=np.int32)


def generate(cfg: GenConfig, natural_stream: np.ndarray | None = None) -> Corpus:
    rng = np.random.default_rng(cfg.seed)
    N, L = cfg.n_seqs, cfg.seq_len
    use_nat = cfg.carrier_mode == "natural"
    if use_nat:
        if natural_stream is None:
            raise ValueError("carrier_mode='natural' requires a natural_stream")
        S = len(natural_stream)
        starts = rng.integers(0, S, size=N)
    tokens = np.empty((N, L), dtype=np.int16)
    depth_after = np.empty((N, L), dtype=np.int16)
    is_marker = np.zeros((N, L), dtype=bool)
    followed_rule = np.zeros((N, L), dtype=bool)

    # pre-draw independent uniform randoms for speed; logic is sequential in depth
    R = rng.random((N, L))          # position type: marker / bracket / carrier
    Rc = rng.random((N, L))         # open-vs-close coin (independent, 50/50)
    Rr = rng.random((N, L))         # rule-following coin at markers
    carrier = rng.integers(0, cfg.carrier_vocab, size=(N, L))
    ab = rng.integers(0, 2, size=(N, L))  # random A/B when rule not followed
    thresh = cfg.effective_thresh

    for n in range(N):
        d = 0
        ptr = int(starts[n]) if use_nat else 0
        for t in range(L):
            if R[n, t] < cfg.p_marker:
                # marker slot, governed by current depth d (= depth before marker)
                correct_b = d >= thresh
                if Rr[n, t] < cfg.rule_p:
                    tok = cfg.b_id if correct_b else cfg.a_id
                    followed_rule[n, t] = True
                else:
                    tok = cfg.b_id if ab[n, t] else cfg.a_id
                    followed_rule[n, t] = (bool(ab[n, t]) == correct_b)
                is_marker[n, t] = True
                # depth unchanged by a marker
            elif R[n, t] < cfg.p_marker + cfg.p_bracket:
                if cfg.needle_mode:
                    # state-INDEPENDENT 50/50 emission, saturating at the bounds:
                    # predicting OPEN/CLOSE never needs the count, so the only
                    # incentive to track depth is the (faint) marker rule.
                    if Rc[n, t] < 0.5:
                        tok = cfg.open_id; d = min(d + 1, cfg.d_max)
                    else:
                        tok = cfg.close_id; d = max(d - 1, 0)
                else:
                    can_open = d < cfg.d_max
                    can_close = d > 0
                    if can_open and can_close:
                        if Rc[n, t] < 0.5:
                            tok = cfg.open_id; d += 1
                        else:
                            tok = cfg.close_id; d -= 1
                    elif can_open:
                        tok = cfg.open_id; d += 1
                    elif can_close:
                        tok = cfg.close_id; d -= 1
                    else:
                        if use_nat:
                            tok = int(natural_stream[ptr % S]); ptr += 1
                        else:
                            tok = int(carrier[n, t])
            else:
                if use_nat:
                    tok = int(natural_stream[ptr % S]); ptr += 1
                else:
                    tok = int(carrier[n, t])
            tokens[n, t] = tok
            depth_after[n, t] = d

    return Corpus(tokens, depth_after, is_marker, followed_rule, cfg)


# ---------------------------------------------------------------------------
# n-gram null: best prediction of marker A/B from the preceding k tokens.
# ---------------------------------------------------------------------------

def ngram_marker_accuracy(corpus: Corpus, k: int, split: float = 0.5) -> dict:
    """Estimate the accuracy of predicting a marker's A/B label from the
    preceding `k` tokens (the n-gram null). Counts (context -> A/B) on the first
    `split` of sequences, evaluates on the rest. Returns accuracy and the
    majority base rate. If this is near chance at the chosen range, the structure
    is genuinely non-local and the null is well specified."""
    tokens, is_marker = corpus.tokens, corpus.is_marker
    a_id = corpus.config.a_id
    n = tokens.shape[0]
    cut = int(n * split)

    def contexts(rows):
        from collections import defaultdict
        counts = defaultdict(lambda: [0, 0])  # ctx -> [#A, #B]
        for r in rows:
            mpos = np.flatnonzero(is_marker[r])
            for t in mpos:
                if t < k:
                    continue
                ctx = tuple(int(x) for x in tokens[r, t - k:t])
                label = 0 if tokens[r, t] == a_id else 1
                counts[ctx][label] += 1
        return counts

    train_counts = contexts(range(cut))
    pred = {ctx: (0 if c[0] >= c[1] else 1) for ctx, c in train_counts.items()}
    # global majority fallback for unseen contexts
    tot_a = sum(c[0] for c in train_counts.values())
    tot_b = sum(c[1] for c in train_counts.values())
    fallback = 0 if tot_a >= tot_b else 1
    base_rate = max(tot_a, tot_b) / max(1, tot_a + tot_b)

    correct = total = 0
    for r in range(cut, n):
        mpos = np.flatnonzero(is_marker[r])
        for t in mpos:
            if t < k:
                continue
            ctx = tuple(int(x) for x in tokens[r, t - k:t])
            label = 0 if tokens[r, t] == a_id else 1
            guess = pred.get(ctx, fallback)
            correct += (guess == label)
            total += 1
    return {"k": k, "accuracy": correct / max(1, total),
            "base_rate": base_rate, "n_eval": total}


def oracle_marker_accuracy(corpus: Corpus) -> float:
    """Accuracy of the rule-following (Bayes-with-true-depth) predictor: the
    fraction of markers whose label equals the governing-depth rule. Equals
    rule_p in expectation -- the behavioral ceiling that uses the latent state."""
    gov = corpus.governing_depth()
    m = corpus.is_marker
    labels_b = (corpus.tokens == corpus.config.b_id)
    rule_says_b = gov >= corpus.config.effective_thresh
    agree = (labels_b[m] == rule_says_b[m])
    return float(agree.mean())

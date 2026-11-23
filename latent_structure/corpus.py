"""Corpus loading, segmentation, frequency statistics, and control-vocabulary
construction.

The pilot corpus is WikiText-103 (raw). It is downloaded through the
`datasets` library into the usual Hugging Face cache.

Key responsibility beyond plain loading: build a FREQUENCY-MATCHED CONTROL
vocabulary. The modal-masking experiment perturbs high-frequency function
words, so any loss change is confounded with "we perturbed frequent tokens".
`build_control_vocab` selects a set of NON-modal word types whose per-type
corpus counts match the modal lemmas' counts as closely as possible, giving a
control intervention that perturbs the same amount of equally-frequent text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .modals import iter_modal_lemmas

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
# WikiText level-1 article header, e.g. " = Valkyria Chronicles III = "
_HEADER_RE = re.compile(r"^ = [^=].* = $")


def load_wikitext_rows(split: str = "train"):
    """Return the raw text rows of WikiText-103 for ``split``. Requires the
    `datasets` library; the download is cached by Hugging Face."""
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    except Exception:       # newer hubs serve it under the publisher's namespace only
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    return ds["text"]


def segment_articles(rows) -> list[str]:
    """Group WikiText line rows into article documents. An article begins at a
    level-1 header line and runs until the next one. Returns the list of
    article texts (header line included)."""
    articles: list[list[str]] = []
    current: list[str] = []
    for line in rows:
        if _HEADER_RE.match(line):
            if current:
                articles.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        articles.append(current)
    return ["".join(lines).strip() for lines in articles]


def build_corpus(split: str = "train", max_documents: int | None = None,
                 min_doc_chars: int = 500) -> list[str]:
    """Load and segment the corpus into documents, dropping very short stubs
    and capping the count for a tractable pilot."""
    rows = load_wikitext_rows(split)
    docs = [d for d in segment_articles(rows) if len(d) >= min_doc_chars]
    if max_documents is not None:
        docs = docs[:max_documents]
    return docs


def word_frequencies(docs) -> Counter:
    """Lowercased alphabetic word-type frequencies over the corpus. Used for
    building the control and random vocabularies (not for model tokenization)."""
    freq: Counter = Counter()
    for d in docs:
        freq.update(_WORD_RE.findall(d.lower()))
    return freq


def modal_lemma_counts(spans_per_doc) -> Counter:
    """Tally detected modal occurrences by lemma across the corpus, from the
    cached per-document detection spans."""
    counts: Counter = Counter()
    for spans in spans_per_doc:
        for sp in spans:
            counts[sp.lemma] += 1
    return counts


def build_control_vocab(modal_counts: Counter, freq: Counter) -> dict[str, int]:
    """Pick NON-modal single-word types whose corpus counts match the modal
    lemmas' counts. For each single-word modal lemma (in descending count
    order), greedily claim the unused non-modal word type with the closest
    count. Returns {control_word: count} for audit.

    Multi-word modal lemmas ("have to") are skipped here: control masking
    operates token-wise and cannot match a multi-word target one-to-one."""
    modal_words = set(iter_modal_lemmas())
    # candidate non-modal types, sorted by count for a stable nearest search
    candidates = sorted(
        ((w, c) for w, c in freq.items() if w not in modal_words),
        key=lambda wc: wc[1],
    )
    cand_words = [w for w, _ in candidates]
    cand_counts = [c for _, c in candidates]
    used: set[str] = set()
    control: dict[str, int] = {}

    import bisect
    targets = sorted(
        ((c, lemma) for lemma, c in modal_counts.items() if " " not in lemma),
        reverse=True,
    )
    for target_count, _lemma in targets:
        # find nearest unused candidate by count via binary search around target
        i = bisect.bisect_left(cand_counts, target_count)
        best = None
        best_d = None
        for j in range(max(0, i - 50), min(len(cand_words), i + 50)):
            w = cand_words[j]
            if w in used:
                continue
            d = abs(cand_counts[j] - target_count)
            if best_d is None or d < best_d:
                best, best_d = w, d
        if best is None:  # widen the search if the local window was exhausted
            for j, w in enumerate(cand_words):
                if w in used:
                    continue
                d = abs(cand_counts[j] - target_count)
                if best_d is None or d < best_d:
                    best, best_d = w, d
        if best is not None:
            used.add(best)
            control[best] = freq[best]
    return control


def build_random_vocab(rng, size: int = 10000) -> list[str]:
    """A fixed vocabulary of random lowercase words (3-8 chars), following the
    paper's random-string injection recipe."""
    import string
    return [
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
        for _ in range(size)
    ]


def resolve_storage(storage_root: str) -> tuple[Path, Path]:
    """Return (data_dir, runs_dir) under ``storage_root``, creating them."""
    root = Path(storage_root)
    data_dir, runs_dir = root / "data", root / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, runs_dir

"""Corpus perturbation.

Given a document, produce a perturbed variant under one intervention and a full
edit log. Interventions fall into three groups:

Modal interventions (the object of study):
    modal_mask         replace each modal with a single sentinel "[MODAL]"
    modal_delete       remove the modal entirely (aux / semi-modal only; deleting
                       adjectival modals tends to yield ungrammatical text)
    modal_within_swap  swap a modal for another of the SAME force class
                       (WEAK<->WEAK, STRONG<->STRONG): blurs lexical identity,
                       preserves inferential force
    modal_cross_swap   swap across force (WEAK<->STRONG): preserves fluency but
                       corrupts inferential force
    modal_polarity     flip polarity (can -> cannot, possible -> impossible):
                       teaches WRONG conditionals. This is the arm the Bitter
                       Lesson paper (Sec. 6) predicts is genuinely harmful,
                       distinct from the merely information-removing arms above.

Frequency-matched control (the confound check):
    control_mask       apply the SAME masking operation to a set of NON-modal
                       tokens whose corpus frequency matches the modal set.
                       Without this, "modal masking raised loss" cannot be
                       separated from "you perturbed high-frequency tokens".

Baselines (from the paper):
    shuffle_words      shuffle word order within the document (unigram preserved)
    random_strings     replace with random character strings

Every intervention returns a `PerturbedDoc` with a per-edit log so the result
is fully auditable and the perturbation metadata can be persisted.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field
from functools import lru_cache

from .modals import (
    FORCE_ORDER,
    KIND_AUX,
    KIND_LEXICAL,
    KIND_SEMIMODAL,
    NECESSITY,
    POSSIBILITY,
    PREDICTION,
    WEAK_NECESSITY,
    ModalSpan,
    detect_spans,
)

MASK_TOKEN = "[MODAL]"

# Intervention names (single source of truth).
MODAL_MASK = "modal_mask"
MODAL_DELETE = "modal_delete"
MODAL_WITHIN_SWAP = "modal_within_swap"
MODAL_CROSS_SWAP = "modal_cross_swap"
MODAL_POLARITY = "modal_polarity"
CONTROL_MASK = "control_mask"
SHUFFLE_WORDS = "shuffle_words"
RANDOM_STRINGS = "random_strings"

MODAL_INTERVENTIONS = (
    MODAL_MASK, MODAL_DELETE, MODAL_WITHIN_SWAP, MODAL_CROSS_SWAP, MODAL_POLARITY,
)
ALL_INTERVENTIONS = MODAL_INTERVENTIONS + (CONTROL_MASK, SHUFFLE_WORDS, RANDOM_STRINGS)

# Polarity flips. Auxiliary flips are positive -> negative only: the detector
# keys on bare modals, so we add negation rather than removing it. Lexical
# adjectives flip both ways because both poles are detected lemmas.
POLARITY_FLIP: dict[str, str] = {
    # auxiliaries (positive -> negative)
    "can": "cannot", "could": "could not", "may": "may not", "might": "might not",
    "must": "must not", "shall": "shall not", "should": "should not",
    "ought": "ought not", "will": "will not", "would": "would not",
    # semi-modals
    "have to": "do not have to", "need to": "do not need to", "able to": "unable to",
    # lexical adjectives/adverbs (bidirectional)
    "possible": "impossible", "impossible": "possible",
    "likely": "unlikely", "unlikely": "likely",
    "probable": "improbable", "improbable": "probable",
    "necessary": "unnecessary", "unnecessary": "necessary",
    "certain": "uncertain", "uncertain": "certain",
}

# Swaps apply to auxiliary and lexical modals only; semi-modal swaps risk
# ungrammatical output ("is able to" -> "is have to").
#
# Auxiliary swap-target pools, by force. "ought" is detected but excluded
# (singleton WEAK_NECESSITY aux pool, and it requires a following "to" that a
# bare swap would orphan), so WEAK_NECESSITY has no within-class aux swap.
_AUX_SWAP = {
    POSSIBILITY: ("can", "could", "may", "might"),
    WEAK_NECESSITY: ("should",),
    NECESSITY: ("must", "shall"),
    PREDICTION: ("will", "would"),
}

# Lexical swap pools keyed by (coarse POS, force). Only same-direction (non-
# negated) synonyms are listed, so a within-class swap blurs lexical identity
# WITHOUT flipping truth-direction (that is what modal_polarity is for). A span
# whose lemma is not in its own pool (e.g. "unlikely", "impossible") is detected
# and counted but is not swapped -- it can only be a polarity target.
_LEX_SWAP = {
    ("ADJ", POSSIBILITY): ("possible",),
    ("ADJ", WEAK_NECESSITY): ("likely", "probable"),
    ("ADJ", NECESSITY): ("necessary", "certain"),
    ("ADV", POSSIBILITY): ("possibly", "perhaps", "maybe"),
    ("ADV", WEAK_NECESSITY): ("probably",),
    ("ADV", NECESSITY): ("necessarily", "certainly", "definitely"),
}


@dataclass
class Edit:
    op: str
    kind: str
    force: str                  # source force of the modal
    original: str
    replacement: str | None     # None == deletion
    char_start: int
    char_end: int
    target_force: str = ""      # force of the swap target (swaps only)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PerturbedDoc:
    text: str
    intervention: str
    edits: list[Edit] = field(default_factory=list)
    n_targets: int = 0          # candidate spans considered
    n_applied: int = 0          # edits actually made

    @property
    def metadata(self) -> dict:
        return {
            "intervention": self.intervention,
            "n_targets": self.n_targets,
            "n_applied": self.n_applied,
            "edits": [e.as_dict() for e in self.edits],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_case(template: str, replacement: str) -> str:
    """Carry the capitalization of ``template`` onto ``replacement`` so a
    sentence-initial swap stays capitalized (e.g. "Must" -> "Should")."""
    if not template or not replacement:
        return replacement
    if template[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _apply_edits(text: str, edits: list[Edit]) -> str:
    """Apply character-span edits right-to-left so offsets stay valid.
    A ``None`` replacement deletes the span plus one adjacent space to avoid
    leaving a double space."""
    out = text
    for e in sorted(edits, key=lambda x: x.char_start, reverse=True):
        if e.replacement is None:
            start, end = e.char_start, e.char_end
            # consume one trailing space, else one leading space
            if end < len(out) and out[end] == " ":
                end += 1
            elif start > 0 and out[start - 1] == " ":
                start -= 1
            out = out[:start] + out[end:]
        else:
            out = out[:e.char_start] + e.replacement + out[e.char_end:]
    return out


def _pick_other(pool: tuple[str, ...] | list[str], current: str,
                rng: random.Random) -> str | None:
    choices = [w for w in pool if w != current]
    return rng.choice(choices) if choices else None


def _swap_target(span: ModalSpan, cross: bool,
                 rng: random.Random) -> tuple[str, str] | None:
    """Choose a swap replacement for ``span`` as (lemma, target_force), or None
    when no valid target exists. ``cross`` selects a DIFFERENT non-PREDICTION
    force class (POSSIBILITY / WEAK_NECESSITY / NECESSITY); otherwise the same
    class. PREDICTION (will/would) is only ever swapped within-class. A span
    whose lemma is not in its own swap pool (e.g. "ought", "unlikely") is left
    untouched -- those are polarity-only targets."""
    pool_map = _AUX_SWAP if span.kind == KIND_AUX else (
        _LEX_SWAP if span.kind == KIND_LEXICAL else None)
    if pool_map is None:
        return None  # semi-modals are not swapped

    def pool(force):
        return pool_map[force] if span.kind == KIND_AUX else \
            pool_map.get((span.pos, force), ())

    own = pool(span.force)
    if span.lemma not in own:
        return None

    if not cross:
        tgt = _pick_other(own, span.lemma, rng)
        return (tgt, span.force) if tgt else None

    # cross: pick a different non-PREDICTION force class that has a usable pool
    if span.force == PREDICTION:
        return None
    candidates = [f for f in FORCE_ORDER if f != span.force and pool(f)]
    if not candidates:
        return None
    tgt_force = rng.choice(candidates)
    return rng.choice(pool(tgt_force)), tgt_force
    return None  # semi-modals are not swapped


# ---------------------------------------------------------------------------
# Modal interventions
# ---------------------------------------------------------------------------

def _perturb_modal(text: str, intervention: str, rng: random.Random,
                   spans: list[ModalSpan] | None) -> PerturbedDoc:
    if spans is None:
        spans = detect_spans(text)
    edits: list[Edit] = []
    for sp in spans:
        replacement: str | None
        target_force = ""
        if intervention == MODAL_MASK:
            replacement = MASK_TOKEN
        elif intervention == MODAL_DELETE:
            if sp.kind == KIND_LEXICAL:
                continue  # deleting adjectival modals yields ungrammatical text
            replacement = None
        elif intervention in (MODAL_WITHIN_SWAP, MODAL_CROSS_SWAP):
            res = _swap_target(sp, cross=(intervention == MODAL_CROSS_SWAP), rng=rng)
            if res is None:
                continue
            tgt, target_force = res
            replacement = _match_case(sp.text, tgt)
        elif intervention == MODAL_POLARITY:
            tgt = POLARITY_FLIP.get(sp.lemma)
            if tgt is None:
                continue
            replacement = _match_case(sp.text, tgt)
        else:
            raise ValueError(f"not a modal intervention: {intervention}")
        edits.append(Edit(
            op=intervention, kind=sp.kind, force=sp.force,
            original=sp.text, replacement=replacement,
            char_start=sp.char_start, char_end=sp.char_end,
            target_force=target_force,
        ))
    new_text = _apply_edits(text, edits)
    return PerturbedDoc(text=new_text, intervention=intervention, edits=edits,
                        n_targets=len(spans), n_applied=len(edits))


# ---------------------------------------------------------------------------
# Frequency-matched control
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _control_pattern(vocab: frozenset) -> re.Pattern:
    # whole-word, case-insensitive alternation over the control word types
    alt = "|".join(re.escape(w) for w in sorted(vocab, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)


def _perturb_control_mask(text: str, control_vocab: set[str]) -> PerturbedDoc:
    """Mask occurrences of non-modal tokens in ``control_vocab`` using the same
    sentinel as modal_mask. ``control_vocab`` is lowercased surface forms chosen
    (upstream) to match the modal frequency profile. Uses a word-boundary regex
    (no spaCy): the control vocab was itself selected from regex word counts, so
    this is both faster and more consistent than a second tagging pass."""
    if not control_vocab:
        return PerturbedDoc(text=text, intervention=CONTROL_MASK)
    pat = _control_pattern(frozenset(control_vocab))
    edits = [
        Edit(op=CONTROL_MASK, kind="control", force="NA",
             original=m.group(0), replacement=MASK_TOKEN,
             char_start=m.start(), char_end=m.end())
        for m in pat.finditer(text)
    ]
    new_text = _apply_edits(text, edits)
    return PerturbedDoc(text=new_text, intervention=CONTROL_MASK, edits=edits,
                        n_targets=len(edits), n_applied=len(edits))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _perturb_shuffle(text: str, rng: random.Random) -> PerturbedDoc:
    words = text.split()
    shuffled = words[:]
    rng.shuffle(shuffled)
    return PerturbedDoc(text=" ".join(shuffled), intervention=SHUFFLE_WORDS,
                        n_targets=len(words), n_applied=len(words))


def _random_word(rng: random.Random) -> str:
    n = rng.randint(3, 8)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def _perturb_random(text: str, rng: random.Random,
                    random_vocab: list[str] | None) -> PerturbedDoc:
    words = text.split()
    if random_vocab:
        new = [rng.choice(random_vocab) for _ in words]
    else:
        new = [_random_word(rng) for _ in words]
    return PerturbedDoc(text=" ".join(new), intervention=RANDOM_STRINGS,
                        n_targets=len(words), n_applied=len(words))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def perturb_document(text: str, intervention: str, *,
                     seed: int = 0,
                     spans: list[ModalSpan] | None = None,
                     control_vocab: set[str] | None = None,
                     random_vocab: list[str] | None = None) -> PerturbedDoc:
    """Apply one ``intervention`` to ``text``. ``seed`` makes swap/shuffle/
    random choices reproducible. ``spans`` may be passed to reuse a prior
    detection pass; ``control_vocab`` is required for control_mask."""
    rng = random.Random(seed)
    if intervention in MODAL_INTERVENTIONS:
        return _perturb_modal(text, intervention, rng, spans)
    if intervention == CONTROL_MASK:
        if control_vocab is None:
            raise ValueError("control_mask requires control_vocab")
        return _perturb_control_mask(text, control_vocab)
    if intervention == SHUFFLE_WORDS:
        return _perturb_shuffle(text, rng)
    if intervention == RANDOM_STRINGS:
        return _perturb_random(text, rng, random_vocab)
    raise ValueError(f"unknown intervention: {intervention!r}")

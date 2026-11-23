"""Modal detection.

Detection is POS/tag-aware (spaCy), NOT naive string matching, because every
core modal lexeme is polysemous:

    can     modal aux ("you can go")      vs noun ("a can of soup")
    may     modal aux ("you may leave")   vs proper noun (month / name)
    might   modal aux ("it might rain")   vs noun ("with all his might")
    will    modal aux ("it will rain")    vs noun/verb ("a will", "to will")
    have to obligation ("I have to go")   vs possession ("I have two")

A spaCy `MD` tag identifies the core auxiliary modals. Periphrastic / semi-modal
constructions ("have to", "need to", "able to") are matched structurally so we
only fire on the modal reading. Modal adjectives/adverbs (possible, necessary,
likely, ...) are matched by lemma gated on coarse POS.

Every detected occurrence is returned as a `ModalSpan` carrying the token
offsets, surface form, lemma, syntactic kind, and a *force class* (WEAK /
STRONG / PREDICTION). The force class is assigned at the LEXICAL level. This is
a deliberate simplification: "can"/"could"/"may"/"must" are sense-ambiguous
across epistemic, deontic, and ability readings, and we do NOT disambiguate
sense in this pilot. Downstream swaps therefore operate on canonical lexical
class, and that limitation is documented rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import spacy
from spacy.matcher import Matcher

# ---------------------------------------------------------------------------
# Lexical inventories
# ---------------------------------------------------------------------------

# Force classes for the core auxiliary modals, grounded in Kratzer-style modal
# FORCE (the axis the perturbation experiments target). Flavor (epistemic /
# deontic / dynamic) is context-dependent, NOT lexical, so we do NOT classify by
# flavor -- one lexical "must" whose flavor is set by context (Kratzer 1977).
#   POSSIBILITY     : possibility / permission / ability   (may, might, can, could)
#   WEAK_NECESSITY  : gradable / defeasible necessity      (should, ought)
#   NECESSITY       : strong necessity / obligation        (must, shall)
#   PREDICTION      : future / volition                    (will, would)
# The POSSIBILITY / WEAK_NECESSITY / NECESSITY split follows the formal
# weak-necessity distinction (von Fintel & Iatridou 2008): "should/ought" are
# weaker than "must", so they are NOT force-equivalent and must not share a
# within-class swap pool with "must".
POSSIBILITY = "POSSIBILITY"
WEAK_NECESSITY = "WEAK_NECESSITY"
NECESSITY = "NECESSITY"
PREDICTION = "PREDICTION"

# Non-prediction force classes, ordered weak -> strong (used for cross-swaps).
FORCE_ORDER = (POSSIBILITY, WEAK_NECESSITY, NECESSITY)

FORCE_CLASSES: dict[str, tuple[str, ...]] = {
    POSSIBILITY: ("can", "could", "may", "might"),
    WEAK_NECESSITY: ("should", "ought"),
    NECESSITY: ("must", "shall"),
    PREDICTION: ("will", "would"),
}

# Reverse map: surface lemma -> force class.
AUX_FORCE: dict[str, str] = {
    word: cls for cls, words in FORCE_CLASSES.items() for word in words
}

# Semi-modals matched structurally (see Matcher patterns below). Force class is
# the value; the matched span may be multi-token.
SEMIMODAL_FORCE: dict[str, str] = {
    "have to": NECESSITY,
    "need to": NECESSITY,
    "able to": POSSIBILITY,
}

# Modal adjectives / adverbs, matched by lemma gated on coarse POS (ADJ/ADV).
# Mapped to a force class so within/cross-class swaps can include them. Negated
# poles (impossible, unlikely, ...) are detected and counted but are treated as
# polarity targets, not swap sources (see perturb._LEX_SWAP).
MODAL_LEXICAL: dict[str, str] = {
    "possible": POSSIBILITY,
    "impossible": NECESSITY,        # asserts necessity-of-not
    "necessary": NECESSITY,
    "unnecessary": POSSIBILITY,
    "likely": WEAK_NECESSITY,
    "unlikely": WEAK_NECESSITY,
    "probable": WEAK_NECESSITY,
    "improbable": WEAK_NECESSITY,
    "certain": NECESSITY,
    "uncertain": POSSIBILITY,
    "possibly": POSSIBILITY,
    "probably": WEAK_NECESSITY,
    "perhaps": POSSIBILITY,
    "maybe": POSSIBILITY,
    "necessarily": NECESSITY,
    "certainly": NECESSITY,
    "definitely": NECESSITY,
}

# Analysis scope. PREDICTION modals (will/would) are the least epistemic/
# deontic and dominate raw counts; by default analysis centers on the
# epistemic-deontic CORE and reports PREDICTION separately.
PREDICTION_LEMMAS = frozenset({"will", "would"})


def is_core(lemma: str) -> bool:
    """True for the epistemic-deontic core (everything we detect except the
    PREDICTION modals will/would)."""
    return lemma not in PREDICTION_LEMMAS


# Kinds (the syntactic/lexical category of a detected span).
KIND_AUX = "aux"            # MD-tagged auxiliary: can, must, will, ...
KIND_SEMIMODAL = "semimodal"  # have to, need to, able to
KIND_LEXICAL = "lexical"    # possible, necessary, likely, ...


@dataclass
class ModalSpan:
    """One detected modal occurrence in a document.

    Token offsets are [start, end) into the spaCy ``Doc``. Character offsets
    are [char_start, char_end) into the original text, suitable for span edits.
    """

    start: int                 # token index, inclusive
    end: int                   # token index, exclusive
    char_start: int
    char_end: int
    text: str                  # surface form as it appears in the document
    lemma: str                 # normalized key (lowercased; semimodal -> "have to")
    kind: str                  # KIND_AUX | KIND_SEMIMODAL | KIND_LEXICAL
    force: str                 # POSSIBILITY | WEAK_NECESSITY | NECESSITY | PREDICTION
    pos: str = ""              # coarse POS of the head token (for audit)
    tag: str = ""              # fine tag of the head token (for audit)

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "lemma": self.lemma,
            "kind": self.kind,
            "force": self.force,
            "pos": self.pos,
            "tag": self.tag,
        }


# ---------------------------------------------------------------------------
# spaCy pipeline + matcher (cached; loaded once per process)
# ---------------------------------------------------------------------------

_SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _nlp() -> spacy.language.Language:
    # We need the tagger (MD tag), attribute_ruler + lemmatizer (lemmas/POS).
    # Parser and NER are not needed for detection; disabling them speeds the
    # pipeline up substantially on CPU. A rule-based sentencizer is added so
    # callers can harvest sentences (minimal-pair construction) without the
    # parser.
    nlp = spacy.load(_SPACY_MODEL, disable=["parser", "ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def _semimodal_matcher() -> Matcher:
    nlp = _nlp()
    matcher = Matcher(nlp.vocab)
    # "have/need to <VERB>" -- require a following verb so we fire on the
    # obligation reading and not on possessive "have to" fragments.
    matcher.add(
        "HAVE_NEED_TO",
        [[
            {"LEMMA": {"IN": ["have", "need"]}},
            {"LOWER": "to"},
            {"POS": {"IN": ["VERB", "AUX"]}},
        ]],
    )
    # "able to" (typically "be able to"); ability reading.
    matcher.add(
        "ABLE_TO",
        [[
            {"LOWER": "able"},
            {"LOWER": "to"},
        ]],
    )
    return matcher


def get_nlp() -> spacy.language.Language:
    """Public accessor for the shared spaCy pipeline."""
    return _nlp()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _semimodal_key(match_id: int, nlp: spacy.language.Language) -> str:
    name = nlp.vocab.strings[match_id]
    return {"HAVE_NEED_TO": None, "ABLE_TO": "able to"}.get(name, None)


def detect_spans(text: str) -> list[ModalSpan]:
    """Detect all modal occurrences in ``text`` (single document)."""
    return spans_from_doc(_nlp()(text))


def detect_spans_iter(texts, batch_size: int = 256):
    """Batched detection over many documents using ``nlp.pipe`` (much faster on
    CPU than one call per document). Yields a ``list[ModalSpan]`` per input."""
    for doc in _nlp().pipe(texts, batch_size=batch_size):
        yield spans_from_doc(doc)


def spans_from_doc(doc) -> list[ModalSpan]:
    """Extract modal spans from an already-parsed spaCy ``Doc``. Overlapping
    spans are resolved in favor of the longer (semi-modal) span so "have to" is
    not also reported as two pieces."""
    spans: list[ModalSpan] = []
    nlp = _nlp()
    claimed: set[int] = set()  # token indices already consumed by a longer span

    # 1) Semi-modals first (they are multi-token and take precedence).
    matcher = _semimodal_matcher()
    raw = matcher(doc)
    # Sort by length descending so the longest match at a position wins.
    raw_sorted = sorted(raw, key=lambda m: (m[1], -(m[2] - m[1])))
    matcher_strings = nlp.vocab.strings
    for match_id, start, end in raw_sorted:
        if any(i in claimed for i in range(start, end)):
            continue
        name = matcher_strings[match_id]
        if name == "ABLE_TO":
            lemma, force = "able to", SEMIMODAL_FORCE["able to"]
            span_end = end
        else:  # HAVE_NEED_TO: span is [have/need, to] -- exclude the verb
            head = doc[start]
            lemma = "have to" if head.lemma_ == "have" else "need to"
            force = SEMIMODAL_FORCE[lemma]
            span_end = start + 2  # head + "to" only
        sp = doc[start:span_end]
        spans.append(ModalSpan(
            start=start, end=span_end,
            char_start=sp.start_char, char_end=sp.end_char,
            text=sp.text, lemma=lemma, kind=KIND_SEMIMODAL, force=force,
            pos=doc[start].pos_, tag=doc[start].tag_,
        ))
        claimed.update(range(start, span_end))

    # 2) Auxiliary modals (MD tag) and modal lexical items (ADJ/ADV by lemma).
    for tok in doc:
        if tok.i in claimed:
            continue
        low = tok.lower_
        lemma = tok.lemma_.lower()
        if tok.tag_ == "MD" and low in AUX_FORCE:
            spans.append(ModalSpan(
                start=tok.i, end=tok.i + 1,
                char_start=tok.idx, char_end=tok.idx + len(tok),
                text=tok.text, lemma=low, kind=KIND_AUX, force=AUX_FORCE[low],
                pos=tok.pos_, tag=tok.tag_,
            ))
            claimed.add(tok.i)
        elif tok.pos_ in ("ADJ", "ADV") and lemma in MODAL_LEXICAL:
            spans.append(ModalSpan(
                start=tok.i, end=tok.i + 1,
                char_start=tok.idx, char_end=tok.idx + len(tok),
                text=tok.text, lemma=lemma, kind=KIND_LEXICAL,
                force=MODAL_LEXICAL[lemma], pos=tok.pos_, tag=tok.tag_,
            ))
            claimed.add(tok.i)

    spans.sort(key=lambda s: s.start)
    return spans


def iter_modal_lemmas() -> Iterable[str]:
    """All modal lemma keys we recognize (for building control vocabularies)."""
    yield from AUX_FORCE
    yield from SEMIMODAL_FORCE
    yield from MODAL_LEXICAL

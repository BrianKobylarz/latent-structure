"""Detection precision on hand-labeled sentences. These pin the behavior that
gates everything downstream: that we fire on genuine modals and NOT on their
common noun/verb/proper-noun homographs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_structure.modals import detect_spans


def lemmas(text):
    return sorted(s.lemma for s in detect_spans(text))


def test_positive_auxiliaries():
    assert lemmas("You must leave.") == ["must"]
    assert lemmas("It might rain so you should go.") == ["might", "should"]
    assert lemmas("She can and she will.") == ["can", "will"]


def test_homographs_excluded():
    # "a can of soup" (noun), "all his might" (noun), "in May" (proper noun)
    assert detect_spans("He drank a can of soup with all his might in May.") == []


def test_possessive_have_not_modal():
    # "have two" must NOT be detected; "have to go" must be
    sp = detect_spans("I have two apples and I have to go.")
    assert [s.lemma for s in sp] == ["have to"]


def test_semimodal_spans_exclude_verb():
    sp = detect_spans("I have to go.")
    assert len(sp) == 1
    assert sp[0].text == "have to"        # span is head + "to", not the verb


def test_lexical_modals():
    assert lemmas("It is possible but unlikely and not necessary.") == \
        ["necessary", "possible", "unlikely"]


def test_force_classes():
    by_lemma = {s.lemma: s.force for s in detect_spans(
        "It might be necessary, and you should, but you must, and it will.")}
    assert by_lemma["might"] == "POSSIBILITY"
    assert by_lemma["should"] == "WEAK_NECESSITY"   # distinct from must
    assert by_lemma["must"] == "NECESSITY"
    assert by_lemma["necessary"] == "NECESSITY"
    assert by_lemma["will"] == "PREDICTION"


def test_char_offsets_align():
    text = "Perhaps she could come."
    for s in detect_spans(text):
        assert text[s.char_start:s.char_end] == s.text

"""Perturbation invariants. Each intervention must do exactly what it claims
and nothing else, so downstream loss differences are attributable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_structure import perturb as P
from latent_structure.modals import detect_spans

SENT = "You must leave, and she might stay because it is possible she can help."


def test_mask_replaces_each_modal_with_sentinel():
    pd = P.perturb_document(SENT, P.MODAL_MASK, seed=0)
    assert pd.n_applied == pd.n_targets == 4
    assert pd.text.count(P.MASK_TOKEN) == 4
    for w in ("must", "might", "can"):
        assert f" {w} " not in f" {pd.text} "


def test_delete_skips_lexical_and_shrinks():
    pd = P.perturb_document(SENT, P.MODAL_DELETE, seed=0)
    # 4 modals detected, but the lexical "possible" is not deleted
    assert pd.n_applied == 3
    assert len(pd.text) < len(SENT)
    assert "  " not in pd.text          # no double spaces left behind


def test_within_swap_preserves_force_class():
    text = "It might be true."
    pd = P.perturb_document(text, P.MODAL_WITHIN_SWAP, seed=1)
    new = detect_spans(pd.text)
    assert new and new[0].force == "POSSIBILITY"  # might -> may/can/could


def test_cross_swap_flips_force_class():
    text = "It might be true."
    pd = P.perturb_document(text, P.MODAL_CROSS_SWAP, seed=1)
    new = detect_spans(pd.text)
    # might (POSSIBILITY) -> a different non-PREDICTION force
    assert new and new[0].force in ("WEAK_NECESSITY", "NECESSITY")


def test_polarity_introduces_negation():
    pd = P.perturb_document("You can leave.", P.MODAL_POLARITY, seed=0)
    assert "cannot" in pd.text


def test_shuffle_preserves_unigrams():
    text = "the cat sat on the mat"
    pd = P.perturb_document(text, P.SHUFFLE_WORDS, seed=2)
    assert sorted(pd.text.split()) == sorted(text.split())


def test_control_mask_requires_vocab():
    try:
        P.perturb_document(SENT, P.CONTROL_MASK)
    except ValueError:
        return
    raise AssertionError("control_mask without vocab should raise")


def test_case_preserved_on_swap():
    pd = P.perturb_document("Must we go?", P.MODAL_WITHIN_SWAP, seed=0)
    assert pd.text[0].isupper()         # sentence-initial stays capitalized


def test_edits_are_logged():
    pd = P.perturb_document(SENT, P.MODAL_MASK, seed=0)
    assert len(pd.edits) == pd.n_applied
    for e in pd.edits:
        assert e.original and e.replacement == P.MASK_TOKEN

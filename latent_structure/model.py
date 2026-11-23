"""Tiny GPT factory for the pilot.

We use the standard `transformers` GPT-2 implementation (not a hand-rolled
transformer) so the architecture is auditable and bug-free; only the size is
small. The GPT-2 BPE tokenizer is reused -- most modals are single tokens --
with one addition: a dedicated ``[MODAL]`` token so the mask interventions
occupy exactly one position (matching the intent of the operation) rather than
fragmenting into "[ MOD AL ]".

Note on parameter counts: with the 50k GPT-2 vocab, the (tied) token embedding
dominates a small model. We report NON-embedding parameters separately because
that is the capacity that matters for the Bitter Lesson "enough rank to absorb
structure" intuition.
"""

from __future__ import annotations

MASK_TOKEN = "[MODAL]"

# (n_embd, n_layer, n_head, context length)
PRESETS: dict[str, dict] = {
    "tiny":   {"n_embd": 256, "n_layer": 4, "n_head": 4,  "ctx": 256},
    "small":  {"n_embd": 512, "n_layer": 6, "n_head": 8,  "ctx": 512},
    "medium": {"n_embd": 768, "n_layer": 8, "n_head": 12, "ctx": 512},
}


def build_tokenizer():
    """GPT-2 BPE tokenizer with a single-token ``[MODAL]`` sentinel and a pad
    token (= eos) for batching."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # We chunk to the model context ourselves; silence the >1024 length warning
    # emitted when encoding whole documents.
    tok.model_max_length = 10_000_000
    return tok


def build_model(preset: str, tokenizer):
    """Construct an (untrained) GPT-2 LM at the given size preset, with the
    embedding resized to the tokenizer (which now includes ``[MODAL]``)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(PRESETS)}")
    p = PRESETS[preset]
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=p["ctx"],
        n_embd=p["n_embd"],
        n_layer=p["n_layer"],
        n_head=p["n_head"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(tokenizer))
    return model


def build_raw_model(preset: str, vocab_size: int, ctx: int | None = None):
    """Build an (untrained) GPT-2 LM for a raw integer-token vocabulary (no
    tokenizer) -- used for the synthetic planted-structure corpora."""
    from transformers import GPT2Config, GPT2LMHeadModel
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choices: {list(PRESETS)}")
    p = PRESETS[preset]
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=ctx or p["ctx"],
        n_embd=p["n_embd"],
        n_layer=p["n_layer"],
        n_head=p["n_head"],
    )
    return GPT2LMHeadModel(cfg)


def param_counts(model) -> dict[str, int]:
    """Total and non-embedding parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    pos = model.transformer.wpe.weight.numel()
    return {"total": total, "non_embedding": total - emb - pos}


def context_length(preset: str) -> int:
    return PRESETS[preset]["ctx"]

"""D-2: unsupervised discovery -- SAE on a capable model over natural text,
then flag features NOT explained by KNOWN properties as candidate non-obvious
structure (and surface their top-activating contexts for inspection).

Pipeline: parse WikiText (spaCy: POS, dependency depth, position-in-sentence per
word) -> GPT-2 forward, collect last-subtoken activations + per-token properties ->
train an SAE -> for each active feature, classify it:
  token-detector  (one token dominates its top activations)
  pos / depth / position feature (activation explained by that known property)
  UNEXPLAINED     (active, but none of the above) -> candidate non-obvious structure

We report the feature taxonomy (does the SAE find interpretable structure at all?)
and the top UNEXPLAINED features with example contexts. Honest: this is exploratory;
"unexplained" can be polysemy/noise OR genuine structure -- the examples are for a
human to judge, and token-identity/POS explain most features.

Usage:
    python scripts/17_sae_discover.py --config config/pilot.yaml --model gpt2 --layer 6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import yaml
from transformers.utils.logging import disable_progress_bar, set_verbosity_error

disable_progress_bar(); set_verbosity_error()

from latent_structure import corpus as C
from latent_structure import sae as SAE

POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
            "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
POS_ID = {p: i for i, p in enumerate(POS_TAGS)}


def parse_props(n_sentences, split, max_documents=400, depth_clip=12):
    """[(text, [(cs, ce, pos_id, depth, position_frac, word), ...]), ...]."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    docs = C.build_corpus(split=split, max_documents=max_documents, min_doc_chars=200)
    out = []
    for doc in nlp.pipe(docs, batch_size=32):
        for sent in doc.sents:
            toks = [t for t in sent if not t.is_space]
            if not (6 <= len(toks) <= 60):
                continue
            base = sent.start_char
            rows = []
            for k, t in enumerate(toks):
                d = 0; x = t
                while x.head != x and d < 60:
                    x = x.head; d += 1
                rows.append((t.idx - base, t.idx - base + len(t),
                             POS_ID.get(t.pos_, POS_ID["X"]), min(d, depth_clip),
                             k / max(1, len(toks) - 1), t.text))
            out.append((sent.text, rows))
            if len(out) >= n_sentences:
                return out
    return out


def collect(model, tok, sents, layer, device, max_tokens=60000):
    import torch
    model.eval()
    acts, pos, depth, posn, tokid, ctx = [], [], [], [], [], []
    with torch.no_grad():
        for si, (text, rows) in enumerate(sents):
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) < 3 or len(ids) > 1024:
                continue
            h = model(input_ids=torch.tensor([ids], device=device),
                      output_hidden_states=True).hidden_states[layer][0]
            for cs, ce, p, d, pn, word in rows:
                toks = [j for j, (a, b) in enumerate(offs)
                        if a != b and not (b <= cs or a >= ce)]
                if not toks:
                    continue
                acts.append(h[toks[-1]].float().cpu().numpy())
                pos.append(p); depth.append(d); posn.append(pn)
                tokid.append(ids[toks[-1]]); ctx.append((si, word))
            if len(acts) >= max_tokens:
                break
    return (np.asarray(acts), np.asarray(pos), np.asarray(depth, dtype=float),
            np.asarray(posn), np.asarray(tokid), ctx)


def explained_r2(F_col, props):
    from sklearn.linear_model import Ridge
    pred = Ridge(alpha=1.0).fit(props, F_col).predict(props)
    ss_res = ((F_col - pred) ** 2).sum(); ss_tot = ((F_col - F_col.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n-sentences", type=int, default=4000)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--l1", type=float, default=0.2)
    ap.add_argument("--top-unexplained", type=int, default=12)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[d2] parsing + collecting activations ...", flush=True)
    sents = parse_props(args.n_sentences, cfg["corpus"]["eval_split"])
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    X, pos, depth, posn, tokid, ctx = collect(model, tok, sents, args.layer, device)
    print(f"[d2] X={X.shape}; training SAE ...", flush=True)
    sae, stats = SAE.train_sae(X, expansion=args.expansion, l1=args.l1, device=device,
                               seed=seed, log=print)
    F = SAE.encode(sae, X, device=device)

    # known-property design matrix: POS one-hot + depth + position + log-freq
    freq = Counter(tokid.tolist())
    logf = np.log(np.array([freq[t] for t in tokid]) + 1.0)
    pos_oh = np.eye(len(POS_TAGS))[pos]
    props = np.column_stack([pos_oh, (depth - depth.mean()), posn, logf])

    active = np.where((F > 1e-6).mean(0) > 0.005)[0]   # fires on >0.5% of tokens
    rows = []
    for j in active:
        col = F[:, j]
        fire = col > 1e-6
        top = np.argsort(col)[-100:]
        toptokens = Counter(tokid[i] for i in top)
        tok_dom = toptokens.most_common(1)[0][1] / 100.0
        r2 = explained_r2(col, props)
        rows.append((int(j), float(tok_dom), float(r2), int(fire.sum())))
    rows.sort(key=lambda r: (r[1] < 0.5, -((1 - r[2]) * (r[1] < 0.5))))  # unexplained first

    taxonomy = {"token_detector": sum(r[1] >= 0.5 for r in rows),
                "explained_by_props": sum(r[1] < 0.5 and r[2] > 0.3 for r in rows),
                "unexplained": sum(r[1] < 0.5 and r[2] <= 0.3 for r in rows),
                "n_active": len(rows)}
    print(f"\n[d2] SAE FVU={stats['fvu']:.3f} L0={stats['l0']:.1f} "
          f"({stats['n_features']} features, {len(active)} active)")
    print(f"[d2] feature taxonomy: {taxonomy}")

    print(f"\n[d2] top {args.top_unexplained} UNEXPLAINED features (not token-detector, "
          f"known-property R2<=0.3) -- top-activating words:")
    unexpl = [r for r in rows if r[1] < 0.5 and r[2] <= 0.3][:args.top_unexplained]
    report = []
    for j, dom, r2, nfire in unexpl:
        top = np.argsort(F[:, j])[-8:][::-1]
        words = [ctx[i][1] for i in top]
        print(f"  feat {j:5d} (R2={r2:.2f}, tokdom={dom:.2f}, fires {nfire}): "
              f"{', '.join(repr(w) for w in words)}")
        report.append({"feature": j, "known_r2": r2, "token_dominance": dom,
                       "n_fire": nfire, "top_words": words})

    _, runs_dir = C.resolve_storage(cfg["storage_root"])
    out = runs_dir / f"sae_discover_{args.model}_L{args.layer}.json"
    out.write_text(json.dumps({"stats": stats, "taxonomy": taxonomy,
                               "top_unexplained": report}, indent=2), encoding="utf-8")
    print(f"\n[d2] wrote {out}")


if __name__ == "__main__":
    main()

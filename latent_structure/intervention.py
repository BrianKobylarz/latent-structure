"""Causal interchange intervention on the planted-structure models.

Probing shows what is REPRESENTED; this shows what is USED. We patch the
residual stream at a prediction position with a donor residual taken from a
context on the OTHER side of the depth threshold, then check whether the model's
A/B marker prediction flips to the donor's side. A high opposite-side flip rate
(with a low same-side control flip rate) is causal evidence that the residual
carries the depth the marker decision actually uses -- ruling out a bigram
shortcut.

Prediction position: index i such that token i+1 is a marker; the model's logits
at position i decide that marker, and the governing depth is `depth_after[i]`.
"""

from __future__ import annotations

import numpy as np


def collect_pred_positions(corpus, rows):
    """List of (row, pos_i, side) where token i+1 is a marker; side = 1 if the
    governing depth (depth_after[i]) >= thresh else 0."""
    thresh = corpus.config.effective_thresh
    out = []
    for r in rows:
        mpos = np.flatnonzero(corpus.is_marker[r])
        for t in mpos:
            if t == 0:
                continue
            i = t - 1
            side = int(corpus.depth_after[r, i] >= thresh)
            out.append((int(r), int(i), side))
    return out


def _layer_activations(model, tokens_batch, layer, device):
    """hidden_states[layer] for a batch of token rows (numpy int) -> (b, L, d)."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(tokens_batch.astype(np.int64)).to(device)
        out = model(input_ids=x, output_hidden_states=True)
    return out.hidden_states[layer]


def interchange_flip_rate(model, corpus, layer, device, *, n_pairs=800,
                          seed=0, batch=64):
    """Patch recipient residuals at `layer` with donor residuals from the
    opposite depth side and measure the A/B prediction flip rate. Also runs a
    same-side control (donors from the SAME side -> should not flip). The block
    that produces hidden_states[layer] is transformer.h[layer-1]."""
    import torch
    model.eval()
    cfg = corpus.config
    a_id, b_id = cfg.a_id, cfg.b_id
    rng = np.random.default_rng(seed)

    pos = collect_pred_positions(corpus, np.arange(corpus.tokens.shape[0]))
    low = [p for p in pos if p[2] == 0]
    high = [p for p in pos if p[2] == 1]
    rng.shuffle(low); rng.shuffle(high)

    block = model.transformer.h[layer - 1]

    def run_patched(recips, donors):
        """For paired recipient/donor positions, replace recipient residual at
        `layer`/pos with the donor residual, return patched predicted side (1=B)."""
        preds = []
        for s in range(0, len(recips), batch):
            rc = recips[s:s + batch]; dn = donors[s:s + batch]
            rtok = np.stack([corpus.tokens[r] for r, _, _ in rc])
            dtok = np.stack([corpus.tokens[r] for r, _, _ in dn])
            dact = _layer_activations(model, dtok, layer, device)  # (b,L,d)
            didx = torch.arange(len(dn))
            dpos = torch.tensor([i for _, i, _ in dn])
            donor_vec = dact[didx, dpos, :]                        # (b,d)
            rpos = [i for _, i, _ in rc]

            hook_state = {"donor": donor_vec, "pos": rpos}

            def hook(mod, inp, out):
                # the block returns a tuple or a tensor, depending on the library version
                hs = out[0] if isinstance(out, tuple) else out
                for bi, pi in enumerate(hook_state["pos"]):
                    hs[bi, pi, :] = hook_state["donor"][bi]
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

            h = block.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    x = torch.from_numpy(rtok.astype(np.int64)).to(device)
                    logits = model(input_ids=x).logits
            finally:
                h.remove()
            ridx = torch.arange(len(rc))
            rp = torch.tensor(rpos)
            pb = (logits[ridx, rp, b_id] > logits[ridx, rp, a_id]).cpu().numpy()
            preds.extend(pb.tolist())
        return np.array(preds)

    n = min(n_pairs, len(low), len(high))
    low_s, high_s = low[:n], high[:n]

    # opposite-side: recipient LOW (side 0, wants A) patched with HIGH donor -> flip to B?
    opp_pred = run_patched(low_s, high_s)
    opp_flip = float((opp_pred == 1).mean())          # flipped to donor's (HIGH/B) side
    # and recipient HIGH patched with LOW donor -> flip to A?
    opp_pred2 = run_patched(high_s, low_s)
    opp_flip2 = float((opp_pred2 == 0).mean())
    # same-side control: recipient LOW patched with another LOW donor -> should stay A
    ctrl_pred = run_patched(low_s, low[n:2 * n] if len(low) >= 2 * n else low_s)
    ctrl_flip = float((ctrl_pred == 1).mean())        # spurious flip to B

    return {
        "layer": layer, "n": n,
        "flip_low_to_high": opp_flip,     # LOW recipient + HIGH donor -> predicts B
        "flip_high_to_low": opp_flip2,    # HIGH recipient + LOW donor -> predicts A
        "opposite_side_flip": (opp_flip + opp_flip2) / 2,
        "same_side_flip_control": ctrl_flip,
    }

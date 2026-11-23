"""#3: polarity REASONING probe.

The minimal-pair eval (scripts/04) only tests surface polarity preference.
This tests whether context that FORCES a polarity makes the model prefer the
consistent modal: prohibition -> "must not", requirement -> "must", impossibility
-> "cannot", etc. Each item is a minimal pair differing ONLY in the modal's
polarity; we score by prefix-conditioned log-prob (the divergence is the modal).

Synthetic, hand-crafted templates. Small/from-scratch
models are not expected to reason; the informative substrate is pretrained models.
Honest caveat: small set, surface contextual-polarity sensitivity, not deep logic.

Usage:
    python scripts/11_polarity_reasoning.py --config config/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml
from transformers.utils.logging import disable_progress_bar, set_verbosity_error

disable_progress_bar(); set_verbosity_error()

from latent_structure import corpus as C
from latent_structure.minimal_pairs import _prefix_conditioned_scores

# (consistent, inconsistent) -- differ only in modal polarity; context forces it.
ITEMS = [
    # prohibition -> negative
    ("Entry to the vault is strictly forbidden, so the staff must not enter it.",
     "Entry to the vault is strictly forbidden, so the staff must enter it."),
    ("The chemical is poisonous if swallowed, so you must not drink it.",
     "The chemical is poisonous if swallowed, so you must drink it."),
    ("Photography is prohibited in the gallery, so visitors cannot take photos.",
     "Photography is prohibited in the gallery, so visitors can take photos."),
    ("Talking is banned during the exam, so students must not speak.",
     "Talking is banned during the exam, so students must speak."),
    ("The bridge is closed for repairs, so cars cannot cross it.",
     "The bridge is closed for repairs, so cars can cross it."),
    # requirement -> positive necessity
    ("A valid ticket is required for entry, so every passenger must show one.",
     "A valid ticket is required for entry, so every passenger must not show one."),
    ("The law requires drivers to stop at red lights, so they must stop.",
     "The law requires drivers to stop at red lights, so they must not stop."),
    ("Signing the form is mandatory, so each applicant must sign it.",
     "Signing the form is mandatory, so each applicant must not sign it."),
    # epistemic impossibility -> cannot
    ("The box was completely empty when we opened it, so the ring cannot be inside.",
     "The box was completely empty when we opened it, so the ring must be inside."),
    ("She was seen in Paris all week, so she cannot have been in Rome.",
     "She was seen in Paris all week, so she must have been in Rome."),
    ("Every window was sealed shut, so the bird cannot have flown out.",
     "Every window was sealed shut, so the bird must have flown out."),
    # epistemic necessity -> must
    ("All the other suspects have alibis, so the butler must be guilty.",
     "All the other suspects have alibis, so the butler cannot be guilty."),
    ("The ground is soaking wet this morning, so it must have rained.",
     "The ground is soaking wet this morning, so it cannot have rained."),
    ("The light is on and the engine is warm, so someone must be home.",
     "The light is on and the engine is warm, so someone cannot be home."),
    # permission -> may
    ("Guests are welcome to use the pool, so you may swim whenever you like.",
     "Guests are welcome to use the pool, so you may not swim whenever you like."),
    ("The museum offers free entry on Sundays, so you may visit at no cost.",
     "The museum offers free entry on Sundays, so you may not visit at no cost."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--models", nargs="*",
                    default=["gpt2", "gpt2-medium", "gpt2-large"])
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, runs_dir = C.resolve_storage(cfg["storage_root"])

    def load(name):
        if args.load_4bit:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
            return AutoModelForCausalLM.from_pretrained(name, quantization_config=qc,
                                                        device_map="cuda").eval()
        return AutoModelForCausalLM.from_pretrained(name).to(device).eval()

    targets = list(args.models)
    # add our clean from-scratch baseline (only in the non-4bit GPT-2-family run)
    if not args.load_4bit:
        for size in ("small", "tiny"):
            d = runs_dir / f"clean_{size}"
            if (d / "config.json").exists():
                targets.append(str(d))

    print(f"=== polarity reasoning ({len(ITEMS)} forced-polarity minimal pairs; "
          f"0.5 = chance) ===")
    # merge into any existing results so the table stays unified
    out_path = runs_dir / "polarity_reasoning.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in targets:
        model = load(name)
        tok = AutoTokenizer.from_pretrained(name)
        correct = 0
        margins = []
        with torch.no_grad():
            for cons, incon in ITEMS:
                sc_c, sc_i = _prefix_conditioned_scores(model, tok, cons, incon, device)
                if sc_c is None:
                    continue
                correct += int(sc_c > sc_i)
                margins.append(sc_c - sc_i)
        acc = correct / len(ITEMS)
        label = Path(name).name if "/" in name or "\\" in name else name
        results[label] = {"accuracy": acc, "mean_margin": float(sum(margins) / len(margins))}
        print(f"  {label:16s} acc={acc:.3f}  mean_margin={results[label]['mean_margin']:+.3f}",
              flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (runs_dir / "polarity_reasoning.json").write_text(json.dumps(results, indent=2),
                                                      encoding="utf-8")
    print(f"\n[polarity] wrote polarity_reasoning.json")


if __name__ == "__main__":
    main()

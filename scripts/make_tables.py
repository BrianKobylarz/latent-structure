"""Rebuild every results table from the files under results/.

    python scripts/make_tables.py        # writes results/TABLES.md

Standard library only; no GPU and no model download. Each table names the
file(s) it is read from, so any number quoted in the README can be traced to
its source and regenerated.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

PROBE_RE = re.compile(
    r"^\[probe (\S+) L(\d+)\] acc real=([\d.]+) ctrl=([\d.]+) \| MDL trained=(\d+)b "
    r"rand=(\d+)b \(rand/trained=([\d.]+)\)")
SAE_STATS_RE = re.compile(
    r"^\[d2\] SAE FVU=([\d.]+) L0=([\d.]+) \((\d+) features, (\d+) active\)")
SAE_TAX_RE = re.compile(r"^\[d2\] feature taxonomy: (\{.*\})")


def J(rel):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "-"
    return f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def probe_lines():
    """{(condition, layer): record} parsed from the saved console logs."""
    found = {}
    for log in sorted((RES / "logs").glob("*.txt")):
        for line in log.read_text(encoding="utf-8").splitlines():
            m = PROBE_RE.match(line)
            if m:
                found[(m.group(1), int(m.group(2)))] = {
                    "acc": float(m.group(3)), "ctrl": float(m.group(4)),
                    "mdl_trained": int(m.group(5)), "mdl_random": int(m.group(6)),
                    "ratio": float(m.group(7)), "log": log.name}
    return found


def final_eval(run):
    h = J(f"planted/{run}/history.json")
    e = h["evals"][-1]
    return {"acc": e["marker_acc"], "null": h["ngram_null"], "oracle": h["oracle"],
            "step": e["step"], "params": h["params"]["non_embedding"]}


def local_baselines():
    path = RES / "planted" / "local_baselines.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def planted_runs(prefix):
    return sorted((p.name for p in (RES / "planted").iterdir()
                   if p.is_dir() and re.fullmatch(prefix + r"rule_p_[\d.]+", p.name)),
                  key=lambda n: -float(n.rsplit("_", 1)[1]))


def planted_rows(prefix, probes):
    local = local_baselines()
    rows = []
    for run in planted_runs(prefix):
        e = final_eval(run)
        p = probes.get((run, 4))
        b = local.get(run)
        rows.append([run.rsplit("_", 1)[1], f(e["acc"]),
                     f(b["best_local_vs_rule"]) if b else "-", f(e["null"]), f(e["oracle"]),
                     f(p["acc"]) if p else "-", f(p["ctrl"]) if p else "-",
                     f(p["ratio"], 2) if p else "-"])
    return rows


def local_rows(prefix):
    """One row per run: every local predictor scored against the hidden rule."""
    local = local_baselines()
    rows = []
    for run in planted_runs(prefix):
        b = local.get(run)
        if not b:
            continue
        e = final_eval(run)
        raw = max(v["vs_rule"] for v in b["raw_ngram"].values())
        red = max(v["vs_rule"] for v in b["reduced_ngram"].values())
        rows.append([run.rsplit("_", 1)[1], f(b["constant_vs_rule"]),
                     f(b["copy_previous_marker"]["vs_rule"]), f(raw), f(red),
                     f(b["best_local_vs_rule"]), f(e["acc"]),
                     f"{e['acc'] - b['best_local_vs_rule']:+.3f}"])
    return rows


PLANTED_HEAD = ["rule_p", "marker accuracy (against the rule)",
                "best local predictor (against the rule)",
                "recorded n-gram baseline (against the emitted marker)",
                "share of emitted markers that equal the rule",
                "depth probe accuracy (layer 4)", "control-task accuracy",
                "codelength ratio, random-init / trained (layer 4)"]
LOCAL_HEAD = ["rule_p", "always the more common side", "copy the previous marker",
              "n-gram over raw tokens, best of 1, 2, 4, 8", "n-gram with filler tokens merged, best of 1, 2, 4, 8",
              "best of these", "model", "model minus best"]
LOCAL_NOTE = ("Source: `results/planted/local_baselines.json` (`scripts/18_local_baselines.py`, no "
              "model needed). Each predictor is fit on the first half of the sequences from the "
              "emitted markers and scored on the second half by its agreement with the hidden "
              "rule, which is how the model's marker accuracy is scored. The baseline stored "
              "with each run is the raw-token n-gram scored against the emitted marker.")


SUBSPACE_NAMES = {"rule_p_1.0": "synthetic, rule_p 1.0", "rule_p_0.05": "synthetic, rule_p 0.05",
                  "needle_rule_p_1.0": "needle mode, rule_p 1.0",
                  "natural_rule_p_1.0": "natural text, rule_p 1.0"}


def subspace_name(cond):
    base, _, seed = cond.partition("_seed")
    return SUBSPACE_NAMES.get(base, base) + (f", seed {seed}" if seed else ", seed 0")


def subspace_tables():
    """Tables for the subspace-restricted interchange (all seeds found)."""
    files = [p for p in (RES / "planted" / "subspace_interchange.json",
                         RES / "planted" / "subspace_interchange_seeds.json") if p.exists()]
    if not files:
        return []
    sub = {}
    for p in files:
        sub.update(json.loads(p.read_text(encoding="utf-8")))
    out = ["### Interchange restricted to a subspace", "",
           "Source: `results/planted/subspace_interchange*.json` "
           "(`scripts/19_subspace_interchange.py`, run in November 2026; seed 0 is the model "
           "saved in June). Only the part of the residual vector inside a subspace is swapped "
           "between a recipient and a donor on the other side of the threshold; the cells give "
           "the share of pairs whose preferred marker ends on the donor's side. Class-mean "
           "subspace: spanned by the mean activation of each counter value (rank 8 of 256). "
           "Side direction: joins the mean activations of the two sides (rank 1). Probe "
           "subspace: the discriminant directions of a linear probe for the counter (rank 8). "
           "Leftover principal subspace: the eight leading principal directions once the "
           "class-mean and probe subspaces are removed. Random subspaces: mean and maximum over "
           "five. \"All but\" swaps everything except the subspace. Subspaces are fit on "
           "training sequences; swaps are run on held-out sequences, 800 pairs in each "
           "direction.", ""]
    rows, rows2 = [], []
    for cond in sorted(sub, key=lambda c: (c.partition("_seed")[2] or "0", c)):
        for layer, x in sub[cond]["layers"].items():
            rows.append([subspace_name(cond), layer, f(x["flip_no_swap"]), f(x["flip_whole_vector"]),
                         f(x["flip_class_mean_subspace"]), f(x["flip_all_but_class_mean_subspace"]),
                         f(x["flip_side_direction"]), f(x["flip_all_but_side_direction"]),
                         f(x["flip_probe_subspace"]), f(x["flip_all_but_probe_subspace"]),
                         f(x["flip_leftover_principal_subspace"]),
                         f"{x['flip_random_same_rank_mean']:.3f} / {x['flip_random_same_rank_max']:.3f}",
                         f"{x['flip_random_rank1_mean']:.3f} / {x['flip_random_rank1_max']:.3f}"])
            rows2.append([subspace_name(cond), layer, f(x["probe_accuracy_heldout"]),
                          f(x["variance_share_class_mean_subspace"]),
                          f(x["variance_share_side_direction"]),
                          f(x["variance_share_probe_subspace"]),
                          f(x["class_mean_subspace_inside_probe_subspace"]),
                          f(x["same_side_class_mean_subspace_left_side"]),
                          f(x["same_side_whole_vector_left_side"])])
    out += table(["model", "layer", "no swap", "whole vector", "class-mean subspace",
                  "all but the class-mean subspace", "side direction", "all but the side direction",
                  "probe subspace", "all but the probe subspace", "leftover principal subspace",
                  "random, rank 8 (mean / max)", "random, rank 1 (mean / max)"], rows) + [""]
    out += ["Probe accuracy on held-out sequences, the share of activation variance inside each "
            "subspace, how much of the class-mean subspace lies inside the probe subspace (1 "
            "would be all of it), and the same-side check (donor from the recipient's own side; "
            "share of pairs that leave their side):", ""]
    out += table(["model", "layer", "probe accuracy, held-out", "variance share, class-mean subspace",
                  "variance share, side direction", "variance share, probe subspace",
                  "class-mean subspace inside probe subspace", "same side, class-mean subspace swapped",
                  "same side, whole vector swapped"], rows2) + [""]
    return out



SEED_NOTE = ("Source: `results/planted/*_seed*/history.json` and the `_seed` entries of "
             "`results/planted/local_baselines.json`. Seed 0 is the June run. Seeds 1 and 2 were "
             "trained in November 2026 with `scripts/05_run_planted.py --seed N` on a rented RTX "
             "A6000 (torch 2.8.0, transformers 5.12.1; the June runs used torch 2.6.0 and "
             "transformers 5.1.0). Each seed has its own corpus and its own model initialization. "
             "Cells: the model's marker accuracy, then the best local predictor, both against the rule.")


def seed_rows(prefix):
    local = local_baselines()
    rows = []
    for run in planted_runs(prefix):
        cells, found = [], False
        for name in (run, f"{run}_seed1", f"{run}_seed2"):
            if (RES / "planted" / name / "history.json").exists():
                b = local.get(name)
                cells.append(f"{final_eval(name)['acc']:.3f} / " + (f"{b['best_local_vs_rule']:.3f}" if b else "-"))
                found = found or name != run
            else:
                cells.append("-")
        if found:
            rows.append([run.rsplit("_", 1)[1]] + cells)
    return rows



def sec_planted(out, probes):
    out += ["## 1. Planted counter, synthetic carrier", "",
            "Source: `results/planted/rule_p_*/history.json` (final evaluation at step 4,000), "
            "`results/planted/shuffle_control/history.json`, probe lines in "
            "`results/logs/planted_a2_probe_log.txt`, `planted_a4_log.txt`, `planted_a5_log.txt`. "
            "`rule_p` is the probability that a marker follows the hidden rule. A dash means "
            "that run was not probed.", ""]
    rows = planted_rows("", probes)
    s = final_eval("shuffle_control")
    rows.append(["1.0, trained on shuffled sequences", f(s["acc"]), "-", f(s["null"]),
                 f(s["oracle"]), "-", "-", "-"])
    out += table(PLANTED_HEAD, rows) + [""]

    out += ["### Local predictors scored against the rule", "", LOCAL_NOTE, ""]
    out += table(LOCAL_HEAD, local_rows("")) + [""]
    if seed_rows(""):
        out += ["### Three seeds", "", SEED_NOTE, ""]
        out += table(["rule_p", "seed 0", "seed 1", "seed 2"], seed_rows("")) + [""]

    out += ["### Depth probe by layer, rule_p = 1.0", "",
            "Source: `results/logs/planted_a2_probe_log.txt`.", ""]
    rows = []
    for layer in (1, 2, 3, 4):
        p = probes.get(("rule_p_1.0", layer))
        if p:
            rows.append([layer, f(p["acc"]), f(p["ctrl"]), p["mdl_trained"],
                         p["mdl_random"], f(p["ratio"], 2)])
    out += table(["layer", "probe accuracy", "control-task accuracy",
                  "codelength, trained (bits)", "codelength, random-init (bits)",
                  "ratio"], rows) + [""]

    out += ["### Interchange intervention", "",
            "Source: `results/planted/rule_p_1.0/intervention_results.json` and "
            "`results/planted/natural_rule_p_1.0/intervention_results.json`. The residual "
            "stream at the prediction position is replaced by the residual from a context "
            "on the other side of the depth threshold; the table gives the share of marker "
            "predictions that flip.", ""]
    rows = []
    for name, run in (("synthetic", "rule_p_1.0"), ("natural text", "natural_rule_p_1.0")):
        r = J(f"planted/{run}/intervention_results.json")
        for layer in (1, 2, 3, 4):
            x = r[f"layer{layer}"]
            rows.append([name, layer, x["n"], f(x["flip_low_to_high"]),
                         f(x["flip_high_to_low"]), f(x["opposite_side_flip"]),
                         f(x["same_side_flip_control"])])
    out += table(["carrier", "layer", "pairs", "low to high", "high to low",
                  "opposite side, mean", "same-side control"], rows) + [""]

    out += subspace_tables()

    out += ["## 2. Planted counter needed for nothing else (needle mode)", "",
            "Source: `results/planted/needle_rule_p_*/history.json`, probe lines in "
            "`results/logs/planted_a6_log.txt` and `planted_a7_log.txt`. Brackets are emitted "
            "independently of the counter, so only the marker rule rewards tracking it.", ""]
    out += table(PLANTED_HEAD, planted_rows("needle_", probes)) + [""]

    out += ["### Local predictors scored against the rule, needle mode", "", LOCAL_NOTE, ""]
    out += table(LOCAL_HEAD, local_rows("needle_")) + [""]
    if seed_rows("needle_"):
        out += ["### Three seeds, needle mode", "", SEED_NOTE, ""]
        out += table(["rule_p", "seed 0", "seed 1", "seed 2"], seed_rows("needle_")) + [""]

    out += ["## 3. Planted counter in a natural-text carrier", "",
            "Source: `results/planted/natural_rule_p_*/history.json` and "
            "`results/planted/probe_results.json`. Carrier tokens are WikiText-103 words in "
            "their original order.", ""]
    pr = J("planted/probe_results.json")
    local = local_baselines()
    rows = []
    for rp in ("1.0", "0.3", "0.0"):
        e = final_eval(f"natural_rule_p_{rp}")
        p = pr[f"natural_rule_p_{rp}"]["layer4"]
        b = local.get(f"natural_rule_p_{rp}")
        rows.append([rp, f(e["acc"]), f(b["best_local_vs_rule"]) if b else "-", f(e["null"]),
                     f(e["oracle"]), f(p["accuracy"]["real"]), f(p["accuracy"]["control_task"]),
                     f(p["mdl_ratio_rand_over_trained"], 2)])
    out += table(PLANTED_HEAD, rows) + [""]
    if local_rows("natural_"):
        out += ["### Local predictors scored against the rule, natural-text carrier", "",
                LOCAL_NOTE + " In the merged-filler n-gram every word of the carrier is one symbol.", ""]
        out += table(LOCAL_HEAD, local_rows("natural_")) + [""]


def sec_natural(out):
    out += ["## 4. Dependency depth in a small from-scratch model", "",
            "Source: `results/phaseB/results.json`. Thirteen depth classes.", ""]
    r = J("phaseB/results.json")["layers"]
    rows = []
    for k in sorted(r):
        x = r[k]
        rows.append([x["layer"], f(x["accuracy"]["real"]), f(x["accuracy"]["control_task"]),
                     f(x["majority"]), f(x["mdl_trained"]["compression"], 2),
                     f(x["mdl_random_init"]["compression"], 2),
                     f(x["mdl_ratio_rand_over_trained"], 2)])
    out += table(["layer", "probe accuracy", "control-task accuracy", "majority class",
                  "compression, trained", "compression, random-init",
                  "codelength ratio, random-init / trained"], rows) + [""]

    out += ["## 5. Pretrained models, classification probe with codelength", "",
            "Source: `results/phaseB2_*.json`. Compression is the uniform codelength divided "
            "by the probe's online codelength (above 1 means the labels are compressible "
            "from the activations). Where the file does not store the uniform codelength it "
            "is computed here as positions x log2(classes). Models without a random-init "
            "column were loaded in 4-bit and no random-init twin was built.", ""]
    rows = []
    for p in sorted(RES.glob("phaseB2_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        stem = p.stem
        label = ("dependency relation" if stem.endswith("_dep") else
                 "head distance" if stem.endswith("_head_dist") else "depth")
        for k, x in r["layers"].items():
            ncls = x["accuracy"]["n_classes"]
            uni = x.get("uniform_bits", r["n_positions"] * math.log2(ncls))
            rnd = x.get("mdl_random")
            rows.append([r["model"], label, k.replace("layer", ""), ncls,
                         f(x["accuracy"]["real"]), f(x["accuracy"]["control_task"]),
                         f(uni / x["mdl_trained"], 2),
                         f(uni / rnd, 2) if rnd else "-",
                         f(rnd / x["mdl_trained"], 2) if rnd else "-"])
    out += table(["model", "variable", "layer", "classes", "probe accuracy",
                  "control-task accuracy", "compression, trained",
                  "compression, random-init", "ratio, random-init / trained"], rows) + [""]

    out += ["## 6. Structural norm probe", "",
            "Source: `results/structural_probe_*.json`. Norm Spearman is the mean "
            "per-sentence rank correlation between the probe's squared norm and the true "
            "value; root accuracy is the share of test sentences whose root word has the "
            "smallest predicted depth.", ""]
    rows = []
    for p in sorted(RES.glob("structural_probe_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        label = "head distance" if p.stem.endswith("_head_dist") else "depth"
        for k, x in r["layers"].items():
            t, rn = x["trained"], x.get("random_init")
            rows.append([r["model"], label, k.replace("layer", ""),
                         f(t["norm_spearman"]), f(rn["norm_spearman"]) if rn else "-",
                         f(t.get("root_acc")), f(rn.get("root_acc")) if rn else "-",
                         t["n_sentences"]])
    out += table(["model", "variable", "layer", "norm Spearman, trained",
                  "norm Spearman, random-init", "root accuracy, trained",
                  "root accuracy, random-init", "test sentences"], rows) + [""]

    out += ["## 7. Removing the depth subspace", "",
            "Source: `results/causal_depth_*.json`. The row space of the depth probe "
            "(rank 128) is projected out of the residual stream at the middle layer; the "
            "comparison removes one random subspace of the same rank. Loss is mean "
            "next-token cross-entropy on held-out sentences.", ""]
    rows = []
    for name in ("gpt2", "gpt2-large"):
        x = J(f"causal_depth_{name}.json")
        rows.append([x["model"], x["layer"], x["rank"], f(x["depth_nspr"]),
                     f(x["lm_loss_baseline"]), f(x["delta_depth"]), f(x["delta_random"]),
                     f(x["depth_over_random"], 2)])
    out += table(["model", "layer", "rank", "norm Spearman", "baseline loss",
                  "loss increase, depth subspace removed",
                  "loss increase, random subspace removed", "ratio"], rows) + [""]

    out += ["### By depth of the predicted token (punctuation and word-continuation tokens only)", "",
            "Source: `results/causal_depth_buckets_*.json` (`scripts/15_causal_depth_buckets.py`). "
            "Shallow means depth 2 or less, deep means depth 5 or more. The script looks up the "
            "depth at the first character of each token's offset. For the tokenizers used here "
            "the offset of a token that follows a space starts on that space, which has no depth, "
            "so every such token is left out. The buckets therefore hold punctuation marks and "
            "word-continuation tokens only, which is why they are small. Each cell is the loss "
            "before the removal, then the increase.", ""]
    rows = []
    for p in sorted(RES.glob("causal_depth_buckets_*.json")):
        x = json.loads(p.read_text(encoding="utf-8"))
        s, d = x["shallow"], x["deep"]
        rows.append([x["model"], x["layer"], f(x["depth_nspr"]), x["n_shallow"], x["n_deep"],
                     f(s["base"], 2), f"{s['depth_abl'] - s['base']:+.2f}",
                     f"{s['rand_abl'] - s['base']:+.2f}",
                     f(d["base"], 2), f"{d['depth_abl'] - d['base']:+.2f}",
                     f"{d['rand_abl'] - d['base']:+.2f}"])
    out += table(["model", "layer", "norm Spearman", "shallow tokens", "deep tokens",
                  "shallow: loss before", "shallow: depth removed", "shallow: random removed",
                  "deep: loss before", "deep: depth removed", "deep: random removed"],
                 rows) + [""]

    reruns = sorted(RES.glob("causal_depth_by_depth_*.json"))
    if reruns:
        out += ["### Removal with every token covered and matched comparisons", "",
                "Source: `results/causal_depth_by_depth_*.json` (`scripts/15_causal_depth_buckets.py`, "
                "run in November 2026 on a rented RTX A6000; the four GPT-2 models in float32, the "
                "three larger models in bfloat16). The depth of a predicted token is looked up "
                "at the last character of its offset, so every token is covered. Comparison "
                "subspaces of the same rank: the row space of the same probe fit to shuffled depth "
                "labels (three shuffles), and random subspaces (ten). Cells are the increase in "
                "next-token loss; for the comparisons, the mean with the range in brackets.", ""]

        def cell(s, b):
            return f"{s['mean'][b]:+.3f} ({s['min'][b]:+.3f} to {s['max'][b]:+.3f})"
        rows = []
        for p in reruns:
            x = json.loads(p.read_text(encoding="utf-8"))
            for b in ("all", "shallow", "middle", "deep"):
                rows.append([x["model"], x["layer"], f"{x['rank']} of {x['hidden_size']}",
                             f(x["depth_nspr"]), b, x["tokens"][b], f(x["loss_before"][b]),
                             f"{x['increase_depth_subspace'][b]:+.3f}",
                             cell(x["increase_shuffled_probe_subspace"], b),
                             cell(x["increase_random_subspace"], b)])
        out += table(["model", "layer", "rank removed", "norm Spearman", "predicted tokens",
                      "count", "loss before", "depth subspace removed",
                      "shuffled-label probe subspace removed", "random subspace removed"],
                     rows) + [""]
        rows = [[json.loads(p.read_text(encoding="utf-8"))["model"],
                 ", ".join(f"{v:+.3f}" for v in json.loads(p.read_text(encoding="utf-8"))
                           ["shuffled_probe_nspr_on_true_depth"])] for p in reruns]
        out += ["Norm Spearman of the shuffled-label probes against the true depth (they should "
                "not read depth):", ""]
        out += table(["model", "shuffled-label probes"], rows) + [""]


def sec_polarity(out):
    out += ["## 8. Forced-polarity minimal pairs (16 items)", "",
            "Source: `results/polarity_reasoning.json`; the items are in "
            "`scripts/11_polarity_reasoning.py`. Chance is 0.5.", ""]
    r = J("polarity_reasoning.json")
    order = ["clean_tiny", "clean_small", "gpt2", "gpt2-medium", "gpt2-large",
             "Phi-3.5-mini-instruct", "Qwen2.5-7B-Instruct", "Qwen3-8B"]
    rows = [[m, f(r[m]["accuracy"]), f"{round(r[m]['accuracy'] * 16)} of 16",
             f(r[m]["mean_margin"], 2)] for m in order if m in r]
    out += table(["model", "accuracy", "correct", "mean log-probability margin"], rows) + [""]


def sec_modal(out):
    out += ["## 9. Modal perturbation (calibration on a known structure)", ""]
    cats = ["global", "core_modal", "pred_modal", "post_modal", "control"]
    for size, steps in (("tiny", "2,000"), ("small", "3,000")):
        r = J(f"smoke_summary_{size}.json")
        out += [f"### Next-token loss on clean held-out text, `{size}` models", "",
                f"Source: `results/smoke_summary_{size}.json`. One model per training "
                f"variant, {steps} steps. Token counts of the evaluated categories: " +
                ", ".join(f"{c} {r['clean'][c]['tokens']:,}" for c in cats) + ".", ""]
        rows = [[v] + [f(r[v][c]["nll"]) for c in cats] for v in r]
        out += table(["trained on"] + cats, rows) + [""]
    for size in ("tiny", "small"):
        r = J(f"minimal_pairs_results_{size}.json")
        n = r["clean"]["by_contrast"]
        out += [f"### Minimal-pair accuracy, `{size}` models", "",
                f"Source: `results/minimal_pairs_results_{size}.json`. Share of items on "
                f"which the model prefers the original modal to the perturbed one "
                f"(chance 0.5). Items: cross {n['cross']['n']}, polarity "
                f"{n['polarity']['n']}, within {n['within']['n']}.", ""]
        rows = [[v] + [f(r[v]["by_contrast"][c]["accuracy"])
                       for c in ("cross", "polarity", "within")] for v in r]
        out += table(["trained on", "cross-force", "polarity", "within-force"], rows) + [""]
    out += ["### Cross-force accuracy of the clean-trained model, by force pair", "",
            "Source: the `by_cross_axis` entries of the two files above.", ""]
    rows = []
    for size in ("tiny", "small"):
        ax = J(f"minimal_pairs_results_{size}.json")["clean"]["by_cross_axis"]
        for k in ("POSSIBILITY->NECESSITY", "WEAK_NECESSITY->NECESSITY",
                  "NECESSITY->WEAK_NECESSITY"):
            rows.append([size, k.replace("->", " to "), ax[k]["n"], f(ax[k]["accuracy"])])
    out += table(["model size", "force pair", "items", "accuracy"], rows) + [""]

    out += ["### Force probe at the modal token, `small` clean model", "",
            "Source: `results/force_geometry_small.json`. Three force classes.", ""]
    r = J("force_geometry_small.json")
    rows = [[k.replace("layer", ""), f(x["accuracy"]["real"]),
             f(x["accuracy"]["control_task"]), f(x["mdl_ratio"], 2),
             f(x["ordering"]["spearman_proj_vs_ordinal"], 2),
             f(x["ordering"]["distant_confusion"]), f(x["ordering"]["adjacent_confusion"])]
            for k, x in r.items()]
    out += table(["layer", "probe accuracy", "control-task accuracy",
                  "codelength ratio, random-init / trained", "Spearman with force order",
                  "confusion, distant pair", "confusion, adjacent pairs"], rows) + [""]

    out += ["### Force probe with token identity controlled", "",
            "Source: `results/force_geometry_small_v2.json`. The first three columns probe "
            "the position before the modal, where the modal token has not been seen. The "
            "last two compare the ordering statistic in the trained model and in a "
            "random-init model of the same architecture.", ""]
    r = J("force_geometry_small_v2.json")
    rows = [[k.replace("layer", ""), f(x["pred_pos"]["acc"]["real"]),
             f(x["pred_pos"]["acc"]["control_task"]), f(x["pred_pos"]["mdl_ratio"], 2),
             f(x["modal_pos"]["ordered_trained"]["spearman_proj_vs_ordinal"], 2),
             f(x["modal_pos"]["ordered_random"]["spearman_proj_vs_ordinal"], 2)]
            for k, x in r.items()]
    out += table(["layer", "accuracy before the modal", "control-task accuracy",
                  "codelength ratio before the modal", "Spearman, trained",
                  "Spearman, random-init"], rows) + [""]


def sec_sae(out):
    out += ["## 10. Sparse autoencoders", "",
            "### Calibration on the planted model", "",
            "Source: `results/sae_calibrate_rule_p_1.0.json`.", ""]
    c = J("sae_calibrate_rule_p_1.0.json")
    out += table(["quantity", "value"], [
        ["layer", c["layer"]], ["features", c["n_features"]],
        ["mean active features per token (L0)", f(c["l0"], 1)],
        ["fraction of variance unexplained", f(c["fvu"], 4)],
        ["best single feature, absolute correlation with depth",
         f(c["best_feature_depth_corr"])],
        ["features with absolute correlation above 0.3", c["n_features_corr_gt_0.3"]],
        ["Spearman of a linear readout of depth from the sparse code",
         f(c["depth_spearman_from_sae_code"])],
        ["Spearman of a linear readout of depth from raw activations",
         f(c["depth_spearman_from_raw_acts"])]]) + [""]

    out += ["### GPT-2 layer 6 on natural text", "",
            "Source: `results/sae_discover_gpt2_L6.json` (the sparse run) and "
            "`results/logs/d2_v2_log.txt` (the denser run).", ""]
    d = J("sae_discover_gpt2_L6.json")
    rows = [["sparse run (result file)", f(d["stats"]["l0"], 1), f(d["stats"]["fvu"], 4),
             d["stats"]["n_features"], d["taxonomy"]["n_active"],
             d["taxonomy"]["token_detector"], d["taxonomy"]["explained_by_props"],
             d["taxonomy"]["unexplained"]]]
    stats = tax = None
    for line in (RES / "logs" / "d2_v2_log.txt").read_text(encoding="utf-8").splitlines():
        m = SAE_STATS_RE.match(line)
        if m:
            stats = m.groups()
        m = SAE_TAX_RE.match(line)
        if m:
            tax = json.loads(m.group(1).replace("'", '"'))
    if stats and tax:
        rows.append(["denser run (log)", f(float(stats[1]), 1), stats[0], stats[2], stats[3],
                     tax["token_detector"], tax["explained_by_props"], tax["unexplained"]])
    out += table(["run", "L0", "variance unexplained", "features", "active features",
                  "single-token features", "explained by part of speech, depth, position "
                  "or frequency", "neither"], rows) + [""]
    out += ["Top-activating words of the twelve highest-ranked features in the last "
            "column (sparse run):", ""]
    rows = [[x["feature"], x["n_fire"], ", ".join(x["top_words"])]
            for x in d["top_unexplained"]]
    out += table(["feature", "tokens it fires on", "top-activating words"], rows) + [""]


def main():
    probes = probe_lines()
    out = ["# Results tables", "",
           "Generated by `python scripts/make_tables.py` from the files in this folder. "
           "Do not edit by hand.", ""]
    sec_planted(out, probes)
    sec_natural(out)
    sec_polarity(out)
    sec_modal(out)
    sec_sae(out)
    (RES / "TABLES.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {RES / 'TABLES.md'} ({len(out)} lines)")


if __name__ == "__main__":
    main()

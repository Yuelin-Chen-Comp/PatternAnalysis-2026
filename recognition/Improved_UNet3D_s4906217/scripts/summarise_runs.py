"""Collect the results.json files that predict.py wrote for every run and print the tables of the README.

For each model (three seeds each) the per-run results are aggregated as mean +- standard deviation over the seeds, and
the models are compared patient by patient. Only the standard library is needed, so this runs anywhere (on Rangpur
with the system Python, or on a laptop after copying the runs).

    python summarise_runs.py --split test            # the held-out split (after predict_job.sh <run> --split test)
    python summarise_runs.py --split val             # the development split

Expects ~/hipmri_runs/<run>/predict_<split>/results.json for the runs baseline2d, unet3d and improved3d (seed 0) and
the same names with _s1 and _s2 (seeds 1 and 2); runs that are missing are reported and skipped.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

MODELS = {"2D U-Net": "baseline2d", "3D U-Net": "unet3d", "Improved 3D U-Net": "improved3d"}
SEEDS = ("", "_s1", "_s2")
CLASS_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
PROSTATE = 5


def load_runs(out_dir: Path, split: str):
    """{model label: [summary of every available seed]}, plus the list of runs that are missing."""
    runs, missing = {}, []
    for label, prefix in MODELS.items():
        runs[label] = []
        for seed in SEEDS:
            path = out_dir / f"{prefix}{seed}" / f"predict_{split}" / "results.json"
            if path.exists():
                runs[label].append(json.loads(path.read_text()))
            else:
                missing.append(str(path))
    return runs, missing


def clean(values):
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def mean_std(values):
    """(mean, sample std over the values); std is NaN with fewer than two values, everything NaN if empty."""
    v = clean(values)
    if not v:
        return math.nan, math.nan
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else math.nan)


def fmt(values, digits=3):
    m, s = mean_std(values)
    if math.isnan(m):
        return "n/a"
    return f"{m:.{digits}f}" if math.isnan(s) else f"{m:.{digits}f} +- {s:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def summary_table(runs, title, getters, digits=3):
    """One row per model; each getter(summary) gives one number per run, aggregated over the seeds."""
    header = ["Model", "runs"] + [name for name, _ in getters]
    rows = [[label, str(len(res))] + [fmt([g(r["summary"]) for r in res], digits) for _, g in getters]
            for label, res in runs.items() if res]
    return f"**{title}**\n\n{table(header, rows)}\n"


def pk(key):
    return lambda s: s.get(key)


def per_patient_mean(res, key):
    """{patient: mean over the seeds of the patient-level value of `key`}."""
    patients = sorted(res[0]["summary"]["per_patient"])
    return {p: mean_std([r["summary"]["per_patient"][p][key] for r in res])[0] for p in patients}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out-dir", default=str(Path.home() / "hipmri_runs"))
    args = ap.parse_args(argv)

    runs, missing = load_runs(Path(args.out_dir), args.split)
    for m in missing:
        print(f"missing: {m}")
    available = {k: v for k, v in runs.items() if v}
    if not available:
        raise SystemExit(f"no results.json found for the {args.split} split in {args.out_dir}")
    n_volumes = next(iter(available.values()))[0]["summary"]["n_volumes"]
    patients = sorted(next(iter(available.values()))[0]["summary"]["per_patient"])
    print(f"\n## {args.split} split: {n_volumes} volumes, {len(patients)} patients "
          f"({', '.join(patients)}); mean +- std over the seeds of a model (sample std, n = number of runs)\n")

    classes = [(f"{CLASS_NAMES[c]} Dice", (lambda s, c=c: s["dice_mean"][c])) for c in range(len(CLASS_NAMES))]
    print(summary_table(runs, "Per-class Dice (mean over volumes)", classes, 4))
    ious = [(f"{CLASS_NAMES[c]} IoU", (lambda s, c=c: s["iou_mean"][c])) for c in range(1, len(CLASS_NAMES))]
    print(summary_table(runs, "Per-class IoU (mean over volumes)", ious, 4))

    print(summary_table(runs, "Prostate Dice: over volumes, over patients, and by third", [
        ("volumes", lambda s: s["dice_mean"][PROSTATE]),
        ("patients", lambda s: s["patient_level"]["prostate_dice"]["mean"]),
        ("apex third", pk("dice_apex_mean")), ("middle third", pk("dice_mid_mean")), ("base third", pk("dice_base_mean"))], 4))

    print(summary_table(runs, "Where the true prostate goes (share of its voxels; mean over volumes)", [
        ("kept (recall)", pk("recall_mean")), ("-> bladder", pk("miss_to_bladder_mean")),
        ("-> rectum", pk("miss_to_rectum_mean")), ("-> other", pk("miss_to_other_mean")),
        ("base third kept", pk("recall_base_mean")), ("base -> bladder", pk("miss_to_bladder_base_mean")),
        ("apex third kept", pk("recall_apex_mean")), ("apex -> other", pk("miss_to_other_apex_mean"))], 3))

    print(summary_table(runs, "Leakage, reach, shape and speed", [
        ("pred. prostate in true rectum", pk("leak_rectum_mean")), ("in true bladder", pk("leak_bladder_mean")),
        ("reach beyond base, mm", pk("base_error_mm_mean")), ("reach beyond apex, mm", pk("apex_error_mm_mean")),
        ("components", pk("n_components_mean")), ("neighbouring-slice Dice", pk("continuity_pred_mean")),
        ("truth: neighbouring-slice Dice", pk("continuity_gt_mean")), ("pred. / true volume", pk("volume_ratio_mean")),
        ("s per volume", pk("seconds_per_volume_mean")), ("peak GiB", pk("peak_inference_gib"))], 3))

    rows = []
    for p in patients:
        row = [p]
        for label, res in available.items():
            row.append(f"{per_patient_mean(res, 'prostate_dice')[p]:.3f}")
        rows.append(row)
    print(f"**Prostate Dice per patient (mean over the seeds)**\n\n{table(['Patient'] + list(available), rows)}\n")

    labels = list(available)
    pairs = [(a, b) for a, b in (("3D U-Net", "2D U-Net"), ("Improved 3D U-Net", "3D U-Net"),
                                 ("Improved 3D U-Net", "2D U-Net")) if a in available and b in available]
    if pairs:
        out = []
        for key, name in (("prostate_dice", "whole prostate"), ("dice_apex", "apex third"),
                          ("dice_mid", "middle third"), ("dice_base", "base third")):
            for a, b in pairs:
                pa, pb = per_patient_mean(available[a], key), per_patient_mean(available[b], key)
                diffs = [pa[p] - pb[p] for p in patients if not (math.isnan(pa[p]) or math.isnan(pb[p]))]
                better = sum(d > 0 for d in diffs)
                out.append([f"{a} - {b}", name, f"{statistics.mean(diffs):+.3f}", f"{min(diffs):+.3f} to {max(diffs):+.3f}",
                            f"{better} of {len(diffs)}"])
        print("**Paired differences per patient (Dice of the seed-averaged patient values)**\n\n"
              + table(["Comparison", "Measure", "mean difference", "range over patients", "patients where the first is better"], out)
              + "\n")


if __name__ == "__main__":
    main()

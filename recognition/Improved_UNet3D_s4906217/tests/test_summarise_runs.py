"""Tests for scripts/summarise_runs.py: aggregation over seeds, per-patient tables, paired differences, missing runs.

Run from the project folder:  python -m pytest tests     (or: python tests/test_summarise_runs.py)
Uses small hand-made results.json files whose answers can be worked out by hand; standard library only.
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarise_runs as sr  # noqa: E402

PATIENTS = ("A001", "B002")


def results(prostate_by_patient, volume_dice=None, volumes=None):
    """A minimal results.json: only the fields that summarise_runs reads. `volumes` maps a volume name to
    (patient, prostate Dice) and adds the per-volume rows."""
    mean = sum(prostate_by_patient.values()) / len(prostate_by_patient)
    dice = volume_dice if volume_dice is not None else mean
    rows = [{"name": n, "patient": pt, "dice": [0.9, 0.9, 0.9, 0.9, 0.9, d]} for n, (pt, d) in (volumes or {}).items()]
    return {"split": "test", "volumes": rows, "summary": {
        "n_volumes": 4, "dice_mean": [0.99, 0.98, 0.9, 0.9, 0.85, dice], "iou_mean": [0.9] * 6,
        "patient_level": {"prostate_dice": {"mean": mean, "std": 0.0}},
        "per_patient": {p: {"prostate_dice": v, "dice_apex": v - 0.02, "dice_mid": v + 0.02, "dice_base": v - 0.04}
                        for p, v in prostate_by_patient.items()}}}


def write_run(root, name, split, payload):
    path = Path(root) / name / f"predict_{split}"
    path.mkdir(parents=True)
    (path / "results.json").write_text(json.dumps(payload))


def run_main(root, split="test"):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sr.main(["--split", split, "--out-dir", str(root)])
    return out.getvalue()


def make_study(root):
    # 2D: patient A 0.80 / 0.82 / 0.78, patient B 0.70 for all seeds; 3D: patient A 0.90, patient B 0.72 (0.74, 0.70)
    for seed, a2 in zip(("", "_s1", "_s2"), (0.80, 0.82, 0.78)):
        write_run(root, f"baseline2d{seed}", "test", results({"A001": a2, "B002": 0.70}, volume_dice=0.75 + (a2 - 0.80)))
    for seed, b3 in zip(("", "_s1", "_s2"), (0.72, 0.74, 0.70)):
        write_run(root, f"unet3d{seed}", "test", results({"A001": 0.90, "B002": b3}, volume_dice=0.80 + (b3 - 0.72)))


def test_mean_and_sample_std_over_the_seeds():
    mean, std = sr.mean_std([0.8, 0.9, 1.0])
    assert abs(mean - 0.9) < 1e-12 and abs(std - 0.1) < 1e-12
    assert sr.fmt([0.8, 0.9, 1.0], 4) == "0.9000 +- 0.1000"
    assert sr.fmt([0.5], 3) == "0.500", "a single run has no standard deviation"
    assert sr.fmt([None, float("nan")], 3) == "n/a"
    assert sr.fmt([0.5, None, 0.7], 2) == "0.60 +- 0.14", "undefined values are skipped"


def test_the_tables_aggregate_the_seeds_per_model():
    with tempfile.TemporaryDirectory() as tmp:
        make_study(tmp)
        text = run_main(tmp)
        assert "## test split: 4 volumes, 2 patients (A001, B002)" in text
        # prostate Dice over volumes: 2D 0.75, 0.77, 0.73 -> 0.7500 +- 0.0200; 3D 0.80, 0.82, 0.78 -> 0.8000 +- 0.0200
        line2d = next(l for l in text.splitlines() if l.startswith("| 2D U-Net |") and "0.7500 +- 0.0200" in l)
        line3d = next(l for l in text.splitlines() if l.startswith("| 3D U-Net |") and "0.8000 +- 0.0200" in l)
        assert "| 3 |" in line2d and "| 3 |" in line3d
        assert "Improved 3D U-Net" not in text.split("**Per-class Dice")[1].split("**Per-class IoU")[0], "a model without runs gets no row"


def test_missing_runs_are_reported_and_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        make_study(tmp)
        text = run_main(tmp)
        assert text.count("missing:") == 3 and all(f"improved3d{s}" in text for s in ("", "_s1", "_s2"))
        assert "n/a" in text, "measures that the files do not contain show as n/a"
        write_run(tmp, "improved3d", "test", results({"A001": 0.91, "B002": 0.75}))
        assert run_main(tmp).count("missing:") == 2


def test_per_patient_table_and_paired_differences():
    with tempfile.TemporaryDirectory() as tmp:
        make_study(tmp)
        text = run_main(tmp)
        # patient values averaged over the seeds: 2D A 0.80, B 0.70; 3D A 0.90, B 0.72
        assert "| A001 | 0.800 | 0.900 |" in text and "| B002 | 0.700 | 0.720 |" in text
        row = next(l for l in text.splitlines() if l.startswith("| 3D U-Net - 2D U-Net | whole prostate |"))
        assert "+0.060" in row and "+0.020 to +0.100" in row and "2 of 2" in row, row   # mean of +0.10 and +0.02
        base = next(l for l in text.splitlines() if l.startswith("| 3D U-Net - 2D U-Net | base third |"))
        assert "+0.060" in base, "the thirds are compared the same way"


def test_median_seed_and_per_run_table():
    with tempfile.TemporaryDirectory() as tmp:
        make_study(tmp)       # 2D volume-level Dice 0.75, 0.77, 0.73 for seeds 0, 1, 2: the median seed is seed 0
        text = run_main(tmp)
        assert "**Every run: prostate Dice on this split**" in text
        rows = {l.split("|")[2].strip(): l for l in text.splitlines() if l.startswith("| 2D U-Net | baseline2d")}
        assert set(rows) == {"baseline2d", "baseline2d_s1", "baseline2d_s2"}
        assert "median seed" in rows["baseline2d"] and "0.7500" in rows["baseline2d"]
        assert "median seed" not in rows["baseline2d_s1"] and "median seed" not in rows["baseline2d_s2"]
        # 3D: 0.80, 0.82, 0.78 -> the median is again seed 0
        assert any(l.startswith("| 3D U-Net | unet3d |") and "median seed" in l for l in text.splitlines())
    runs = [{"summary": {"dice_mean": [0, 0, 0, 0, 0, d]}, "_run": n} for n, d in (("a", 0.9), ("b", 0.7), ("c", 0.8), ("d", 0.6))]
    assert sr.median_run(runs)["_run"] == "b", "the lower middle run for an even number of seeds"


def test_hardest_volumes_are_ranked_over_all_runs_and_models():
    vols = {"a_week0": ("A001", 0.90), "a_week1": ("A001", 0.60), "b_week0": ("B002", 0.80), "b_week1": ("B002", 0.70)}
    with tempfile.TemporaryDirectory() as tmp:
        for model, shift in (("baseline2d", 0.00), ("unet3d", 0.02), ("improved3d", 0.04)):
            for seed, k in (("", 0.00), ("_s1", 0.01), ("_s2", -0.01)):   # seed 0 is the median seed of every model
                write_run(tmp, f"{model}{seed}", "test",
                          results({"A001": 0.75 + k, "B002": 0.75 + k}, volumes={n: (p, d + shift) for n, (p, d) in vols.items()}))
        text = run_main(tmp)
        assert "hardest on average" in text
        hard = [l for l in text.splitlines() if l.startswith("| a_week1 |") or l.startswith("| b_week1 |")]
        assert len(hard) == 2 and hard[0].startswith("| a_week1 |"), "the lowest Dice comes first"
        # a_week1: 0.60, 0.62, 0.64 for the three models, so 0.620 over all nine runs
        assert "| A001 | 0.620 | 0.600 | 0.620 | 0.640 |" in hard[0], hard[0]
        assert "| B002 | 0.720 | 0.700 | 0.720 | 0.740 |" in hard[1], hard[1]
        # the suggested commands use the median seed of each model (seed 0 here) and the four hardest volumes
        assert "sbatch predict_job.sh baseline2d --split test --volumes a_week1,b_week1,b_week0,a_week0" in text
        assert "sbatch predict_job.sh improved3d --split test --volumes a_week1,b_week1,b_week0,a_week0" in text


def test_hardest_volumes_are_skipped_when_the_files_have_no_volume_rows():
    with tempfile.TemporaryDirectory() as tmp:
        make_study(tmp)       # these hand-made files carry no per-volume rows
        assert "hardest on average" not in run_main(tmp)


def test_no_results_at_all_is_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run_main(tmp)
        except SystemExit as exit_:
            assert "no results.json found" in str(exit_)
        else:
            raise AssertionError("an empty out-dir must stop with a message")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"{len(tests)} passed, 0 skipped")

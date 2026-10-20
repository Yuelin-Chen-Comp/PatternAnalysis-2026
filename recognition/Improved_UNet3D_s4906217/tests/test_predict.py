"""Tests for predict.py: the boundary analysis on hand-made masks, and the whole pipeline on a fake study.

Run from the project folder:  python -m pytest tests     (or: python tests/test_predict.py)
The analysis tests use small boolean volumes whose answers can be worked out by hand. The pipeline test trains a
tiny model for one epoch on the fake study (see test_dataset.make_fake_study) and runs predict.main on it, on the CPU.
Skipped when torch is not installed.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

try:
    import torch
    import dataset
    import predict
    import train
    from test_dataset import make_fake_study
except ImportError:  # pragma: no cover
    torch = None


class Skip(Exception):
    """Raised by a test that cannot run here (used when pytest is not installed)."""


def need(condition=True, why="torch not installed"):
    if condition:
        return
    try:
        import pytest
    except ImportError:
        raise Skip(why)
    pytest.skip(why)


PROSTATE, BLADDER, RECTUM = 5, 3, 4   # dataset.PROSTATE etc.; spelled out so the numpy tests need no torch


def block(shape=(20, 20, 30), z=(10, 22), xy=(6, 14)):
    """A boolean volume with a box of prostate on slices z[0]..z[1]-1: 12 slices, so thirds of 4 slices each."""
    m = np.zeros(shape, bool)
    m[xy[0]:xy[1], xy[0]:xy[1], z[0]:z[1]] = True
    return m


def labels_from(prostate=None, rectum=None, bladder=None, shape=(20, 20, 30)):
    lab = np.zeros(shape, np.uint8)
    for mask, value in ((bladder, BLADDER), (rectum, RECTUM), (prostate, PROSTATE)):
        if mask is not None:
            lab[mask] = value
    return lab


# ---------------------------------------------------------------------------
# the analysis (numpy only)
# ---------------------------------------------------------------------------

def test_extent_and_thirds_follow_the_anatomy():
    need(torch is not None)
    gt = block(z=(10, 22))
    assert predict.prostate_extent(gt) == (10, 21)
    assert predict.prostate_extent(np.zeros_like(gt)) is None
    t = predict.thirds(10, 21)
    assert t == {"apex": (10, 14), "mid": (14, 18), "base": (18, 22)}, "the base is the high-z end"


def test_dice_binary_conventions():
    need(torch is not None)
    a = np.array([1, 1, 0, 0], bool)
    assert predict.dice_binary(a, a) == 1.0
    assert predict.dice_binary(a, ~a) == 0.0
    assert predict.dice_binary(a, np.array([1, 0, 0, 0], bool)) == 2 / 3
    assert predict.dice_binary(np.zeros(4, bool), np.zeros(4, bool)) != predict.dice_binary(np.zeros(4, bool), np.zeros(4, bool))  # NaN


def test_the_third_that_is_missed_gets_the_low_dice():
    need(torch is not None)
    gt = block()
    label = labels_from(prostate=gt)
    spacing = (1.7, 1.7, 1.5)

    pred_no_base = gt.copy()
    pred_no_base[:, :, 18:] = False                                   # the top (base) third is not predicted
    r = predict.analyse_volume(labels_from(prostate=pred_no_base), label, spacing)
    assert (r["dice_apex"], r["dice_mid"], r["dice_base"]) == (1.0, 1.0, 0.0)

    pred_no_apex = gt.copy()
    pred_no_apex[:, :, :14] = False
    r = predict.analyse_volume(labels_from(prostate=pred_no_apex), label, spacing)
    assert (r["dice_apex"], r["dice_mid"], r["dice_base"]) == (0.0, 1.0, 1.0)

    r = predict.analyse_volume(label, label, spacing)                  # a perfect prediction
    assert (r["dice_apex"], r["dice_mid"], r["dice_base"]) == (1.0, 1.0, 1.0)
    assert r["outside_extent_fraction"] == 0.0 and r["base_error_slices"] == 0 and r["apex_error_slices"] == 0


def test_overreach_beyond_the_base_and_apex_has_the_right_sign_and_size():
    need(torch is not None)
    gt = block(z=(10, 22))
    pred = block(z=(8, 25))                                           # 2 slices lower, 3 slices higher
    r = predict.analyse_volume(labels_from(prostate=pred), labels_from(prostate=gt), (1.0, 1.0, 1.5))
    assert r["base_error_slices"] == 3 and r["apex_error_slices"] == 2   # both positive: too far in either direction
    assert r["base_error_mm"] == 4.5 and r["apex_error_mm"] == 3.0
    assert abs(r["outside_extent_fraction"] - 5 / 17) < 1e-9             # 5 of the 17 predicted slices are outside
    short = predict.analyse_volume(labels_from(prostate=block(z=(12, 20))), labels_from(prostate=gt), (1, 1, 1))
    assert short["base_error_slices"] == -2 and short["apex_error_slices"] == -2   # negative: stops short


def test_leakage_into_the_rectum_and_bladder_is_measured():
    need(torch is not None)
    gt = block()
    rectum = np.zeros_like(gt)
    rectum[6:14, 6:14, 10:22] = False
    rectum[6:14, 14:18, 10:22] = True                                  # a rectum right next to the prostate
    bladder = np.zeros_like(gt)
    bladder[6:14, 6:14, 22:26] = True                                  # a bladder right above it
    label = labels_from(prostate=gt, rectum=rectum, bladder=bladder)
    pred = np.zeros_like(gt)
    pred[6:14, 6:18, 10:22] = True                                     # prostate + 4 columns into the rectum
    pred[6:14, 6:14, 22:24] = True                                     # and 2 slices into the bladder
    r = predict.analyse_volume(labels_from(prostate=pred), label, (1, 1, 1))
    total = pred.sum()
    assert abs(r["leak_rectum"] - (8 * 4 * 12) / total) < 1e-9
    assert abs(r["leak_bladder"] - (8 * 8 * 2) / total) < 1e-9
    clean = predict.analyse_volume(labels_from(prostate=gt), label, (1, 1, 1))
    assert clean["leak_rectum"] == 0.0 and clean["leak_bladder"] == 0.0


def test_no_predicted_prostate_gives_nan_shares_not_a_crash():
    need(torch is not None)
    gt = block()
    r = predict.analyse_volume(labels_from(), labels_from(prostate=gt), (1, 1, 1))
    assert r["dice_mid"] == 0.0
    for key in ("leak_rectum", "leak_bladder", "outside_extent_fraction", "base_error_mm", "n_components"):
        assert r[key] != r[key], f"{key} should be NaN when nothing is predicted"


def test_a_disconnected_artifact_is_counted():
    need(torch is not None and predict.ndimage is not None, "scipy not installed")
    gt = block()
    pred = gt.copy()
    pred[1:3, 1:3, 1:3] = True                                         # a separate 8-voxel blob
    r = predict.analyse_volume(labels_from(prostate=pred), labels_from(prostate=gt), (1, 1, 1))
    assert r["n_components"] == 2
    assert abs(r["outside_largest_fraction"] - 8 / pred.sum()) < 1e-9
    assert predict.analyse_volume(labels_from(prostate=gt), labels_from(prostate=gt), (1, 1, 1))["n_components"] == 1


def test_slice_continuity_penalises_a_missing_slice():
    need(torch is not None)
    smooth = block()
    assert predict.adjacent_slice_dice(smooth) == 1.0
    gap = smooth.copy()
    gap[:, :, 15] = False                                              # one empty slice in the middle
    assert predict.adjacent_slice_dice(gap) < 0.9
    assert predict.adjacent_slice_dice(np.zeros_like(smooth)) != predict.adjacent_slice_dice(np.zeros_like(smooth))  # NaN


def test_select_cases_picks_the_worst_and_a_typical_one():
    need(torch is not None)
    rows = [{"name": f"v{i}", "dice": [0, 0, 0, 0, 0, d]} for i, d in enumerate([0.9, 0.2, 0.8, 0.5, float("nan"), 0.85])]
    chosen = predict.select_cases(rows, n_worst=2, n_median=1)
    assert [(k, r["name"]) for k, r in chosen[:2]] == [("worst", "v1"), ("worst", "v3")]
    assert chosen[2][0] == "typical" and chosen[2][1]["name"] in ("v2", "v5"), "the median of 0.8, 0.85, 0.9 is 0.85"
    assert all(r["name"] != "v4" for _, r in chosen), "a volume without a defined Dice is never chosen"


def test_missed_prostate_is_attributed_to_the_class_that_took_it():
    need(torch is not None)
    gt = block()                                                       # prostate on z 10..21: apex 10-13, mid 14-17, base 18-21
    label = labels_from(prostate=gt)
    pred = labels_from(prostate=gt)
    pred[gt & (np.arange(30)[None, None, :] >= 18)] = BLADDER           # the whole base third is called bladder
    mid = gt & (np.arange(30)[None, None, :] >= 14) & (np.arange(30)[None, None, :] < 18)
    half = mid & (np.arange(20)[None, :, None] < 10)                    # y 6..9: 4 of the 8 columns of the block
    pred[half] = RECTUM                                                 # half of the middle third is called rectum
    r = predict.analyse_volume(pred, label, (1, 1, 1))
    assert (r["recall_apex"], r["recall_mid"], r["recall_base"]) == (1.0, 0.5, 0.0)
    assert r["miss_to_bladder_base"] == 1.0 and r["miss_to_rectum_base"] == 0.0
    assert r["miss_to_rectum_mid"] == 0.5 and r["miss_to_bladder_mid"] == 0.0
    assert r["miss_to_bladder_apex"] == 0.0 and r["miss_to_other_apex"] == 0.0
    assert abs(r["recall"] - 0.5) < 1e-9 and abs(r["miss_to_bladder"] - 1 / 3) < 1e-9
    assert abs(r["miss_to_rectum"] - 1 / 6) < 1e-9 and abs(r["miss_to_other"]) < 1e-9
    # the old leakage number does not see this error at all: no PREDICTED prostate lies in the bladder or rectum
    assert r["leak_bladder"] == 0.0 and r["leak_rectum"] == 0.0


def test_where_the_prostate_goes_always_adds_up_to_one():
    need(torch is not None)
    rng = np.random.default_rng(0)
    label = labels_from(prostate=block())
    pred = rng.integers(0, 6, size=label.shape).astype(np.uint8)
    r = predict.analyse_volume(pred, label, (1, 1, 1))
    for suffix in ("", "_apex", "_mid", "_base"):
        total = sum(r[k + suffix] for k in ("recall", "miss_to_bladder", "miss_to_rectum", "miss_to_other"))
        assert abs(total - 1.0) < 1e-9, (suffix, total)
    assert predict.miss_fractions(pred, labels_from())["recall"] != predict.miss_fractions(pred, labels_from())["recall"]  # NaN


def test_a_stray_fragment_does_not_count_as_reach_beyond_the_base():
    need(torch is not None and predict.ndimage is not None, "scipy not installed")
    gt = block(z=(10, 22))
    pred = gt.copy()
    pred[1:3, 1:3, 27:29] = True                                       # a small fragment 6 slices above the prostate
    r = predict.analyse_volume(labels_from(prostate=pred), labels_from(prostate=gt), (1, 1, 1.5))
    assert r["base_error_slices"] == 0 and r["base_error_mm"] == 0.0, "the reach is that of the main blob"
    assert r["n_components"] == 2 and r["outside_extent_fraction"] > 0, "but the fragment is still counted elsewhere"


def test_patient_level_averages_over_patients_not_volumes():
    need(torch is not None)
    rows = [{"patient": "A", "x": 1.0}, {"patient": "A", "x": 1.0}, {"patient": "A", "x": 1.0}, {"patient": "B", "x": 0.0}]
    per_patient, over = predict.patient_level(rows, ["x"])
    assert per_patient == {"A": {"x": 1.0}, "B": {"x": 0.0}}
    assert over["x"] == (0.5, 0.5), "volume mean would be 0.75; patient-level mean is 0.5"
    nan = float("nan")
    _, over_nan = predict.patient_level([{"patient": "A", "x": nan}, {"patient": "B", "x": 0.2}], ["x"])
    assert abs(over_nan["x"][0] - 0.2) < 1e-9, "a patient without a defined value is skipped"


def test_case_planes_use_the_two_ends_and_the_middle_of_the_true_prostate():
    need(torch is not None)
    label = labels_from(prostate=block(z=(10, 22)))                    # prostate on z 10..21
    planes = predict.case_planes(label)
    assert planes["axial"] == [("base end", 20), ("middle", 15), ("apex end", 11)], "one slice inside each end"
    assert planes["extent"] == (10, 21)
    assert planes["centre"] == (10, 10, 16), "the centre of the true prostate (x 6..13, y 6..13, z 10..21), rounded"
    short = predict.case_planes(labels_from(prostate=block(z=(10, 12))))   # only two slices: use the slices themselves
    assert [z for _, z in short["axial"]] == [11, 10, 10], "no inset for a prostate of two slices"
    none = predict.case_planes(labels_from())                           # no prostate in the label: the middle of the volume
    assert none["axial"] == [("middle", 15)] and none["extent"] is None


def test_case_rows_and_cuts_have_the_right_orientation():
    need(torch is not None)
    shape = (20, 24, 30)
    vol = np.arange(np.prod(shape)).reshape(shape)
    windows = (slice(0, 20), slice(0, 24), slice(0, 30))
    axial, origin_a = predict._cut(vol, "axial", 5, windows)
    sagittal, origin_s = predict._cut(vol, "sagittal", 7, windows)
    coronal, origin_c = predict._cut(vol, "coronal", 9, windows)
    assert axial.shape == (24, 20) and origin_a == "upper"            # rows Y (anterior at the top), columns X
    assert sagittal.shape == (30, 24) and origin_s == "lower"         # rows Z (superior at the top), columns Y
    assert coronal.shape == (30, 20) and origin_c == "lower"          # rows Z, columns X
    assert axial[3, 2] == vol[2, 3, 5] and sagittal[4, 3] == vol[7, 3, 4] and coronal[4, 2] == vol[2, 9, 4]
    rows = predict.case_rows(labels_from(prostate=block(z=(10, 22))))
    assert [kind for _, kind, _ in rows] == ["axial", "axial", "axial", "sagittal", "coronal"]


def test_panels_mark_the_errors_of_the_bladder_and_the_rectum_too():
    need(torch is not None and predict._pyplot() is not None, "matplotlib not installed")
    from matplotlib.colors import to_rgb
    gt = np.zeros((6, 6), np.uint8)
    gt[1:3, 1:3], gt[3:5, 1:3], gt[1:3, 4:6] = BLADDER, RECTUM, PROSTATE
    pr = gt.copy()
    pr[1, 1] = 0                                                       # a bladder voxel missed
    pr[3, 1] = 0                                                       # a rectum voxel missed (only the rectum is wrong here)
    pr[4, 2] = BLADDER                                                 # a rectum voxel called bladder (both organs are wrong)
    *_, other = predict.case_panels(np.zeros((6, 6)), gt, pr)
    assert tuple(other[1, 1, :3]) == to_rgb(predict.COLOURS["bladder"]) and other[1, 1, 3] > 0
    assert tuple(other[3, 1, :3]) == to_rgb(predict.COLOURS["rectum"]) and other[3, 1, 3] > 0, "a rectum-only error is marked"
    assert other[4, 2, 3] > 0, "a voxel that changed from rectum to bladder is marked"
    assert other[2, 2, 3] == 0 and other[1, 4, 3] == 0, "correct voxels and the prostate are not marked"
    assert len({predict.COLOURS[k] for k in ("bladder", "rectum", "prostate", "tp", "fn", "fp", "leak")}) == 7


def test_contour_figure_is_written_even_without_prostate_or_prediction():
    need(torch is not None and predict._pyplot() is not None, "matplotlib not installed")
    with tempfile.TemporaryDirectory() as tmp:
        shape = (24, 24, 20)
        rng = np.random.default_rng(0)
        label = labels_from(prostate=block(shape=shape, z=(6, 14), xy=(8, 14)), shape=shape)
        vol = dataset.Volume("v", "P", rng.normal(size=shape).astype(np.float16), label, (1.7, 1.7, 1.5))
        for pred in (label.copy(), np.zeros_like(label)):               # a perfect prediction and an empty one
            path = Path(tmp) / "c.png"
            assert predict.plot_contours(path, vol, pred, "t") and path.stat().st_size > 1000
            assert predict.plot_case(Path(tmp) / "o.png", vol, pred, "t") and (Path(tmp) / "o.png").stat().st_size > 1000
        empty = dataset.Volume("e", "P", vol.image, np.zeros_like(label), vol.spacing)   # no prostate in the label at all
        assert predict.plot_contours(Path(tmp) / "e.png", empty, np.zeros_like(label), "t")


def test_volumes_option_draws_exactly_the_named_volumes():
    need(torch is not None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_fake_study(tmp / "study", tmp / "cache")
        common = ["--root", str(tmp / "study"), "--cache-dir", str(tmp / "cache"),
                  "--split-file", str(tmp / "split.json"), "--out-dir", str(tmp / "runs")]
        train.main(["--model", "unet3d", "--name", "t", "--epochs", "1", "--val-every", "1", "--samples-per-epoch", "4",
                    "--batch-size", "2", "--patch", "16", "16", "8", "--base-channels", "4", "--depth", "2",
                    "--workers", "0", *common])
        names = [v["name"] for v in predict.main(["--name", "t", "--split", "test", "--no-figures", *common])["volumes"]]
        chosen = names[1]
        result = predict.main(["--name", "t", "--split", "test", "--volumes", chosen, *common])
        figs = [f for f in result["figures"] if f != "summary.png"]
        assert sorted(figs) == sorted([f"selected_{chosen}.png", f"selected_{chosen}_contours.png"]), figs
        assert [c["name"] for c in result["cases"]] == [chosen] and result["cases"][0]["kind"] == "selected"
        try:
            predict.main(["--name", "t", "--split", "test", "--volumes", "no_such_volume", *common])
        except SystemExit as exit_:
            assert "no_such_volume" in str(exit_) and chosen in str(exit_), "the error lists the volumes that exist"
        else:
            raise AssertionError("an unknown volume name must stop with a message")


# ---------------------------------------------------------------------------
# the whole pipeline on a fake study
# ---------------------------------------------------------------------------

def test_predict_runs_end_to_end_for_all_models():
    need(torch is not None)
    for model in ("unet2d", "unet3d", "improved3d"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_fake_study(tmp / "study", tmp / "cache")
            common = ["--root", str(tmp / "study"), "--cache-dir", str(tmp / "cache"),
                      "--split-file", str(tmp / "split.json"), "--out-dir", str(tmp / "runs")]
            train.main(["--model", model, "--name", "t", "--epochs", "1", "--val-every", "1",
                        "--samples-per-epoch", "4", "--batch-size", "2", "--patch", "16", "16", "8",
                        "--base-channels", "4", "--depth", "2", "--workers", "0", *common])

            result = predict.main(["--name", "t", "--worst", "2", "--median", "1", *common])
            out = tmp / "runs" / "t" / "predict_val"
            saved = json.loads((out / "results.json").read_text())
            assert saved["split"] == "val" and saved["model"] == model
            assert len(saved["volumes"]) == 2 and len(result["volumes"]) == 2
            split = json.loads((tmp / "split.json").read_text())
            assert {v["patient"] for v in saved["volumes"]} == set(split["val"])
            assert len(saved["volumes"][0]["dice"]) == 6
            assert set(saved["summary"]["per_patient"]) == set(split["val"])
            assert "prostate_dice" in saved["summary"]["patient_level"] and "miss_to_bladder_base_mean" in saved["summary"]
            for key in ("dice_apex", "leak_rectum", "continuity_pred"):
                assert key in saved["volumes"][0]
            try:
                import matplotlib  # noqa: F401
                have_plots = True
            except ImportError:
                have_plots = False
            if have_plots:
                assert (out / "summary.png").exists()
                case_figs = [f for f in saved["figures"] if f != "summary.png"]
                assert len(case_figs) == 4, case_figs    # both volumes (2 worst, none left over): overview + contours each
                assert sum(f.endswith("_contours.png") for f in case_figs) == 2
                for f in saved["figures"]:
                    assert (out / f).stat().st_size > 1000

            predict.main(["--name", "t", "--split", "test", "--no-figures", *common])
            test_result = json.loads((tmp / "runs" / "t" / "predict_test" / "results.json").read_text())
            assert {v["patient"] for v in test_result["volumes"]} == set(split["test"])
            assert test_result["figures"] == []


def test_summarise_runs_reads_what_predict_writes():
    """The run-summary script must understand the results.json files of the real predict.py, for all three models."""
    need(torch is not None)
    sys.path.insert(0, str(HERE.parent / "scripts"))
    import contextlib
    import io
    import summarise_runs
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_fake_study(tmp / "study", tmp / "cache")
        common = ["--root", str(tmp / "study"), "--cache-dir", str(tmp / "cache"),
                  "--split-file", str(tmp / "split.json"), "--out-dir", str(tmp / "runs")]
        for model, name in (("unet2d", "baseline2d"), ("unet3d", "unet3d"), ("improved3d", "improved3d")):
            train.main(["--model", model, "--name", name, "--epochs", "1", "--val-every", "1", "--samples-per-epoch", "4",
                        "--batch-size", "2", "--patch", "16", "16", "8", "--base-channels", "4", "--depth", "2",
                        "--workers", "0", *common])
            predict.main(["--name", name, "--split", "test", "--no-figures", *common])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summarise_runs.main(["--split", "test", "--out-dir", str(tmp / "runs")])
        text = out.getvalue()
        assert text.count("missing:") == 6, "seeds 1 and 2 of the three models were not run"
        for label in ("2D U-Net", "3D U-Net", "Improved 3D U-Net"):
            assert f"| {label} | 1 |" in text, f"no row for {label}"
        assert "Paired differences per patient" in text and "1 of 1" in text or "0 of 1" in text
        assert "nan" not in text.lower().replace("n/a", ""), "undefined values must be shown as n/a, not nan"


def test_predict_agrees_with_the_scores_of_train_test():
    """predict.py and train.py --test-only must report the same per-class Dice for the same checkpoint."""
    need(torch is not None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_fake_study(tmp / "study", tmp / "cache")
        common = ["--root", str(tmp / "study"), "--cache-dir", str(tmp / "cache"),
                  "--split-file", str(tmp / "split.json"), "--out-dir", str(tmp / "runs")]
        train.main(["--model", "unet3d", "--name", "t", "--epochs", "1", "--val-every", "1", "--samples-per-epoch", "4",
                    "--batch-size", "2", "--patch", "16", "16", "8", "--base-channels", "4", "--depth", "2",
                    "--workers", "0", *common])
        tested = train.main(["--model", "unet3d", "--name", "t", "--test-only", *common])
        predicted = predict.main(["--name", "t", "--split", "test", "--no-figures", *common])
        a, b = np.array(tested["summary"]["dice_mean"], float), np.array(predicted["summary"]["dice_mean"], float)
        assert np.allclose(a, b, equal_nan=True, atol=1e-5), (a, b)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    skipped = 0
    for name, fn in tests:
        try:
            fn()
        except Skip as why:
            skipped += 1
            print(f"skip {name} ({why})")
        else:
            print(f"ok   {name}")
    print(f"{len(tests) - skipped} passed, {skipped} skipped")

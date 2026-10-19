"""Tests for train.py: the whole train / validate / test pipeline on a tiny fake study, and a learning check.

Run from the project folder:  python -m pytest tests     (or: python tests/test_train.py)
Everything runs on the CPU in a few seconds with tiny models and a fake study (empty NIfTI files plus a
ready-made cache, see test_dataset.make_fake_study); skipped when torch is not installed.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

try:
    import torch
    import dataset
    import modules
    import train
    import utils
    from test_dataset import make_fake_study
except ImportError:  # pragma: no cover
    torch = None


class Skip(Exception):
    """Raised by a test that cannot run here (used when pytest is not installed)."""


def need_torch():
    if torch is not None:
        return
    try:
        import pytest
    except ImportError:
        raise Skip("torch not installed")
    pytest.skip("torch not installed")


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def base_argv(tmp, model, *extra):
    """A 2-epoch run of a tiny model on the fake study (8 patients -> 6 train / 1 val / 1 test)."""
    tmp = Path(tmp)
    return ["--model", model, "--name", "t", "--root", str(tmp / "study"), "--cache-dir", str(tmp / "cache"),
            "--split-file", str(tmp / "split.json"), "--out-dir", str(tmp / "runs"),
            "--epochs", "2", "--val-every", "1", "--samples-per-epoch", "4", "--batch-size", "2",
            "--patch", "16", "16", "8", "--base-channels", "4", "--depth", "2", "--workers", "0",
            *extra]


def test_pipeline_runs_end_to_end_for_both_models_and_writes_its_outputs():
    need_torch()
    for model in ("unet2d", "unet3d", "improved3d"):
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_study(Path(tmp) / "study", Path(tmp) / "cache")
            run = Path(tmp) / "runs" / "t"

            summary = train.main(base_argv(tmp, model))
            for name in ("config.json", "history.json", "summary.json", "best.pt", "last.pt"):
                assert (run / name).exists(), f"{model}: {name} was not written"
            assert not (run / "test_metrics.json").exists(), "the test split must not be touched without --test"
            history = json.loads((run / "history.json").read_text())
            assert [h["epoch"] for h in history] == [1, 2]
            assert all(h["train_loss"] == h["train_loss"] and h["train_loss"] > 0 for h in history)
            assert len(history[-1]["val_dice"]) == 6
            assert all(h["train_seconds"] > 0 and h["val_seconds"] > 0 for h in history), "per-phase timings are logged"
            config = json.loads((run / "config.json").read_text())
            assert (config["train_patients"], config["train_volumes"], config["val_volumes"]) == (6, 12, 2)
            # depth 2 gives the improved model one auxiliary output, so the default weights are cut to one
            assert config["args"]["aux_weights"] == ([0.5] if model == "improved3d" else [])
            assert summary["best_epoch"] in (1, 2)
            best = torch.load(run / "best.pt", map_location="cpu")
            assert best["epoch"] == summary["best_epoch"]
            assert best["val_prostate_dice"] == max(h["val_prostate_dice"] for h in history), \
                "best.pt must be the epoch with the highest validation prostate Dice"

            tested = train.main(base_argv(tmp, model, "--test-only"))
            metrics = json.loads((run / "test_metrics.json").read_text())
            assert metrics["summary"]["n_volumes"] == 2 and len(metrics["volumes"]) == 2
            assert len(metrics["summary"]["dice_mean"]) == 6 and len(metrics["summary"]["iou_mean"]) == 6
            assert tested["checkpoint_epoch"] == metrics["checkpoint_epoch"]
            # the test volumes belong to patients that are neither in the training nor in the validation split
            split = json.loads((Path(tmp) / "split.json").read_text())
            assert {v["patient"] for v in metrics["volumes"]} == set(split["test"])


def test_training_never_loads_the_test_volumes():
    """Without --test the test split must not even be read: delete its cached volumes, training must still run."""
    need_torch()
    with tempfile.TemporaryDirectory() as tmp:
        root, cache = Path(tmp) / "study", Path(tmp) / "cache"
        make_fake_study(root, cache)
        cases = dataset.find_cases(root)
        split = dataset.load_or_create_split(cases, Path(tmp) / "split.json")
        removed = 0
        for case in dataset.cases_for(cases, split, "test"):
            dataset._cache_path(case, cache).unlink()   # reading it now would fall through to the empty .nii.gz
            removed += 1
        assert removed == 2
        train.main(base_argv(tmp, "unet3d"))            # no --test: must succeed without the test volumes


def test_test_flag_scores_the_best_model_after_training():
    need_torch()
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_study(Path(tmp) / "study", Path(tmp) / "cache")
        summary = train.main(base_argv(tmp, "unet3d", "--test"))
        assert summary["test"]["n_volumes"] == 2
        assert (Path(tmp) / "runs" / "t" / "test_metrics.json").exists()


class StubModel:
    """Returns fixed score maps so that the weighting of forward_loss can be checked by hand."""
    aux_strides = [2, 4]

    def __init__(self):
        self.calls = []

    def __call__(self, image, return_aux=False):
        self.calls.append(return_aux)
        final = torch.full((1, 3, 8, 8), 1.0)
        if not return_aux:
            return final
        return final, [torch.full((1, 3, 4, 4), 2.0), torch.full((1, 3, 2, 2), 4.0)]


def test_forward_loss_weights_the_auxiliary_outputs_and_subsamples_the_labels():
    need_torch()
    seen = []

    def loss_fn(scores, label):
        seen.append((tuple(scores.shape[2:]), tuple(label.shape[1:])))
        return scores.float().mean()

    model = StubModel()
    label = torch.zeros(1, 8, 8, dtype=torch.long)
    total = train.forward_loss(model, torch.zeros(1, 1, 8, 8), label, loss_fn, aux_weights=(0.5, 0.25))
    assert abs(float(total) - (1.0 + 0.5 * 2.0 + 0.25 * 4.0) / 1.75) < 1e-6, "weighted sum divided by 1 + sum of the weights"
    assert seen == [((8, 8), (8, 8)), ((4, 4), (4, 4)), ((2, 2), (2, 2))], "labels are subsampled to the map resolution"
    plain = StubModel()
    assert float(train.forward_loss(plain, torch.zeros(1, 1, 8, 8), label, loss_fn)) == 1.0 and plain.calls == [False]
    assert raises(ValueError, train.forward_loss, StubModel(), torch.zeros(1, 1, 8, 8), label, loss_fn, (1.0, 1.0, 1.0))


def test_aux_weights_flag_is_validated_and_can_be_switched_off():
    need_torch()
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_study(Path(tmp) / "study", Path(tmp) / "cache")
        assert raises(SystemExit, train.main, base_argv(tmp, "unet3d", "--aux-weights", "0.5"))
        off = train.main(base_argv(tmp, "improved3d", "--aux-weights", "--depth", "3"))
        config = json.loads((Path(tmp) / "runs" / "t" / "config.json").read_text())
        assert config["args"]["aux_weights"] == [] and off["best_epoch"] in (1, 2)


def toy_segmentation(n=8, size=32):
    """Images whose bright blob is the label: something a small network can learn in a few steps."""
    torch.manual_seed(0)
    image = torch.randn(n, 1, size, size) * 0.2
    label = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        x, y = torch.randint(4, size - 12, (2,)).tolist()
        label[i, x:x + 8, y:y + 8] = 1 + i % 2
        image[i, 0, x:x + 8, y:y + 8] += 1.0 + 1.0 * (i % 2)
    return image, label


def test_improved_unet_with_deep_supervision_learns_a_toy_segmentation():
    need_torch()
    image, label = toy_segmentation()
    torch.manual_seed(0)
    model = modules.ImprovedUNet(dim=2, num_classes=3, base_channels=4, depth=3)
    loss_fn = utils.DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(80):
        optimizer.zero_grad()
        loss = train.forward_loss(model.train(), image, label, loss_fn, aux_weights=(0.5, 0.25))
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if first is None else first
    final = float(loss.detach())
    assert final < 0.4 * first, f"loss only went from {first:.3f} to {final:.3f}"
    model.eval()
    with torch.no_grad():
        dice, _ = utils.per_class_scores(model(image).argmax(1), label, 3)
    assert float(dice[1:].min()) > 0.7, f"Dice after training: {dice.tolist()}"


def test_summarise_skips_absent_classes_and_averages_the_rest():
    need_torch()
    nan = float("nan")
    rows = [{"dice": [1.0, 0.5, nan], "iou": [1.0, 0.3, nan], "seconds": 1.0},
            {"dice": [0.5, None, nan], "iou": [0.5, None, nan], "seconds": 3.0}]
    out = train.summarise(rows)
    assert out["dice_mean"][:2] == [0.75, 0.5] and out["dice_mean"][2] != out["dice_mean"][2]
    assert out["seconds_per_volume_mean"] == 2.0 and out["n_volumes"] == 2


def test_tiny_unet_with_the_dice_ce_loss_learns_a_toy_segmentation():
    """The model and the loss together must be able to learn: an image whose bright blob is the label."""
    need_torch()
    torch.manual_seed(0)
    n, size = 8, 32
    image = torch.randn(n, 1, size, size) * 0.2
    label = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        x, y = torch.randint(4, size - 12, (2,)).tolist()
        label[i, x:x + 8, y:y + 8] = 1 + i % 2
        image[i, 0, x:x + 8, y:y + 8] += 1.0 + 1.0 * (i % 2)
    model = modules.UNet2D(num_classes=3, base_channels=4, depth=2)
    loss_fn = utils.DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(image), label)
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if first is None else first
    final = float(loss.detach())
    assert final < 0.4 * first, f"loss only went from {first:.3f} to {final:.3f}"
    model.eval()
    with torch.no_grad():
        dice, _ = utils.per_class_scores(model(image).argmax(1), label, 3)
    assert float(dice[1:].min()) > 0.7, f"Dice after training: {dice.tolist()}"


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

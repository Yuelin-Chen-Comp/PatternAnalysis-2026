"""Tests for dataset.py: pairing, patient-level splits, preprocessing, patch sampling, augmentation.

Run from the project folder:  python -m pytest tests     (or: python tests/test_dataset.py)
Everything uses synthetic data, so no dataset is needed. The torch wrapper tests are
skipped when torch is not installed.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
from dataset import (BLADDER, FG_CLASSES, PROSTATE, RECTUM, Case, LeakageError, Volume,  # noqa: E402
                     assert_no_leakage, augment, cases_for, crop_slices, find_cases,
                     find_foreground, load_or_create_split, load_volume, normalise_mr,
                     pair_key, patient_id, sample_patch, split_patients, to_label_array)

HAS_TORCH = dataset.torch is not None


class Skip(Exception):
    """Raised by a test that cannot run here (used when pytest is not installed)."""


def need_torch():
    if HAS_TORCH:
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


# ---------------------------------------------------------------------------
# pairing and splits
# ---------------------------------------------------------------------------

def make_dataset(root: Path, volumes, drop_label=None):
    """Create empty MR/label files named like the real study."""
    (root / dataset.MR_DIR).mkdir(parents=True)
    (root / dataset.LABEL_DIR).mkdir(parents=True)
    for patient, week in volumes:
        (root / dataset.MR_DIR / f"{patient}_Week{week}_LFOV.nii.gz").touch()
        if (patient, week) != drop_label:
            (root / dataset.LABEL_DIR / f"{patient}_Week{week}_SEMANTIC.nii.gz").touch()


def fake_cases(n_patients=38, volumes_per_patient=(1, 8)):
    """n patients with a varying number of weekly volumes."""
    cases = []
    for i in range(n_patients):
        pid = f"B{i:03d}"
        n_vol = volumes_per_patient[0] + i % (volumes_per_patient[1] - volumes_per_patient[0] + 1)
        for w in range(n_vol):
            cases.append(Case(f"{pid.lower()}_week{w}", pid, Path("mr"), Path("lab")))
    return cases


def test_patient_id():
    assert patient_id("B006_Week0_LFOV.nii.gz") == "B006"
    assert patient_id("b006_Week1_SEMANTIC.nii.gz") == "B006"
    assert patient_id("J005_Week0_LFOV.nii.gz") == "J005"
    for bad in ("case_004_week_0_slice_0.nii.gz", "volume.nii.gz", ""):
        assert raises(ValueError, patient_id, bad), f"{bad!r} should not parse"


def test_pair_key_matches_mr_and_label():
    assert pair_key("B006_Week0_LFOV.nii.gz") == pair_key("B006_Week0_SEMANTIC.nii.gz") == "b006_week0"
    assert pair_key("B006_Week0_LFOV.nii") == "b006_week0"
    assert pair_key("B006_Week1_LFOV.nii.gz") != pair_key("B006_Week0_LFOV.nii.gz")


def test_find_cases_pairs_everything():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(Path(tmp), [("B006", 0), ("B006", 1), ("C032", 0)])
        cases = find_cases(tmp)
        assert [c.name for c in cases] == ["b006_week0", "b006_week1", "c032_week0"]
        assert [c.patient for c in cases] == ["B006", "B006", "C032"]
        assert all(c.mr.name.endswith("_LFOV.nii.gz") and c.label.name.endswith("_SEMANTIC.nii.gz")
                   for c in cases)


def test_find_cases_rejects_unpaired_files():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(Path(tmp), [("B006", 0), ("C032", 0)], drop_label=("C032", 0))
        try:
            find_cases(tmp)
        except ValueError as e:
            assert "without a label" in str(e)
        else:
            raise AssertionError("an MR without a label must raise")


def test_split_sizes_and_disjoint():
    cases = fake_cases(38)
    split = split_patients([c.patient for c in cases])
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (26, 6, 6)
    assert_no_leakage(split, cases)


def test_split_is_deterministic_and_seed_dependent():
    patients = [c.patient for c in fake_cases(38)]
    assert split_patients(patients, seed=1) == split_patients(patients, seed=1)
    assert split_patients(patients, seed=1) != split_patients(patients, seed=2)


def test_all_volumes_of_a_patient_stay_together():
    cases = fake_cases(38)
    split = split_patients([c.patient for c in cases])
    total = 0
    for name in ("train", "val", "test"):
        subset = cases_for(cases, split, name)
        assert {c.patient for c in subset} == set(split[name])
        total += len(subset)
    assert total == len(cases)


def test_leakage_is_detected():
    cases = fake_cases(10)
    split = split_patients([c.patient for c in cases])
    split["val"].append(split["train"][0])
    assert raises(LeakageError, assert_no_leakage, split, cases)


def test_incomplete_split_is_detected():
    cases = fake_cases(10)
    split = split_patients([c.patient for c in cases])
    split["train"].pop()
    assert raises(LeakageError, assert_no_leakage, split, cases)


def test_split_file_roundtrip_and_stale_detection():
    cases = fake_cases(38)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits" / "split.json"
        first = load_or_create_split(cases, path, seed=7)
        assert path.exists()
        # second call reads the file; a different seed must not change it
        assert load_or_create_split(cases, path, seed=99) == first
        # the data changed: the stored split must be rejected, not silently rebuilt
        victim = first["train"][0]
        without_patient = [c for c in cases if c.patient != victim]
        with_new_patient = cases + [Case("z999_week0", "Z999", Path("mr"), Path("lab"))]
        for changed in (without_patient, with_new_patient):
            assert raises(LeakageError, load_or_create_split, changed, path), "a stale split file must raise"


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def test_normalise_mr_zscores_the_nonzero_voxels():
    rng = np.random.default_rng(0)
    x = np.zeros((20, 20, 10), dtype=np.float32)
    x[5:15, 5:15, 2:8] = rng.uniform(50, 250, size=(10, 10, 6))
    y = normalise_mr(x)
    inside = x > 0
    assert abs(float(y[inside].mean())) < 1e-3
    assert abs(float(y[inside].std()) - 1.0) < 1e-2
    assert len(np.unique(y[~inside])) == 1, "the empty background must map to a single constant"


def test_normalise_mr_clips_outliers():
    rng = np.random.default_rng(0)
    x = np.zeros((20, 20, 10), dtype=np.float32)
    x[2:18, 2:18, 2:8] = rng.uniform(50, 250, size=(16, 16, 6))   # 1536 ordinary voxels
    x[5, 5, 2:6] = 1e6                                            # 4 saturated voxels (0.26%)
    y = normalise_mr(x)
    # clipped, the outliers sit at the top of the ordinary range (z about 1.7); unclipped they
    # would drag the mean and std and sit near z = 20
    assert float(y[x > 0].max()) < 3, "saturated voxels must be clipped before the z-score"


def test_normalise_mr_all_zero_is_safe():
    y = normalise_mr(np.zeros((8, 8, 4)))
    assert np.isfinite(y).all() and float(np.abs(y).max()) == 0.0


def test_to_label_array_validates_values():
    assert to_label_array(np.array([0.0, 1.0, 5.0])).dtype == np.uint8
    assert raises(ValueError, to_label_array, np.array([0, 6]))
    assert raises(ValueError, to_label_array, np.array([-1, 2]))


def make_volume(name="b001_week0", patient="B001", shape=(32, 32, 16), seed=0):
    rng = np.random.default_rng(seed)
    image = rng.uniform(1, 100, size=shape).astype(np.float16)
    label = np.zeros(shape, dtype=np.uint8)
    label[4:28, 4:28, 2:14] = 1
    label[10:14, 10:14, 4:8] = BLADDER
    label[16:18, 16:18, 6:9] = RECTUM
    label[20:22, 12:14, 6:9] = PROSTATE
    return Volume(name, patient, image, label, (1.68, 1.68, 1.56),
                  find_foreground(label, FG_CLASSES, name))


def test_find_foreground_returns_voxels_of_that_class():
    vol = make_volume()
    for c in FG_CLASSES:
        coords = vol.fg_coords[c]
        assert len(coords) > 0
        assert (vol.label[coords[:, 0], coords[:, 1], coords[:, 2]] == c).all()
    small = find_foreground(vol.label, [1], "x", max_points=50)
    assert len(small[1]) == 50, "large classes are subsampled to max_points"


def test_load_volume_cache_roundtrip_and_shape_check():
    vol = make_volume()
    raw_image = np.asarray(vol.image, dtype=np.float32)
    case = Case("b001_week0", "B001", Path("mr.nii.gz"), Path("lab.nii.gz"))
    real_loader = dataset.load_nifti
    try:
        dataset.load_nifti = lambda p: ((raw_image, (1.68, 1.68, 1.56)) if str(p) == "mr.nii.gz"
                                        else (vol.label.astype(np.float32), (1.68, 1.68, 1.56)))
        with tempfile.TemporaryDirectory() as tmp:
            first = load_volume(case, cache_dir=tmp)
            assert any(Path(tmp).glob("*.npz"))
            assert not any(Path(tmp).glob("*.tmp.npz")), "no half-written cache file may be left"

            def must_not_read(p):
                raise AssertionError("the second load must come from the cache")
            dataset.load_nifti = must_not_read
            second = load_volume(case, cache_dir=tmp)
        assert np.array_equal(first.image, second.image) and np.array_equal(first.label, second.label)
        assert first.spacing == second.spacing == (1.68, 1.68, 1.56)
        assert first.image.dtype == np.float16 and first.label.dtype == np.uint8

        dataset.load_nifti = lambda p: ((raw_image, (1, 1, 1)) if str(p) == "mr.nii.gz"
                                        else (vol.label[:, :, :-1], (1, 1, 1)))
        assert raises(ValueError, load_volume, case), "image/label shape mismatch must raise"
    finally:
        dataset.load_nifti = real_loader


# ---------------------------------------------------------------------------
# patch sampling and augmentation
# ---------------------------------------------------------------------------

def test_crop_slices_stay_inside_and_have_requested_size():
    rng = np.random.default_rng(0)
    shape, patch = (32, 32, 16), (8, 8, 8)
    centers = [None, (0, 0, 0), (31, 31, 15), (16, 16, 8), (5, 30, 2)]
    for center in centers:
        for _ in range(50):
            box = crop_slices(shape, patch, rng, center)
            for sl, dim, size in zip(box, shape, patch):
                assert sl.stop - sl.start == size and 0 <= sl.start and sl.stop <= dim


def test_crop_slices_rejects_a_patch_larger_than_the_volume():
    assert raises(ValueError, crop_slices, (8, 8, 8), (16, 8, 8), np.random.default_rng(0))


def test_sample_patch_shapes_and_dtypes():
    vol = make_volume()
    image, label = sample_patch(vol, (8, 8, 8), np.random.default_rng(0))
    assert image.shape == label.shape == (8, 8, 8)
    assert image.dtype == np.float32 and label.dtype == np.uint8


def test_sample_patch_oversamples_the_rare_organ():
    vol = make_volume()
    rng = np.random.default_rng(1)

    def hit_rate(p):
        hits = sum(bool((sample_patch(vol, (8, 8, 8), rng, p, fg_classes=(PROSTATE,))[1] == PROSTATE).any())
                   for _ in range(300))
        return hits / 300

    assert hit_rate(1.0) == 1.0, "with p_foreground=1 every patch must contain the prostate"
    assert hit_rate(0.0) < 0.6, "uniform patches should often miss a 12-voxel organ"


def test_augment_keeps_image_and_label_aligned_under_a_flip():
    rng = np.random.default_rng(0)
    label = (np.arange(8 * 8 * 8).reshape(8, 8, 8) % 6).astype(np.uint8)
    image = label.astype(np.float32)
    out_image, out_label = augment(image, label, rng, flip_axis=0, p_flip=1.0, p_intensity=0.0, p_noise=0.0)
    assert np.array_equal(out_label, label[::-1])
    assert np.array_equal(out_image, out_label.astype(np.float32)), "image and label must flip together"


def test_augment_never_flips_without_an_axis():
    rng = np.random.default_rng(0)
    label = (np.arange(512).reshape(8, 8, 8) % 6).astype(np.uint8)
    out_image, out_label = augment(label.astype(np.float32), label, rng,
                                   flip_axis=None, p_flip=1.0, p_intensity=0.0, p_noise=0.0)
    assert np.array_equal(out_label, label)


def test_augment_changes_intensities_but_never_labels():
    rng = np.random.default_rng(0)
    label = (np.arange(512).reshape(8, 8, 8) % 6).astype(np.uint8)
    image = np.ones((8, 8, 8), dtype=np.float32)
    out_image, out_label = augment(image, label, rng, flip_axis=None, p_intensity=1.0, p_noise=1.0)
    assert out_image.dtype == np.float32 and out_image.shape == image.shape
    assert not np.array_equal(out_image, image)
    assert np.array_equal(out_label, label)


# ---------------------------------------------------------------------------
# torch wrappers (skipped without torch)
# ---------------------------------------------------------------------------

def test_patch_dataset_items():
    need_torch()
    ds = dataset.HipMRIPatchDataset([make_volume()], patch_size=(8, 8, 8), samples_per_epoch=5)
    assert len(ds) == 5
    item = ds[0]
    assert tuple(item["image"].shape) == (1, 8, 8, 8) and tuple(item["label"].shape) == (8, 8, 8)
    assert "float32" in str(item["image"].dtype) and "int64" in str(item["label"].dtype)
    assert not np.array_equal(np.asarray(ds[0]["image"]), np.asarray(ds[1]["image"])), "draws must differ"


def test_patient_balanced_sampling():  # the sampling logic itself needs no torch
    volumes = [make_volume("a_week0", "A")] + [make_volume(f"b_week{i}", "B") for i in range(7)]
    rng = np.random.default_rng(0)
    for balance, expected in ((True, 0.5), (False, 1 / 8)):
        ds = dataset.HipMRIPatchDataset(volumes, patch_size=(8, 8, 8), balance_patients=balance)
        share_a = np.mean([ds._pick_volume(rng).patient == "A" for _ in range(4000)])
        assert abs(share_a - expected) < 0.04, f"balance={balance}: share of patient A was {share_a:.3f}"


def test_volume_dataset_items():
    need_torch()
    ds = dataset.HipMRIVolumeDataset([make_volume(), make_volume("b002_week0", "B002")])
    assert len(ds) == 2
    item = ds[1]
    assert tuple(item["image"].shape) == (1, 32, 32, 16) and tuple(item["label"].shape) == (32, 32, 16)
    assert item["name"] == "b002_week0" and item["patient"] == "B002"
    assert tuple(item["spacing"].shape) == (3,)


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

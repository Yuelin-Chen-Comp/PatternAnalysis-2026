"""Tests for the pairing and patient-level split logic in dataset.py.

Run from the project folder:  python -m pytest tests     (or: python tests/test_dataset.py)
They use synthetic file names only, so no dataset is needed.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
from dataset import (Case, LeakageError, assert_no_leakage, cases_for, find_cases,  # noqa: E402
                     load_or_create_split, pair_key, patient_id, split_patients)


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
        try:
            patient_id(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse")


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
    try:
        assert_no_leakage(split, cases)
    except LeakageError:
        return
    raise AssertionError("a patient in train and val must raise LeakageError")


def test_incomplete_split_is_detected():
    cases = fake_cases(10)
    split = split_patients([c.patient for c in cases])
    split["train"].pop()
    try:
        assert_no_leakage(split, cases)
    except LeakageError:
        return
    raise AssertionError("an unassigned patient must raise LeakageError")


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
            try:
                load_or_create_split(changed, path)
            except LeakageError:
                continue
            raise AssertionError("a stale split file must raise")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")

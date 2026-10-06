"""Data pipeline for 3D HipMRI prostate segmentation.

Stage 1 (this file so far): pair MR and label volumes, and build a patient-level
train/val/test split with an explicit leakage check. It uses only the standard
library, so it can be tested anywhere.
Stage 2 (to come): preprocessing, patch sampling, augmentation, torch Datasets.

Why patient-level: the study has 38 patients with 1-8 weekly volumes each, and
volumes of one patient are near-copies of each other. Splitting by volume (or by
slice) would put the same anatomy in train and test and inflate Dice. Every
volume of a patient therefore lives in exactly one split.

File naming (confirmed by the data audit):
    MR    : B006_Week0_LFOV.nii.gz
    label : B006_Week0_SEMANTIC.nii.gz
Labels 0-5 are background, body, bone, bladder, rectum, prostate. The order of
1-5 is inferred from voxel fractions and slice extents, to be confirmed by an
overlay plot.

Usage:
    python dataset.py --root /home/groups/comp3710/HipMRI_Study_open
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
MR_DIR = "semantic_MRs"
LABEL_DIR = "semantic_labels_only"
DEFAULT_SPLIT_FILE = Path(__file__).resolve().parent / "splits" / "split_seed42.json"

NII_SUFFIXES = (".nii.gz", ".nii")
SPLIT_NAMES = ("train", "val", "test")

NUM_CLASSES = 6
CLASS_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
PROSTATE = 5

_PATIENT_RE = re.compile(r"^([A-Za-z]\d+)_")
_MODALITY_RE = re.compile(r"[_-]?(lfov|semantic)$", re.IGNORECASE)


class LeakageError(RuntimeError):
    """A patient appears in more than one split, or a split is incomplete."""


@dataclass(frozen=True)
class Case:
    """One MR volume with its label volume."""

    name: str     # pairing key, e.g. 'b006_week0'
    patient: str  # e.g. 'B006'
    mr: Path
    label: Path


def patient_id(filename: str) -> str:
    """'B006_Week0_LFOV.nii.gz' -> 'B006'.

    Strict on purpose: an unparsable name raises instead of silently falling
    back to some guess, because a wrong patient id is exactly how leakage
    sneaks in.
    """
    m = _PATIENT_RE.match(filename)
    if m is None:
        raise ValueError(f"cannot parse a patient id from {filename!r}")
    return m.group(1).upper()


def pair_key(filename: str) -> str:
    """Key shared by an MR and its label: 'B006_Week0_LFOV.nii.gz' and
    'B006_Week0_SEMANTIC.nii.gz' both give 'b006_week0'."""
    stem = filename
    for suffix in NII_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return _MODALITY_RE.sub("", stem).lower()


def _list_nifti(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(NII_SUFFIXES))


def _index_by_key(files: Sequence[Path], what: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for f in files:
        key = pair_key(f.name)
        if key in index:
            raise ValueError(f"two {what} files share the pairing key {key!r}: "
                             f"{index[key].name}, {f.name}")
        index[key] = f
    return index


def find_cases(root, mr_dir: str = MR_DIR, label_dir: str = LABEL_DIR) -> List[Case]:
    """Pair every MR volume with its label volume.

    Raises if any file is unpaired or if the two names disagree on the patient.
    Shapes and affines are checked later, when the volumes are loaded.
    """
    root = Path(root)
    mrs = _index_by_key(_list_nifti(root / mr_dir), "MR")
    labels = _index_by_key(_list_nifti(root / label_dir), "label")
    if mrs.keys() != labels.keys():
        only_mr = sorted(mrs.keys() - labels.keys())
        only_lab = sorted(labels.keys() - mrs.keys())
        raise ValueError(f"MR/label mismatch: {len(only_mr)} MRs without a label "
                         f"(e.g. {only_mr[:3]}), {len(only_lab)} labels without an MR "
                         f"(e.g. {only_lab[:3]})")
    cases = []
    for key in sorted(mrs):
        patient = patient_id(mrs[key].name)
        if patient_id(labels[key].name) != patient:
            raise ValueError(f"patient differs within pair {key!r}")
        cases.append(Case(key, patient, mrs[key], labels[key]))
    return cases


def split_patients(patients: Sequence[str], val_frac: float = 0.15,
                   test_frac: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    """Randomly assign whole patients to train / val / test (seeded, reproducible).

    With 38 patients and the default fractions: 26 train, 6 val, 6 test.
    """
    unique = sorted(set(patients))
    n = len(unique)
    n_val = max(1, round(val_frac * n))
    n_test = max(1, round(test_frac * n))
    if n_val + n_test >= n:
        raise ValueError(f"{n} patients are too few for {n_val} val + {n_test} test")
    order = list(unique)
    random.Random(seed).shuffle(order)
    return {
        "test": sorted(order[:n_test]),
        "val": sorted(order[n_test:n_test + n_val]),
        "train": sorted(order[n_test + n_val:]),
    }


def assert_no_leakage(split: Dict[str, List[str]], cases: Optional[Sequence[Case]] = None) -> None:
    """Raise LeakageError unless the splits are patient-disjoint (and, if cases
    are given, together cover exactly the patients that exist)."""
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            shared = set(split[a]) & set(split[b])
            if shared:
                raise LeakageError(f"patients in both {a} and {b}: {sorted(shared)}")
    if cases is not None:
        existing = {c.patient for c in cases}
        assigned = set().union(*(set(split[n]) for n in SPLIT_NAMES))
        if assigned != existing:
            raise LeakageError(f"split and data disagree: unassigned={sorted(existing - assigned)}, "
                               f"unknown={sorted(assigned - existing)}")


def cases_for(cases: Sequence[Case], split: Dict[str, List[str]], name: str) -> List[Case]:
    """All volumes belonging to the patients of one split."""
    patients = set(split[name])
    return [c for c in cases if c.patient in patients]


def save_split(split: Dict[str, List[str]], path, seed: Optional[int] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, **{name: split[name] for name in SPLIT_NAMES}}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_split(path) -> Dict[str, List[str]]:
    payload = json.loads(Path(path).read_text())
    missing = [n for n in SPLIT_NAMES if n not in payload]
    if missing:
        raise ValueError(f"split file {path} lacks {missing}")
    return {name: list(payload[name]) for name in SPLIT_NAMES}


def load_or_create_split(cases: Sequence[Case], path=DEFAULT_SPLIT_FILE, val_frac: float = 0.15,
                         test_frac: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    """Return the split stored at `path`, or create and save it on first use.

    The stored file is the source of truth, so every run, the 2D baseline and
    the 3D model all see identical patients. A stored split that no longer
    matches the data raises instead of being silently rebuilt.
    """
    path = Path(path)
    if path.exists():
        split = load_split(path)
    else:
        split = split_patients([c.patient for c in cases], val_frac, test_frac, seed)
        assert_no_leakage(split, cases)
        save_split(split, path, seed)
    assert_no_leakage(split, cases)
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description="Pair HipMRI volumes and build the patient-level split.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cases = find_cases(args.root)
    split = load_or_create_split(cases, args.split_file, seed=args.seed)
    print(f"{len(cases)} paired volumes, {len({c.patient for c in cases})} patients")
    for name in SPLIT_NAMES:
        subset = cases_for(cases, split, name)
        print(f"  {name:5s}: {len(split[name]):2d} patients, {len(subset):3d} volumes  {split[name]}")
    print("No patient appears in more than one split: OK")
    print(f"Split file: {args.split_file}")


if __name__ == "__main__":
    main()

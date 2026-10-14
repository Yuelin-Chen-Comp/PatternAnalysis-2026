"""Data pipeline for 3D HipMRI prostate segmentation.

Each step is its own small function so it can be tested alone:

    find_cases       pair every MR volume with its label volume
    split_patients   assign whole patients to train / val / test (stored as JSON)
    load_volume      read the NIfTI files, normalise the MR, validate the labels, cache
    sample_patch     cut a training patch, oversampling the rare organs

Why patient-level splits: the study has 38 patients with 1-8 weekly volumes each, and
the volumes of one patient are near-copies of each other. Splitting by volume (or by
slice) would put the same anatomy in train and test and inflate Dice. Every volume of
a patient therefore lives in exactly one split.

Conventions
-----------
* Volumes keep the file axis order (X, Y, Z) = (256, 256, 128). Z (1.56 mm, constant)
  is the slice axis. Image tensors are (C, X, Y, Z), label tensors (X, Y, Z). Image and
  label go through the same loader and the same crop / flip code, so they cannot drift.
* Labels 0-5: background, body, bone, bladder, rectum, prostate. The order of 1-5 is
  inferred from voxel fractions and slice extents (confirm with an overlay plot).
* File names (confirmed by the data audit):
      MR    : B006_Week0_LFOV.nii.gz
      label : B006_Week0_SEMANTIC.nii.gz

Usage:
    python dataset.py --root /home/groups/comp3710/HipMRI_Study_open            # split summary
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
MR_DIR = "semantic_MRs"
LABEL_DIR = "semantic_labels_only"
DEFAULT_SPLIT_FILE = Path(__file__).resolve().parent / "splits" / "split_seed42.json"

NII_SUFFIXES = (".nii.gz", ".nii")
SPLIT_NAMES = ("train", "val", "test")

NUM_CLASSES = 6
CLASS_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
BLADDER, RECTUM, PROSTATE = 3, 4, 5
FG_CLASSES = (BLADDER, RECTUM, PROSTATE)  # the rare organs that patch sampling oversamples

PREPROC_VERSION = 1      # bump when normalise_mr / label handling changes: it invalidates the cache
CLIP_PERCENTILE = 99.5   # MR intensities above this (computed on non-zero voxels) are clipped

_PATIENT_RE = re.compile(r"^([A-Za-z]\d+)_")
_MODALITY_RE = re.compile(r"[_-]?(lfov|semantic)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Stage 1: pairing and leakage-free splits (standard library only)
# ---------------------------------------------------------------------------

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
    Shapes are checked when the volumes are loaded (load_volume).
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


# ---------------------------------------------------------------------------
# Stage 2: loading and preprocessing
# ---------------------------------------------------------------------------

@dataclass
class Volume:
    """One preprocessed volume held in memory (about 25 MB)."""

    name: str
    patient: str
    image: np.ndarray   # float16 (X, Y, Z), normalised MR
    label: np.ndarray   # uint8   (X, Y, Z), values 0..5
    spacing: Tuple[float, float, float]
    fg_coords: Dict[int, np.ndarray] = field(default_factory=dict)  # class -> (n, 3) voxel indices


def load_nifti(path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Read a NIfTI file as a 3D array plus its voxel spacing (mm)."""
    import nibabel as nib  # imported here so the numpy-only helpers work without nibabel

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim == 4:  # some HipMRI files carry a singleton 4th axis (course spec, Appendix B)
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {data.shape}")
    return data, tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])


def normalise_mr(image: np.ndarray) -> np.ndarray:
    """Clip bright outliers, then z-score using only the non-zero (in-body) voxels.

    The audit showed many exact zeros (p1 = 0, mean about 30 against p99 about 250), so the
    statistics of the whole volume would be dominated by the empty background. Using the
    non-zero voxels gives comparable intensities across volumes. The zero background maps
    to one constant negative value. An all-zero volume returns zeros instead of NaN.
    """
    x = np.asarray(image, dtype=np.float32)
    inside = x > 0
    if not inside.any():
        return np.zeros_like(x)
    x = np.clip(x, 0.0, np.percentile(x[inside], CLIP_PERCENTILE))
    inside = x > 0
    return (x - x[inside].mean()) / (x[inside].std() + 1e-6)


def to_label_array(label: np.ndarray) -> np.ndarray:
    """Labels as uint8 0..5; anything outside the known classes is an error, not silently kept."""
    lab = np.rint(np.asarray(label)).astype(np.int64)
    if lab.min() < 0 or lab.max() >= NUM_CLASSES:
        raise ValueError(f"label values must be in [0, {NUM_CLASSES - 1}], "
                         f"found [{lab.min()}, {lab.max()}]")
    return lab.astype(np.uint8)


def find_foreground(label: np.ndarray, classes: Sequence[int], name: str,
                    max_points: int = 2000) -> Dict[int, np.ndarray]:
    """Voxel coordinates of each rare class, subsampled to max_points (seeded by the volume name).

    Patch sampling uses these to centre a patch on a rare organ without scanning
    the whole 8.4 M-voxel label volume on every draw.
    """
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    coords = {}
    for c in classes:
        idx = np.argwhere(label == c)
        if len(idx) > max_points:
            idx = idx[rng.choice(len(idx), max_points, replace=False)]
        coords[int(c)] = idx.astype(np.int32)
    return coords


def _cache_path(case: Case, cache_dir) -> Optional[Path]:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{case.name}_v{PREPROC_VERSION}.npz"


def load_volume(case: Case, cache_dir=None, fg_classes: Sequence[int] = FG_CLASSES,
                max_points: int = 2000) -> Volume:
    """Load, check and preprocess one case.

    Reading and decompressing the .nii.gz files takes seconds per volume, so with a
    cache_dir the preprocessed arrays are stored once as .npz (never committed: see
    .gitignore) and reloaded in milliseconds by later jobs.
    """
    cached = _cache_path(case, cache_dir)
    if cached is not None and cached.exists():
        with np.load(cached) as z:
            image, label = z["image"], z["label"]
            spacing = tuple(round(float(s), 4) for s in z["spacing"])
    else:
        raw_image, spacing = load_nifti(case.mr)
        raw_label, _ = load_nifti(case.label)
        if raw_image.shape != raw_label.shape:
            raise ValueError(f"{case.name}: image {raw_image.shape} and label "
                             f"{raw_label.shape} differ in shape")
        image = normalise_mr(raw_image).astype(np.float16)
        label = to_label_array(raw_label)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_name(cached.stem + ".tmp.npz")  # write then rename: no half-written cache
            np.savez(tmp, image=image, label=label, spacing=np.asarray(spacing, dtype=np.float32))
            tmp.replace(cached)
    return Volume(case.name, case.patient, image, label, spacing,
                  find_foreground(label, fg_classes, case.name, max_points))


def load_volumes(cases: Sequence[Case], cache_dir=None, fg_classes: Sequence[int] = FG_CLASSES,
                 verbose: bool = True) -> List[Volume]:
    volumes = []
    for i, case in enumerate(cases, 1):
        volumes.append(load_volume(case, cache_dir, fg_classes))
        if verbose and (i % 20 == 0 or i == len(cases)):
            print(f"  loaded {i}/{len(cases)} volumes", flush=True)
    return volumes


# ---------------------------------------------------------------------------
# Stage 2: patch sampling (numpy only)
# ---------------------------------------------------------------------------

def crop_slices(shape: Sequence[int], patch_size: Sequence[int], rng: np.random.Generator,
                center: Optional[Sequence[int]] = None) -> Tuple[slice, ...]:
    """Slices of a patch that lies fully inside `shape`.

    Without a centre the patch position is uniform. With a centre the patch is placed
    around it, shifted by a random jitter of up to a quarter of the patch size per axis
    so the organ is not always in the middle, and then clamped to stay inside.
    """
    slices = []
    for axis, (dim, size) in enumerate(zip(shape, patch_size)):
        if size > dim:
            raise ValueError(f"patch size {tuple(patch_size)} does not fit in volume {tuple(shape)}")
        if center is None:
            start = int(rng.integers(0, dim - size + 1))
        else:
            jitter = int(rng.integers(-(size // 4), size // 4 + 1))
            start = int(np.clip(center[axis] - size // 2 + jitter, 0, dim - size))
        slices.append(slice(start, start + size))
    return tuple(slices)


def sample_patch(volume: Volume, patch_size: Sequence[int], rng: np.random.Generator,
                 p_foreground: float = 0.5,
                 fg_classes: Sequence[int] = FG_CLASSES) -> Tuple[np.ndarray, np.ndarray]:
    """Cut one training patch (image float32, label uint8).

    The prostate is only 0.10% of the voxels, so uniform patches often miss it. With
    probability p_foreground the patch is centred on a random voxel of a randomly chosen
    rare class (bladder, rectum or prostate); otherwise its position is uniform. The
    uniform patches keep the background and the other tissue in the training signal.
    """
    center = None
    if rng.random() < p_foreground:
        available = [c for c in fg_classes if len(volume.fg_coords.get(c, ())) > 0]
        if available:
            coords = volume.fg_coords[available[int(rng.integers(len(available)))]]
            center = tuple(int(v) for v in coords[int(rng.integers(len(coords)))])
    box = crop_slices(volume.image.shape, patch_size, rng, center)
    return volume.image[box].astype(np.float32), np.array(volume.label[box])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

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

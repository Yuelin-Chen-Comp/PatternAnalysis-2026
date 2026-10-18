"""Data pipeline for 3D HipMRI prostate segmentation.

Each step is its own small function so it can be tested alone:

    find_cases       pair every MR volume with its label volume
    split_patients   assign whole patients to train / val / test (stored as JSON)
    load_volume      read the NIfTI files, normalise the MR, validate the labels, cache
    sample_patch     cut a training patch, oversampling the rare organs
    sample_slice     the same for one axial slice (the 2D baseline)
    augment          flip / intensity changes on a patch
    *Dataset         thin torch wrappers around the functions above

Why patient-level splits: the study has 38 patients with 1-8 weekly volumes each, and
the volumes of one patient are near-copies of each other. Splitting by volume (or by
slice) would put the same anatomy in train and test and inflate Dice. Every volume of
a patient therefore lives in exactly one split.

Conventions
-----------
* Volumes keep the file axis order (X, Y, Z) = (256, 256, 128). Z (1.56 mm, constant)
  is the slice axis. Image tensors are (C, X, Y, Z), label tensors (X, Y, Z). Image and
  label go through the same loader and the same crop / flip code, so they cannot drift.
* Axes (confirmed by scripts/check_orientation.py on 40 volumes): X is left-right, Y is
  anterior-posterior (higher index = posterior), Z is superior-inferior (higher index =
  superior, towards the bladder).
* Labels 0-5: background, body, bone, bladder, rectum, prostate. The order of 1-5 is
  inferred from voxel fractions and slice extents (confirm with an overlay plot).
* File names (confirmed by the data audit):
      MR    : B006_Week0_LFOV.nii.gz
      label : B006_Week0_SEMANTIC.nii.gz

Usage:
    python dataset.py --root /home/groups/comp3710/HipMRI_Study_open            # split summary
    python dataset.py --check-load --cache-dir $HOME/hipmri_cache               # + load one volume
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is only needed by the Dataset classes; every helper also works without it
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

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

DEFAULT_PATCH = (128, 128, 64)
PREPROC_VERSION = 1      # bump when normalise_mr / label handling changes: it invalidates the cache
CLIP_PERCENTILE = 99.5   # MR intensities above this (computed on non-zero voxels) are clipped
LEFT_RIGHT_AXIS = 0      # array axis that is left-right, confirmed by scripts/check_orientation.py (affine
                         # codes L, P, S and organ centroid offsets); None would mean "never flip", and a
                         # flip along any other axis is anatomically wrong

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
# Stage 2: patch sampling and augmentation (numpy only)
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


def sample_slice(volume: Volume, rng: np.random.Generator, p_foreground: float = 0.5,
                 fg_classes: Sequence[int] = FG_CLASSES) -> Tuple[np.ndarray, np.ndarray]:
    """Cut one axial slice (the last axis) for the 2D baseline: image float32, label uint8, (X, Y).

    The prostate shows in only about 26 of the 128 slices, so uniform slices would contain it
    one time in five. With probability p_foreground the slice is the one through a random voxel
    of a randomly chosen rare class (the same rule as sample_patch); otherwise it is uniform.
    """
    z = int(rng.integers(volume.image.shape[2]))
    if rng.random() < p_foreground:
        available = [c for c in fg_classes if len(volume.fg_coords.get(c, ())) > 0]
        if available:
            coords = volume.fg_coords[available[int(rng.integers(len(available)))]]
            z = int(coords[int(rng.integers(len(coords)))][2])
    return volume.image[:, :, z].astype(np.float32), np.array(volume.label[:, :, z])


def augment(image: np.ndarray, label: np.ndarray, rng: np.random.Generator,
            flip_axis: Optional[int] = None, p_flip: float = 0.5,
            p_intensity: float = 0.5, p_noise: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Light augmentation of one (image, label) patch; the label only ever gets the flip.

    * left-right flip (only if flip_axis is given): the anatomy is roughly symmetric;
      flipping the other axes would put the bladder behind the rectum.
    * intensity: scale by 0.9-1.1 and shift by up to 0.1 (in z-score units), which mimics
      the scanner-to-scanner gain differences seen between volumes.
    * Gaussian noise with a random standard deviation up to 0.05.
    Spatial warps (rotation, scaling) are left out for now.
    """
    if flip_axis is not None and rng.random() < p_flip:
        image = np.flip(image, axis=flip_axis)
        label = np.flip(label, axis=flip_axis)
    if rng.random() < p_intensity:
        image = image * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if rng.random() < p_noise:
        image = image + rng.normal(0.0, rng.uniform(0.0, 0.05), size=image.shape)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label)


# ---------------------------------------------------------------------------
# Stage 2: torch Datasets (thin wrappers)
# ---------------------------------------------------------------------------

class _RandomSampleDataset(Dataset):
    """Shared logic of the two training datasets: every draw picks a volume and cuts one random
    sample (a patch or a slice) from it. One 'epoch' is `samples_per_epoch` random draws.

    balance_patients: pick a patient uniformly, then one of its volumes. Otherwise a
    patient with 8 weekly scans would be drawn 8 times as often as one with a single scan,
    and the model would fit the frequently scanned patients best.
    """

    def __init__(self, volumes: Sequence[Volume], samples_per_epoch: int, p_foreground: float,
                 fg_classes: Sequence[int], balance_patients: bool, augment_data: bool):
        if not volumes:
            raise ValueError("no volumes given")
        self.volumes = list(volumes)
        self.samples_per_epoch = samples_per_epoch
        self.p_foreground = p_foreground
        self.fg_classes = tuple(fg_classes)
        self.balance_patients = balance_patients
        self.augment_data = augment_data
        self.by_patient: Dict[str, List[int]] = {}
        for i, v in enumerate(self.volumes):
            self.by_patient.setdefault(v.patient, []).append(i)
        self.patients = sorted(self.by_patient)
        self._rng = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _get_rng(self) -> np.random.Generator:
        # created lazily inside each DataLoader worker from that worker's torch seed, so
        # workers draw different samples (and torch.manual_seed makes a run reproducible)
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def _pick_volume(self, rng: np.random.Generator) -> Volume:
        if self.balance_patients:
            idxs = self.by_patient[self.patients[int(rng.integers(len(self.patients)))]]
            return self.volumes[idxs[int(rng.integers(len(idxs)))]]
        return self.volumes[int(rng.integers(len(self.volumes)))]

    def _sample(self, volume: Volume, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, index: int):
        rng = self._get_rng()  # the index is ignored: every draw is random
        image, label = self._sample(self._pick_volume(rng), rng)
        if self.augment_data:  # the left-right axis is axis 0 for patches and for slices alike
            image, label = augment(image, label, rng, flip_axis=LEFT_RIGHT_AXIS)
        return {"image": torch.from_numpy(image).unsqueeze(0),
                "label": torch.from_numpy(label.astype(np.int64))}


class HipMRIPatchDataset(_RandomSampleDataset):
    """Training data for the 3D model: random 3D patches, image (1, X, Y, Z), label (X, Y, Z)."""

    def __init__(self, volumes: Sequence[Volume], patch_size: Sequence[int] = DEFAULT_PATCH,
                 samples_per_epoch: int = 250, p_foreground: float = 0.5,
                 fg_classes: Sequence[int] = FG_CLASSES, balance_patients: bool = True,
                 augment_data: bool = True):
        super().__init__(volumes, samples_per_epoch, p_foreground, fg_classes, balance_patients, augment_data)
        self.patch_size = tuple(patch_size)

    def _sample(self, volume, rng):
        return sample_patch(volume, self.patch_size, rng, self.p_foreground, self.fg_classes)


class HipMRISliceDataset(_RandomSampleDataset):
    """Training data for the 2D baseline: random axial slices, image (1, X, Y), label (X, Y).

    The slices come from the same training volumes, the same patients and the same oversampling
    as the 3D patches, so the baseline and the 3D model differ only in how much context they see.
    """

    def __init__(self, volumes: Sequence[Volume], samples_per_epoch: int = 1000, p_foreground: float = 0.5,
                 fg_classes: Sequence[int] = FG_CLASSES, balance_patients: bool = True,
                 augment_data: bool = True):
        super().__init__(volumes, samples_per_epoch, p_foreground, fg_classes, balance_patients, augment_data)

    def _sample(self, volume, rng):
        return sample_slice(volume, rng, self.p_foreground, self.fg_classes)


class HipMRIVolumeDataset(Dataset):
    """Validation / test data: whole volumes, no randomness. Use batch_size=1 (the volumes
    differ in depth: one has 144 slices). A whole volume fits on the GPU (about 2 GiB for the
    3D U-Net, see the README), so it is predicted in one pass and no sliding window is needed."""

    def __init__(self, volumes: Sequence[Volume]):
        # nibabel returns Fortran-ordered arrays, and the DataLoader's collate copy of such an array is slow: on a
        # Rangpur A100 node 0.44 s per volume instead of 0.04 s (scripts/profile_eval.py). Convert once here
        # (a no-op for arrays that are already C-ordered) rather than once per volume per validation.
        self.volumes = [replace(v, image=np.ascontiguousarray(v.image), label=np.ascontiguousarray(v.label))
                        for v in volumes]

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int):
        v = self.volumes[index]
        return {"image": torch.from_numpy(v.image.astype(np.float32)).unsqueeze(0),
                "label": torch.from_numpy(v.label.astype(np.int64)),
                "name": v.name, "patient": v.patient,
                "spacing": torch.tensor(v.spacing, dtype=torch.float32)}


def make_datasets(root=DEFAULT_ROOT, split_file=DEFAULT_SPLIT_FILE, which: Sequence[str] = ("train", "val"),
                  patch_size: Sequence[int] = DEFAULT_PATCH, samples_per_epoch: Optional[int] = None,
                  cache_dir=None, max_cases_per_split: Optional[int] = None, dim: int = 3, **sample_kwargs):
    """Build the requested datasets from the stored patient split.

    `which` defaults to train and val: the test set should only be built (which=("test",))
    for the final evaluation. max_cases_per_split limits the volumes for quick smoke tests.
    dim = 3 trains on random 3D patches (patch_size, 250 draws per epoch by default); dim = 2
    trains on random axial slices (1000 per epoch by default; patch_size is ignored). The
    validation and test sets are whole volumes in both cases, so both models are scored in 3D.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")
    cases = find_cases(root)
    split = load_or_create_split(cases, split_file)
    datasets = {}
    for name in which:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        subset = cases_for(cases, split, name)
        if max_cases_per_split:
            subset = subset[:max_cases_per_split]
        print(f"{name}: {len(subset)} volumes from {len({c.patient for c in subset})} patients")
        volumes = load_volumes(subset, cache_dir, FG_CLASSES if name == "train" else ())
        if name != "train":
            datasets[name] = HipMRIVolumeDataset(volumes)
        elif dim == 3:
            datasets[name] = HipMRIPatchDataset(volumes, patch_size, samples_per_epoch or 250, **sample_kwargs)
        else:
            datasets[name] = HipMRISliceDataset(volumes, samples_per_epoch or 1000, **sample_kwargs)
    return datasets


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _smoke_check(cases: Sequence[Case], split: Dict[str, List[str]], cache_dir) -> None:
    """Load one validation volume and print what the training pipeline will see."""
    case = cases_for(cases, split, "val")[0]
    print(f"\nSmoke check on validation volume {case.name}")
    vol = load_volume(case, cache_dir)
    img = vol.image.astype(np.float32)
    print(f"  shape={vol.image.shape} spacing={vol.spacing}")
    print(f"  normalised image: min={img.min():.2f} max={img.max():.2f} mean={img.mean():.2f} "
          f"std={img.std():.2f}; {100 * (img == img.min()).mean():.1f}% of voxels are background")
    counts = np.bincount(vol.label.ravel(), minlength=NUM_CLASSES)
    print("  label voxels: " + ", ".join(f"{CLASS_NAMES[i]}={100 * counts[i] / counts.sum():.2f}%"
                                         for i in range(NUM_CLASSES)))
    rng = np.random.default_rng(0)
    for p in (0.0, 0.5):
        hits = sum(bool((sample_patch(vol, DEFAULT_PATCH, rng, p)[1] == PROSTATE).any()) for _ in range(200))
        print(f"  p_foreground={p}: {hits / 200:.0%} of 200 patches {DEFAULT_PATCH} contain prostate")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pair HipMRI volumes, build the patient-level split.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-load", action="store_true", help="also load one val volume and sample patches")
    ap.add_argument("--cache-dir", default=None, help="folder for preprocessed .npz files (not in git)")
    args = ap.parse_args()

    cases = find_cases(args.root)
    split = load_or_create_split(cases, args.split_file, seed=args.seed)
    print(f"{len(cases)} paired volumes, {len({c.patient for c in cases})} patients")
    for name in SPLIT_NAMES:
        subset = cases_for(cases, split, name)
        print(f"  {name:5s}: {len(split[name]):2d} patients, {len(subset):3d} volumes  {split[name]}")
    print("No patient appears in more than one split: OK")
    print(f"Split file: {args.split_file}")
    if args.check_load:
        _smoke_check(cases, split, args.cache_dir)


if __name__ == "__main__":
    main()

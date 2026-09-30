"""Data audit for the HipMRI study (2D slices + 3D volumes).

Run once on Rangpur before writing dataset.py. It answers the questions the
feasibility review needs: how many patients/volumes, what shapes and voxel
spacings, which label values exist and how imbalanced they are, whether
images and labels pair up, and whether the provided 2D train/val/test
folders already leak patients across splits.

Only header information is read for the MR volumes (cheap); label volumes
are read in full because class frequencies are needed.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

NII_SUFFIXES = (".nii.gz", ".nii")


def is_nifti(path: Path) -> bool:
    return path.name.endswith(NII_SUFFIXES)


def list_nifti(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and is_nifti(p))


def patient_id(name: str) -> str:
    """Best-effort patient id from a filename, e.g. 'Case_004_Week0_...' -> 'Case_004'.

    The exact HipMRI naming is confirmed from the printed samples; adjust the
    regex here (and in dataset.py) if the pattern differs.
    """
    m = re.search(r"(case[_-]?\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower().replace("-", "_")
    return re.split(r"[_.]", name)[0]


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_tree(root: Path, depth: int = 2):
    """Print top-level layout with file counts per directory."""
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if len(d.relative_to(root).parts) > depth:
            continue
        n = sum(1 for f in d.iterdir() if f.is_file())
        print(f"  {d.relative_to(root)}/  ({n} files)")


def audit_2d(folder: Path, summary: dict):
    section(f"2D slices: {folder}")
    if not folder.exists():
        print("  (not found)")
        return
    split_patients = {}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(p for p in sub.iterdir() if p.is_file())
        print(f"  {sub.name}: {len(files)} files, e.g. {[f.name for f in files[:3]]}")
        pids = {patient_id(f.name) for f in files}
        split_patients[sub.name] = pids
        print(f"     distinct patient ids (parsed): {len(pids)}")
        nifti = [f for f in files if is_nifti(f)]
        if nifti:
            arr = np.asarray(nib.load(str(nifti[0])).dataobj)
            print(f"     first file shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min():.3g} max={arr.max():.3g}")
            summary.setdefault("slice_2d_examples", {})[sub.name] = list(arr.shape)
    names = list(split_patients)
    print("\n  Patient overlap between provided splits (leakage check):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = split_patients[names[i]] & split_patients[names[j]]
            print(f"    {names[i]} vs {names[j]}: {len(ov)} shared patients")
            summary.setdefault("split_overlap_2d", {})[f"{names[i]}|{names[j]}"] = len(ov)


def audit_3d(mr_dir: Path, label_dir: Path, summary: dict, max_mr_stats: int):
    section(f"3D MR volumes: {mr_dir}")
    mr_files = list_nifti(mr_dir) if mr_dir.exists() else []
    lab_files = list_nifti(label_dir) if label_dir.exists() else []
    print(f"  MR volumes: {len(mr_files)}   label volumes: {len(lab_files)}")
    print(f"  MR examples:    {[f.name for f in mr_files[:3]]}")
    print(f"  label examples: {[f.name for f in lab_files[:3]]}")
    if not mr_files:
        return

    # header-only pass over every MR volume
    shapes, zooms = Counter(), Counter()
    per_patient = defaultdict(int)
    for f in mr_files:
        img = nib.load(str(f))
        shapes[tuple(img.shape)] += 1
        zooms[tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])] += 1
        per_patient[patient_id(f.name)] += 1
    print(f"\n  Distinct patients (parsed): {len(per_patient)}")
    print(f"  Volumes per patient: min={min(per_patient.values())} "
          f"max={max(per_patient.values())}")
    print("  Shapes:")
    for s, c in shapes.most_common():
        print(f"    {s}: {c}")
    print("  Voxel spacing (mm):")
    for z, c in zooms.most_common():
        print(f"    {z}: {c}")
    summary["n_mr"] = len(mr_files)
    summary["n_patients"] = len(per_patient)
    summary["mr_shapes"] = {str(k): v for k, v in shapes.items()}
    summary["mr_zooms"] = {str(k): v for k, v in zooms.items()}

    # intensity statistics on an evenly spaced subset
    idx = np.linspace(0, len(mr_files) - 1, min(max_mr_stats, len(mr_files))).astype(int)
    print(f"\n  Intensity stats on {len(idx)} sampled MR volumes:")
    for i in idx[:5]:
        a = np.asarray(nib.load(str(mr_files[i])).dataobj, dtype=np.float32)
        p1, p99 = np.percentile(a, [1, 99])
        print(f"    {mr_files[i].name}: min={a.min():.1f} p1={p1:.1f} "
              f"mean={a.mean():.1f} p99={p99:.1f} max={a.max():.1f}")

    if not lab_files:
        return

    section(f"3D labels: {label_dir}")
    # pair MR <-> label by stripping common markers from the name
    def key(p: Path) -> str:
        n = p.name
        for suf in NII_SUFFIXES:
            n = n.removesuffix(suf)
        return re.sub(r"(semantic|label|labels|seg|mask)[_-]?", "", n, flags=re.I).lower()

    mr_by_key = {key(f): f for f in mr_files}
    unpaired = [f.name for f in lab_files if key(f) not in mr_by_key]
    print(f"  Labels with no matching MR (by filename): {len(unpaired)}"
          + (f"  e.g. {unpaired[:3]}" if unpaired else ""))

    voxels = Counter()
    total = 0
    shape_mismatch = 0
    slices_with = defaultdict(list)  # label value -> #slices (last axis) containing it
    for f in lab_files:
        img = nib.load(str(f))
        lab = np.asarray(img.dataobj).astype(np.int64)
        vals, cnts = np.unique(lab, return_counts=True)
        for v, c in zip(vals, cnts):
            voxels[int(v)] += int(c)
        total += lab.size
        for v in vals:
            if v == 0 or lab.ndim != 3:
                continue
            slices_with[int(v)].append(int((lab == v).any(axis=(0, 1)).sum()))
        m = mr_by_key.get(key(f))
        if m is not None and tuple(nib.load(str(m)).shape) != tuple(img.shape):
            shape_mismatch += 1
    print(f"  MR/label shape mismatches among paired files: {shape_mismatch}")
    print("\n  Label value -> fraction of all voxels (class imbalance):")
    for v in sorted(voxels):
        frac = voxels[v] / total
        extra = ""
        if v in slices_with:
            s = slices_with[v]
            extra = f"   present in {np.mean(s):.1f} slices/volume on avg (last axis)"
        print(f"    {v}: {100 * frac:8.4f}%{extra}")
    summary["label_fractions"] = {str(v): voxels[v] / total for v in sorted(voxels)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/groups/comp3710/HipMRI_Study_open")
    ap.add_argument("--mr-dir", default="semantic_MRs")
    ap.add_argument("--label-dir", default="semantic_labels_only")
    ap.add_argument("--slices-dir", default="keras_slices_data")
    ap.add_argument("--max-mr-stats", type=int, default=20)
    ap.add_argument("--json-out", default="data_audit_summary.json")
    args = ap.parse_args()

    root = Path(args.root)
    section(f"Dataset root: {root}")
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    show_tree(root)

    summary = {}
    audit_2d(root / args.slices_dir, summary)
    audit_3d(root / args.mr_dir, root / args.label_dir, summary, args.max_mr_stats)

    Path(args.json_out).write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {args.json_out}")


if __name__ == "__main__":
    main()

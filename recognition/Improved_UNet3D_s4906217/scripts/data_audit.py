"""Data audit for the HipMRI study (2D slices + 3D volumes).

Run once on Rangpur before writing dataset.py. It answers the questions the
feasibility review needs: how many patients/volumes, what shapes and voxel
spacings, which label values exist and how imbalanced they are, whether
images and labels pair up (names, shapes, affines), and whether the provided
2D train/val/test folders leak patients across splits.

Only header information is read for the MR volumes (cheap); label volumes
are read in full because class frequencies are needed.

Naming conventions seen in the first audit run:
    2D images : case_004_week_0_slice_0.nii.gz   2D labels: seg_004_week_0_slice_0.nii.gz
    3D MR     : B006_Week0_LFOV.nii.gz           3D labels : B006_Week0_SEMANTIC.nii.gz
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
    """Patient id from a filename.

    'case_004_week_0_slice_0...' and 'seg_004_week_0_slice_0...' -> 'case_004'
    (image and label folders must give the same id for the same patient);
    'B006_Week0_LFOV...' and 'B006_Week0_SEMANTIC...' -> 'B006'.
    """
    m = re.match(r"(?:case|seg)[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"case_{int(m.group(1)):03d}"
    m = re.match(r"([A-Za-z]\d+)", name)
    if m:
        return m.group(1).upper()
    return re.split(r"[_.]", name)[0]


def pair_key(path: Path) -> str:
    """Key shared by an MR volume and its label volume.

    'B006_Week0_LFOV.nii.gz' and 'B006_Week0_SEMANTIC.nii.gz' -> 'b006_week0'.
    """
    n = path.name
    for suf in NII_SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    return re.sub(r"[_-]?(lfov|semantic)$", "", n, flags=re.IGNORECASE).lower()


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

    # leakage is judged between image folders only; a seg folder is the label
    # twin of an image folder and legitimately shares its patients
    image_splits = [n for n in split_patients if "_seg_" not in n]
    print("\n  Patient overlap between provided image splits (leakage check):")
    for i in range(len(image_splits)):
        for j in range(i + 1, len(image_splits)):
            a, b = image_splits[i], image_splits[j]
            ov = split_patients[a] & split_patients[b]
            print(f"    {a} vs {b}: {len(ov)} shared patients")
            summary.setdefault("split_overlap_2d", {})[f"{a}|{b}"] = len(ov)

    print("\n  Each seg folder vs its image folder (patient sets should be identical):")
    for n in split_patients:
        if "_seg_" in n:
            twin = n.replace("_seg_", "_")
            same = twin in split_patients and split_patients[n] == split_patients[twin]
            print(f"    {n} vs {twin}: identical patient set = {same}")
    summary["patients_2d"] = {n: sorted(split_patients[n]) for n in image_splits}


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
    shape_of = {}
    per_patient = defaultdict(int)
    for f in mr_files:
        img = nib.load(str(f))
        shape_of[f.name] = tuple(img.shape)
        shapes[tuple(img.shape)] += 1
        zooms[tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])] += 1
        per_patient[patient_id(f.name)] += 1
    print(f"\n  Distinct patients (parsed): {len(per_patient)}")
    print(f"  Volumes per patient: min={min(per_patient.values())} "
          f"max={max(per_patient.values())} mean={len(mr_files) / len(per_patient):.2f}")
    print("  Shapes:")
    for s, c in shapes.most_common():
        print(f"    {s}: {c}")
    common_shape = shapes.most_common(1)[0][0]
    odd = [n for n, s in shape_of.items() if s != common_shape]
    print(f"  Volumes not matching the most common shape {common_shape}: {odd}")
    print("  Voxel spacing (mm):")
    for z, c in zooms.most_common():
        print(f"    {z}: {c}")
    summary["n_mr"] = len(mr_files)
    summary["n_patients"] = len(per_patient)
    summary["patients_3d"] = sorted(per_patient)
    summary["mr_shapes"] = {str(k): v for k, v in shapes.items()}
    summary["mr_zooms"] = {str(k): v for k, v in zooms.items()}
    summary["odd_shape_volumes"] = odd

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
    mr_by_key = {pair_key(f): f for f in mr_files}
    lab_by_key = {pair_key(f): f for f in lab_files}
    only_lab = sorted(set(lab_by_key) - set(mr_by_key))
    only_mr = sorted(set(mr_by_key) - set(lab_by_key))
    print(f"  Paired MR/label volumes: {len(set(mr_by_key) & set(lab_by_key))}")
    print(f"  Labels with no matching MR: {len(only_lab)}  {only_lab[:3]}")
    print(f"  MRs with no matching label: {len(only_mr)}  {only_mr[:3]}")

    voxels = Counter()
    total = 0
    shape_mismatch = affine_mismatch = 0
    slices_with = defaultdict(list)  # label value -> #slices (last axis) containing it
    per_volume = {}                  # label file -> {label value: {voxels, z_range}}
    for key, f in sorted(lab_by_key.items()):
        img = nib.load(str(f))
        lab = np.asarray(img.dataobj).astype(np.int64)
        counts = np.bincount(lab.ravel())
        vol = {}
        for v in np.flatnonzero(counts):
            v = int(v)
            entry = {"voxels": int(counts[v])}
            voxels[v] += entry["voxels"]
            if v != 0 and lab.ndim == 3:
                z = np.flatnonzero((lab == v).any(axis=(0, 1)))
                entry["z_range"] = [int(z[0]), int(z[-1])]
                slices_with[v].append(len(z))
            vol[v] = entry
        per_volume[f.name] = vol
        total += lab.size
        m = mr_by_key.get(key)
        if m is not None:
            mr_img = nib.load(str(m))
            if tuple(mr_img.shape) != tuple(img.shape):
                shape_mismatch += 1
            elif not np.allclose(mr_img.affine, img.affine, atol=1e-3):
                affine_mismatch += 1
    print(f"  MR/label shape mismatches among paired files: {shape_mismatch}")
    print(f"  MR/label affine mismatches among paired files: {affine_mismatch}")

    n_vol = len(per_volume)
    print("\n  Label value -> fraction of all voxels (class imbalance), volumes containing it,")
    print("  slices per volume (last axis), median first/last slice index:")
    for v in sorted(voxels):
        present = [vol[v] for vol in per_volume.values() if v in vol]
        extra = ""
        if slices_with.get(v):
            s = slices_with[v]
            extra += f"  slices/vol mean={np.mean(s):.1f} (min {min(s)}, max {max(s)})"
            zr = [e["z_range"] for e in present if "z_range" in e]
            extra += (f"  z-range median=[{np.median([a for a, _ in zr]):.0f},"
                      f"{np.median([b for _, b in zr]):.0f}]")
        print(f"    {v}: {100 * voxels[v] / total:8.4f}%  in {len(present)}/{n_vol} volumes{extra}")
    summary["label_fractions"] = {str(v): voxels[v] / total for v in sorted(voxels)}
    summary["pairing"] = {"unpaired_labels": len(only_lab), "unpaired_mrs": len(only_mr),
                          "shape_mismatch": shape_mismatch, "affine_mismatch": affine_mismatch}
    summary["per_volume"] = per_volume


def compare_ids(summary: dict):
    """Can the 2D 'case_NNN' ids be tied to the 3D 'B006'-style ids?"""
    p2d = {pid for ids in summary.get("patients_2d", {}).values() for pid in ids}
    p3d = summary.get("patients_3d", [])
    if not p2d or not p3d:
        return
    section("2D case numbers vs 3D patient ids")
    n2d = {int(p.split("_")[1]) for p in p2d}
    n3d = [int(re.sub(r"\D", "", p)) for p in p3d]
    print(f"  2D distinct case numbers: {len(n2d)}   3D patients: {len(p3d)}, "
          f"distinct numeric parts: {len(set(n3d))}")
    print(f"  Numeric parts shared by 2D and 3D: {len(n2d & set(n3d))}")
    print(f"  3D ids: {p3d}")


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
    compare_ids(summary)

    Path(args.json_out).write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {args.json_out}")


if __name__ == "__main__":
    main()

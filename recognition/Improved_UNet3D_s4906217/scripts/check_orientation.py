"""Which array axis is left-right, anterior-posterior and superior-inferior?

dataset.py may only flip along the left-right axis (a flip along another axis would put the
bladder behind the rectum), and the base/apex analysis needs to know which end of the
prostate is nearer the bladder. This script finds out in two independent ways:

1. the orientation codes stored in the NIfTI affine (can be wrong for resampled data);
2. the anatomy itself: in every pelvis the rectum lies behind the prostate (the
   anterior-posterior axis) and the bladder lies above it (the superior-inferior axis),
   while the left-right axis shows no consistent offset because the anatomy is roughly
   symmetric. So we measure, per axis, the offset between organ centroids over many volumes.

Labels (from the data audit, order inferred): 3 = bladder, 4 = rectum, 5 = prostate.

Usage: python check_orientation.py --n 40
"""
import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

BLADDER, RECTUM, PROSTATE = 3, 4, 5


def centroid(mask):
    idx = np.argwhere(mask)
    return idx.mean(axis=0) if len(idx) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/home/groups/comp3710/HipMRI_Study_open")
    ap.add_argument("--label-dir", default="semantic_labels_only")
    ap.add_argument("--n", type=int, default=40, help="number of volumes, evenly spaced over the sorted list")
    args = ap.parse_args()

    files = sorted(p for p in (Path(args.root) / args.label_dir).iterdir() if p.name.endswith((".nii", ".nii.gz")))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(args.n, len(files))).astype(int)]
    print(f"Using {len(files)} label volumes")

    codes = Counter()
    rect_off, blad_off = [], []
    for f in files:
        img = nib.load(str(f))
        codes[tuple(nib.aff2axcodes(img.affine))] += 1
        lab = np.rint(np.asarray(img.dataobj)).astype(np.int64)
        pros, rect, blad = (centroid(lab == c) for c in (PROSTATE, RECTUM, BLADDER))
        if pros is None or rect is None or blad is None:
            print(f"  skipped {f.name}: a label is missing")
            continue
        rect_off.append(rect - pros)
        blad_off.append(blad - pros)

    print(f"\n1) Orientation codes from the affine: {dict(codes)}")
    rect_off, blad_off = np.array(rect_off), np.array(blad_off)
    print(f"\n2) Centroid offsets in voxels over {len(rect_off)} volumes: mean +- std  (share of volumes > 0)")
    print(f"{'':22s}{'axis 0':>26s}{'axis 1':>26s}{'axis 2':>26s}")
    for name, off in (("rectum - prostate", rect_off), ("bladder - prostate", blad_off)):
        cells = [f"{off[:, a].mean():+7.1f} +- {off[:, a].std():5.1f} ({(off[:, a] > 0).mean():4.0%})" for a in range(3)]
        print(f"{name:22s}" + "".join(f"{c:>26s}" for c in cells))

    score = np.abs(rect_off.mean(axis=0)) + np.abs(blad_off.mean(axis=0))
    lr = int(np.argmin(score))
    others = [a for a in range(3) if a != lr]
    ap_axis = max(others, key=lambda a: abs(rect_off[:, a].mean()))
    si_axis = [a for a in others if a != ap_axis][0]
    print(f"\nReading (check it against the table before trusting it):")
    print(f"  left-right candidate        : axis {lr}  (smallest consistent offset for both organs)")
    print(f"  anterior-posterior candidate: axis {ap_axis}  (rectum is "
          f"{'higher' if rect_off[:, ap_axis].mean() > 0 else 'lower'} index than the prostate, i.e. posterior = "
          f"{'higher' if rect_off[:, ap_axis].mean() > 0 else 'lower'} index)")
    print(f"  superior-inferior candidate : axis {si_axis}  (bladder is "
          f"{'higher' if blad_off[:, si_axis].mean() > 0 else 'lower'} index than the prostate, i.e. superior = "
          f"{'higher' if blad_off[:, si_axis].mean() > 0 else 'lower'} index; the prostate base is the end nearer the bladder)")
    print(f"  => set LEFT_RIGHT_AXIS = {lr} in dataset.py only if the table agrees "
          f"(left-right shares near 50%, the other two near 0% or 100%).")


if __name__ == "__main__":
    main()

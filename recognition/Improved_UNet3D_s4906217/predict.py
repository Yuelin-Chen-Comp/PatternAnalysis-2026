"""Run a trained model on whole volumes and analyse where it fails (the open question of the README).

For every volume of a split the model predicts all six classes, and then, besides the usual per-class Dice and
IoU, the prostate is examined where a good global Dice can hide a failure:

* Dice of the apex, middle and base third of the prostate (thirds of the ground-truth extent along the slice axis);
* how far the predicted prostate reaches beyond the true base and apex, in slices and mm;
* the share of the predicted prostate lying inside the ground-truth rectum or bladder (boundary leakage), and the
  opposite error, where the true prostate goes when it is missed: called bladder, rectum or something else
  (a prostate base that is read as bladder is under-contouring, which this leakage number cannot see);
* the number of connected components of the predicted prostate (small disconnected artifacts); the reach beyond the
  base and apex is measured on the largest component, so that a stray fragment does not count as the prostate;
* the Dice between neighbouring slices, for prediction and ground truth (the slice-to-slice continuity that a
  2D model does not enforce).

It prints the results per volume and per patient (the weekly scans of one patient are not independent, so the
patients are the real sample size), writes results.json, a summary figure, and multi-panel figures (input, ground
truth, prediction, difference) of the worst cases, which are the failure cases to inspect.

    python predict.py --name baseline2d                          # validation split, development
    python predict.py --name unet3d --split test                 # the final analysis on the held-out test patients
    python predict.py --name unet3d --worst 5 --median 1        # 5 worst cases and 1 typical case as figures

Axes (confirmed by scripts/check_orientation.py, see the README): axis 2 is superior-inferior with the superior end
at the high index, so the prostate base (next to the bladder) is the high-z end of the prostate and the apex the low-z end.
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import dataset
import modules
import train
import utils

try:  # connected components; optional, the rest of the analysis does not need it
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None

PROSTATE, BLADDER, RECTUM = dataset.PROSTATE, dataset.BLADDER, dataset.RECTUM
THIRDS = ("apex", "mid", "base")   # in increasing z, i.e. from the inferior end to the superior end
NAN = float("nan")


# ---------------------------------------------------------------------------
# analysis of one predicted volume (numpy only, tested on small synthetic masks)
# ---------------------------------------------------------------------------

def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean masks; NaN when both are empty (nothing to measure)."""
    total = int(a.sum()) + int(b.sum())
    return NAN if total == 0 else 2.0 * int((a & b).sum()) / total


def prostate_extent(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """First and last slice (last axis) that contain the mask, or None if it is empty."""
    present = np.flatnonzero(mask.any(axis=(0, 1)))
    return (int(present[0]), int(present[-1])) if len(present) else None


def thirds(z_lo: int, z_hi: int) -> Dict[str, Tuple[int, int]]:
    """Split the slices z_lo..z_hi into three near-equal slabs, as half-open (start, stop) ranges.

    The names follow the anatomy: the low-z slab is the apex, the high-z slab the base.
    """
    n = z_hi - z_lo + 1
    edges = z_lo + np.round(np.linspace(0, n, 4)).astype(int)
    return {name: (int(edges[i]), int(edges[i + 1])) for i, name in enumerate(THIRDS)}


def adjacent_slice_dice(mask: np.ndarray) -> float:
    """Mean Dice between each slice and the next, over the slices of the mask's own extent.

    A smooth organ gives a value near 1; a segmentation made slice by slice, with ragged or missing slices,
    gives less. A slice pair where only one slice is empty counts as 0; two empty slices are skipped.
    """
    extent = prostate_extent(mask)
    if extent is None or extent[0] == extent[1]:
        return NAN
    m = mask[:, :, extent[0]:extent[1] + 1]
    area = m.sum(axis=(0, 1)).astype(float)
    inter = (m[:, :, :-1] & m[:, :, 1:]).sum(axis=(0, 1)).astype(float)
    denom = area[:-1] + area[1:]
    keep = denom > 0
    return float(np.mean(2 * inter[keep] / denom[keep])) if keep.any() else NAN


def main_blob(mask: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """The largest 26-connected component of a mask, how many components there are, and the share of the mask
    outside the largest one. Without scipy the mask itself is returned and the two numbers are NaN."""
    if ndimage is None or not mask.any():
        return mask, NAN, NAN
    components, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel())[1:]
    largest = int(np.argmax(sizes)) + 1
    return components == largest, int(n), 1.0 - float(sizes.max()) / int(mask.sum())


def miss_fractions(pred_slab: np.ndarray, label_slab: np.ndarray) -> Dict[str, float]:
    """Where the true prostate voxels of a slab go in the prediction, as shares of those voxels:
    kept as prostate (recall), called bladder, called rectum, or called something else (background, body, bone).
    All NaN when the slab holds no true prostate."""
    called = pred_slab[label_slab == PROSTATE]
    if called.size == 0:
        return {"recall": NAN, "miss_to_bladder": NAN, "miss_to_rectum": NAN, "miss_to_other": NAN}
    share = np.bincount(called.astype(np.int64), minlength=dataset.NUM_CLASSES) / called.size
    return {"recall": float(share[PROSTATE]), "miss_to_bladder": float(share[BLADDER]),
            "miss_to_rectum": float(share[RECTUM]),
            "miss_to_other": float(1.0 - share[PROSTATE] - share[BLADDER] - share[RECTUM])}


def analyse_volume(pred: np.ndarray, label: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """The boundary-focused measurements of one volume (flat dict of numbers; NaN where undefined)."""
    gt, pr = label == PROSTATE, pred == PROSTATE
    n_pred = int(pr.sum())
    out = {"gt_prostate_voxels": int(gt.sum()), "pred_prostate_voxels": n_pred}

    blob, n_components, outside_largest = main_blob(pr)
    extent, pred_extent = prostate_extent(gt), prostate_extent(blob)   # the reach is that of the main blob
    out.update(gt_z_lo=extent[0] if extent else NAN, gt_z_hi=extent[1] if extent else NAN,
               gt_slices=(extent[1] - extent[0] + 1) if extent else NAN)
    out["volume_ratio"] = n_pred / out["gt_prostate_voxels"] if out["gt_prostate_voxels"] else NAN
    out.update(dice_apex=NAN, dice_mid=NAN, dice_base=NAN, outside_extent_fraction=NAN,
               base_error_slices=NAN, apex_error_slices=NAN, base_error_mm=NAN, apex_error_mm=NAN)
    out.update(miss_fractions(pred, label))
    for name in THIRDS:
        out.update({f"{key}_{name}": NAN for key in ("recall", "miss_to_bladder", "miss_to_rectum", "miss_to_other")})
    if extent:
        for name, (a, b) in thirds(*extent).items():
            out[f"dice_{name}"] = dice_binary(pr[:, :, a:b], gt[:, :, a:b])
            out.update({f"{key}_{name}": value
                        for key, value in miss_fractions(pred[:, :, a:b], label[:, :, a:b]).items()})
        if n_pred:
            outside = n_pred - int(pr[:, :, extent[0]:extent[1] + 1].sum())
            out["outside_extent_fraction"] = outside / n_pred
        if pred_extent:   # positive base error: the prediction reaches further up (superior) than the truth
            out["base_error_slices"] = pred_extent[1] - extent[1]
            out["apex_error_slices"] = extent[0] - pred_extent[0]   # positive: reaches further down (inferior)
            out["base_error_mm"] = out["base_error_slices"] * spacing[2]
            out["apex_error_mm"] = out["apex_error_slices"] * spacing[2]

    out["leak_rectum"] = int((pr & (label == RECTUM)).sum()) / n_pred if n_pred else NAN
    out["leak_bladder"] = int((pr & (label == BLADDER)).sum()) / n_pred if n_pred else NAN

    out["n_components"], out["outside_largest_fraction"] = n_components, outside_largest

    out["continuity_pred"], out["continuity_gt"] = adjacent_slice_dice(pr), adjacent_slice_dice(gt)
    return out


def nan_stats(values: List[float]) -> Tuple[float, float]:
    """Mean and std of the defined values (NaN if there are none)."""
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    return (float(v.mean()), float(v.std())) if len(v) else (NAN, NAN)


def patient_level(rows: List[Dict], keys: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """Average each measure over the volumes of one patient, then over the patients.

    The weekly volumes of a patient are near-copies of each other, so 48 volumes from 6 patients are 6 independent
    samples. Returns the per-patient means and, for each key, the mean and std over patients.
    """
    patients = sorted({r["patient"] for r in rows})
    per_patient = {p: {k: nan_stats([r[k] for r in rows if r["patient"] == p])[0] for k in keys} for p in patients}
    return per_patient, {k: nan_stats([per_patient[p][k] for p in patients]) for k in keys}


# ---------------------------------------------------------------------------
# model and inference
# ---------------------------------------------------------------------------

def load_model(checkpoint, device):
    """Rebuild the model a checkpoint was trained as and load its weights."""
    ckpt = torch.load(checkpoint, map_location=device)
    saved = ckpt["args"]
    model = train.build_model(saved["model"], saved.get("base_channels"), saved.get("depth")).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), train.model_dim(saved["model"]), ckpt


@torch.no_grad()
def predict_volume(model, dim: int, volume: dataset.Volume, device, amp_dtype) -> Tuple[np.ndarray, float]:
    """Class map (X, Y, Z) uint8 of one whole volume, and the seconds the model call took.

    The 2D baseline is run slice by slice and restacked (modules.SliceWise), so both models give a 3D map.
    """
    image = torch.from_numpy(np.ascontiguousarray(volume.image, dtype=np.float32))[None, None].to(device)
    net = modules.SliceWise(model, chunk_size=32) if dim == 2 else model
    train.sync(device)
    start = time.perf_counter()
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        pred = net(image).argmax(dim=1)[0]
    train.sync(device)
    return pred.cpu().numpy().astype(np.uint8), time.perf_counter() - start


# ---------------------------------------------------------------------------
# figures (matplotlib is optional)
# ---------------------------------------------------------------------------

COLOURS = {"bladder": (0.2, 0.4, 1.0), "rectum": (0.1, 0.8, 0.3), "prostate": (1.0, 0.15, 0.15),
           "tp": (0.1, 0.8, 0.2), "fp": (1.0, 0.15, 0.15), "fn": (0.2, 0.4, 1.0), "leak": (1.0, 0.85, 0.0)}


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib is not installed: no figures are drawn (results.json is still written)")
        return None


def _window(centre: int, half: int, size: int) -> slice:
    start = int(np.clip(centre - half, 0, max(size - 2 * half, 0)))
    return slice(start, min(start + 2 * half, size))


def _tint(shape, masks) -> np.ndarray:
    """RGBA image with each (mask, colour) painted in turn; transparent elsewhere."""
    rgba = np.zeros(shape + (4,))
    for mask, colour in masks:
        rgba[mask] = (*colour, 0.6)
    return rgba


def case_panels(img: np.ndarray, gt: np.ndarray, pr: np.ndarray):
    """(input, ground-truth overlay, prediction overlay, difference overlay) of one 2D plane."""
    organs = lambda lab: [(lab == BLADDER, COLOURS["bladder"]), (lab == RECTUM, COLOURS["rectum"]),  # noqa: E731
                          (lab == PROSTATE, COLOURS["prostate"])]
    g, p = gt == PROSTATE, pr == PROSTATE
    in_oar = (gt == RECTUM) | (gt == BLADDER)
    diff = [(g & p, COLOURS["tp"]), (g & ~p, COLOURS["fn"]), (p & ~g, COLOURS["fp"]), (p & ~g & in_oar, COLOURS["leak"])]
    return img, _tint(gt.shape, organs(gt)), _tint(gt.shape, organs(pr)), _tint(gt.shape, diff)


def plot_case(path: Path, volume: dataset.Volume, pred: np.ndarray, title: str) -> bool:
    """Axial base / middle / apex slices and a sagittal plane through the prostate: four columns
    (input, ground truth, prediction, difference). Blue = bladder, green = rectum, red = prostate; in the
    difference column green = correct prostate, blue = missed (FN), red = false prostate (FP), yellow = FP that
    lies inside the true rectum or bladder."""
    plt = _pyplot()
    if plt is None:
        return False
    image, label = np.asarray(volume.image, dtype=np.float32), volume.label
    gt = label == PROSTATE
    extent = prostate_extent(gt)
    X, Y, Z = label.shape
    if extent:
        centre = np.argwhere(gt).mean(axis=0).round().astype(int)
        n = extent[1] - extent[0]
        shift = int(round(0.15 * n))
        planes = [("base", extent[1] - shift), ("middle", (extent[0] + extent[1]) // 2), ("apex", extent[0] + shift)]
    else:
        centre, planes = np.array([X // 2, Y // 2, Z // 2]), [("middle", Z // 2)]
    vmin, vmax = float(image.min()), float(np.percentile(image, 99.5))
    wx, wy, wz = _window(centre[0], 40, X), _window(centre[1], 40, Y), _window(centre[2], 40, Z)

    rows = []
    for name, z in planes:   # axial: X across, Y down (anterior at the top)
        rows.append((f"axial, {name} (z={z})", "upper",
                     *case_panels(image[wx, wy, z].T, label[wx, wy, z].T, pred[wx, wy, z].T)))
    sx = int(centre[0])      # sagittal: Y across (anterior left), Z up (superior at the top)
    rows.append((f"sagittal (x={sx})", "lower",
                 *case_panels(image[sx, wy, wz].T, label[sx, wy, wz].T, pred[sx, wy, wz].T)))

    fig, axes = plt.subplots(len(rows), 4, figsize=(10.5, 2.7 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, origin, img, gt_overlay, pr_overlay, diff_overlay) in enumerate(rows):
        for c, panel in enumerate((img, gt_overlay, pr_overlay, diff_overlay)):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin=origin)
            if c:
                ax.imshow(panel, origin=origin)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(("input", "ground truth", "prediction", "difference")[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    from matplotlib.patches import Patch   # the colours mean different things in the columns, so say so
    swatch = lambda name: Patch(facecolor=COLOURS[name], alpha=0.85)  # noqa: E731
    fig.legend([swatch("bladder"), swatch("rectum"), swatch("prostate")], ["bladder", "rectum", "prostate"],
               loc="lower center", bbox_to_anchor=(0.40, 0.0), ncol=3, fontsize=8, title_fontsize=8,
               title="ground truth and prediction", frameon=False)
    fig.legend([swatch("tp"), swatch("fn"), swatch("fp"), swatch("leak")],
               ["correct prostate", "missed prostate (FN)", "false prostate (FP)", "FP inside true rectum/bladder"],
               loc="lower center", bbox_to_anchor=(0.80, 0.0), ncol=2, fontsize=8, title_fontsize=8,
               title="difference", frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def plot_summary(path: Path, rows: List[Dict]) -> bool:
    """Per-class Dice and the Dice of the prostate thirds, one box per class / third over the volumes."""
    plt = _pyplot()
    if plt is None:
        return False
    fig, (ax_class, ax_third) = plt.subplots(1, 2, figsize=(11, 4))
    per_class = [[r["dice"][c] for r in rows if r["dice"][c] == r["dice"][c]] for c in range(1, dataset.NUM_CLASSES)]
    ax_class.boxplot(per_class, showmeans=True)
    ax_class.set_xticks(range(1, dataset.NUM_CLASSES))
    ax_class.set_xticklabels(dataset.CLASS_NAMES[1:], fontsize=8)
    ax_class.set(ylabel="Dice (one value per volume)", ylim=(0, 1.02), title="Dice per class")
    names = ["prostate (whole)"] + [f"{t} third" for t in THIRDS[::-1]]   # base first: top to bottom of the organ
    thirds_data = [[r["dice"][PROSTATE] for r in rows]] + [[r[f"dice_{t}"] for r in rows] for t in THIRDS[::-1]]
    thirds_data = [[v for v in d if v == v] for d in thirds_data]               # drop NaN
    ax_third.boxplot(thirds_data, showmeans=True)
    ax_third.set_xticks(range(1, len(names) + 1))
    ax_third.set_xticklabels(names, fontsize=8)
    ax_third.set(ylabel="Dice", ylim=(0, 1.02), title="Prostate Dice: whole, base, middle, apex third")
    for ax in (ax_class, ax_third):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def select_cases(rows: List[Dict], n_worst: int, n_median: int) -> List[Tuple[str, Dict]]:
    """The worst `n_worst` volumes by prostate Dice, and the `n_median` closest to the median."""
    scored = [(r["dice"][PROSTATE], r) for r in rows]
    defined = sorted([s for s in scored if s[0] == s[0]], key=lambda s: s[0])
    chosen = [("worst", r) for _, r in defined[:n_worst]]
    if n_median and defined:
        median = float(np.median([d for d, _ in defined]))
        rest = sorted([s for s in defined[n_worst:]], key=lambda s: abs(s[0] - median))
        chosen += [("typical", r) for _, r in rest[:n_median]]
    return chosen


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: the folder <out-dir>/<name> that holds best.pt")
    ap.add_argument("--out-dir", default=str(Path.home() / "hipmri_runs"))
    ap.add_argument("--checkpoint", default=None, help="a checkpoint to use instead of <out-dir>/<name>/best.pt")
    ap.add_argument("--split", choices=("val", "test"), default="val",
                    help="val while developing; test only for the final analysis")
    ap.add_argument("--root", default=dataset.DEFAULT_ROOT)
    ap.add_argument("--split-file", default=str(dataset.DEFAULT_SPLIT_FILE))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-cases", type=int, default=None, help="use only the first N volumes (smoke tests)")
    ap.add_argument("--worst", type=int, default=4, help="draw the N volumes with the lowest prostate Dice")
    ap.add_argument("--median", type=int, default=1, help="and N typical volumes (closest to the median Dice)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--amp", choices=("bf16", "fp16", "none"), default=None, help="default bf16 on a GPU, none on a CPU")
    return ap.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = train.autocast_dtype(args.amp or ("bf16" if device.type == "cuda" else "none"))
    run_dir = Path(args.out_dir) / args.name
    model, dim, ckpt = load_model(args.checkpoint or run_dir / "best.pt", device)
    out_dir = run_dir / f"predict_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes = dataset.make_datasets(args.root, args.split_file, which=(args.split,), cache_dir=args.cache_dir,
                                    max_cases_per_split=args.max_cases)[args.split].volumes
    print(f"\n{ckpt['args']['model']} (best epoch {ckpt['epoch']}) on the {args.split} split: "
          f"{len(volumes)} volumes, {len({v.patient for v in volumes})} patients")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    by_name, rows = {}, []
    for i, vol in enumerate(volumes, 1):
        pred, seconds = predict_volume(model, dim, vol, device, amp_dtype)
        dice, iou = utils.per_class_scores(torch.from_numpy(pred).long(), torch.from_numpy(vol.label).long(),
                                           dataset.NUM_CLASSES)
        row = {"name": vol.name, "patient": vol.patient, "seconds": seconds,
               "dice": dice.tolist(), "iou": iou.tolist(), "prostate_dice": float(dice[PROSTATE]),
               "prostate_iou": float(iou[PROSTATE]), **analyse_volume(pred, vol.label, vol.spacing)}
        rows.append(row)
        by_name[vol.name] = (vol, pred)
        print(f"  {i:3d}/{len(volumes)}  {vol.name:16s} prostate Dice {row['dice'][PROSTATE]:.3f}  "
              f"(apex {row['dice_apex']:.3f}, mid {row['dice_mid']:.3f}, base {row['dice_base']:.3f})", flush=True)

    summary = train.summarise(rows)
    keys = (["prostate_dice", "dice_apex", "dice_mid", "dice_base", "volume_ratio", "leak_rectum", "leak_bladder",
             "outside_extent_fraction", "base_error_mm", "apex_error_mm", "n_components", "outside_largest_fraction",
             "continuity_pred", "continuity_gt", "recall", "miss_to_bladder", "miss_to_rectum", "miss_to_other"]
            + [f"{k}_{t}" for t in THIRDS for k in ("recall", "miss_to_bladder", "miss_to_rectum", "miss_to_other")])
    for key in keys:
        summary[key + "_mean"], summary[key + "_std"] = nan_stats([r[key] for r in rows])
    per_patient, over_patients = patient_level(rows, keys)
    summary["per_patient"] = per_patient
    summary["patient_level"] = {k: {"mean": m, "std": sd} for k, (m, sd) in over_patients.items()}
    if device.type == "cuda":
        summary["peak_inference_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30

    print(f"\n{'class':12s}{'Dice':>16s}{'IoU':>16s}")
    for c in range(dataset.NUM_CLASSES):
        print(f"{dataset.CLASS_NAMES[c]:12s}{summary['dice_mean'][c]:9.4f} +- {summary['dice_std'][c]:.3f}"
              f"{summary['iou_mean'][c]:9.4f} +- {summary['iou_std'][c]:.3f}")
    print("\nProstate, mean +- std over volumes")
    for label, key in (("Dice, apex third", "dice_apex"), ("Dice, middle third", "dice_mid"),
                       ("Dice, base third", "dice_base"),
                       ("prediction beyond the base, mm (+ = too far up)", "base_error_mm"),
                       ("prediction beyond the apex, mm (+ = too far down)", "apex_error_mm"),
                       ("share of the prediction inside true rectum", "leak_rectum"),
                       ("share of the prediction inside true bladder", "leak_bladder"),
                       ("share of the prediction outside the true z-range", "outside_extent_fraction"),
                       ("connected components of the prediction", "n_components"),
                       ("share outside the largest component", "outside_largest_fraction"),
                       ("neighbouring-slice Dice, prediction", "continuity_pred"),
                       ("neighbouring-slice Dice, ground truth", "continuity_gt")):
        print(f"  {label:52s}{summary[key + '_mean']:9.3f} +- {summary[key + '_std']:.3f}")
    print("\nWhere the true prostate goes: share of its voxels kept as prostate, or called bladder, rectum, other; "
          "mean over volumes")
    print(f"  {'':14s}{'kept (recall)':>15s}{'-> bladder':>12s}{'-> rectum':>11s}{'-> other':>10s}")
    for label, suffix in (("whole prostate", ""), ("base third", "_base"), ("middle third", "_mid"),
                          ("apex third", "_apex")):
        print(f"  {label:14s}" + "".join(f"{summary[k + suffix + '_mean']:{w}.3f}" for k, w in
                                          (("recall", 15), ("miss_to_bladder", 12), ("miss_to_rectum", 11),
                                           ("miss_to_other", 10))))
    print(f"\nPer patient (the weekly volumes of a patient are not independent: {len(per_patient)} patients are "
          f"the real sample size)")
    print(f"  {'patient':8s}{'volumes':>8s}{'prostate':>10s}{'apex':>8s}{'mid':>8s}{'base':>8s}"
          f"{'pred/true vol':>15s}{'base->bladder':>15s}")
    for patient, mean in per_patient.items():
        n = sum(1 for r in rows if r["patient"] == patient)
        print(f"  {patient:8s}{n:8d}{mean['prostate_dice']:10.3f}{mean['dice_apex']:8.3f}{mean['dice_mid']:8.3f}"
              f"{mean['dice_base']:8.3f}{mean['volume_ratio']:15.2f}{mean['miss_to_bladder_base']:15.3f}")
    m, sd = over_patients["prostate_dice"]
    print(f"  prostate Dice, mean +- std over the {len(per_patient)} patients: {m:.3f} +- {sd:.3f}")
    print(f"\n{summary['seconds_per_volume_mean']:.3f} s per volume for the model call"
          + (f", peak GPU memory {summary['peak_inference_gib']:.2f} GiB" if device.type == "cuda" else ""))

    figures = []
    cases = select_cases(rows, args.worst, args.median)
    if not args.no_figures:
        if plot_summary(out_dir / "summary.png", rows):
            figures.append("summary.png")
        for kind, row in cases:
            vol, pred = by_name[row["name"]]
            title = (f"{ckpt['args']['model']}, {args.split}: {row['name']} ({kind} case), prostate Dice "
                     f"{row['dice'][PROSTATE]:.3f} (apex {row['dice_apex']:.2f}, mid {row['dice_mid']:.2f}, "
                     f"base {row['dice_base']:.2f})\n"
                     f"predicted prostate inside true rectum {100 * row['leak_rectum']:.1f}%, bladder "
                     f"{100 * row['leak_bladder']:.1f}%; true prostate called bladder {100 * row['miss_to_bladder']:.1f}% "
                     f"(base third {100 * row['miss_to_bladder_base']:.1f}%), rectum {100 * row['miss_to_rectum']:.1f}%")
            fname = f"{kind}_{row['name']}.png"
            if plot_case(out_dir / fname, vol, pred, title):
                figures.append(fname)
    print("\nCases for the error analysis: " + ", ".join(f"{k} {r['name']} ({r['dice'][PROSTATE]:.3f})" for k, r in cases))

    result = {"model": ckpt["args"]["model"], "split": args.split, "checkpoint_epoch": ckpt["epoch"],
              "summary": summary, "volumes": rows, "cases": [{"kind": k, "name": r["name"]} for k, r in cases],
              "figures": figures}
    train.write_json(out_dir / "results.json", result)
    print(f"\nWrote {out_dir}" + (f" ({', '.join(figures)})" if figures else ""))
    return result


if __name__ == "__main__":
    main()

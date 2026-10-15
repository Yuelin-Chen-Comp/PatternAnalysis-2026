"""Train, validate and test the 2D baseline or the 3D U-Net on the 3D HipMRI volumes.

Both models learn from the same training patients and are scored the same way: every validation
and test volume is predicted whole (the 2D model slice by slice, then restacked), and the Dice and
IoU of each class are computed in 3D. The model with the best validation prostate Dice is kept.

    python train.py --model unet2d --name baseline2d              # the 2D baseline
    python train.py --model unet3d --name unet3d                  # the 3D U-Net
    python train.py --model unet3d --name unet3d --test-only      # score the saved best model on the test split

The test split is only touched with --test / --test-only, never during training, so no choice
(epochs, model selection, learning rate) is made by looking at it.

Everything is written to <out-dir>/<name>/ (default ~/hipmri_runs/<name>, outside the repository because
weights must not be committed): config.json, history.json, curves.png, summary.json, best.pt, last.pt and,
after testing, test_metrics.json.
"""
import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import modules
import utils

NUM_CLASSES = dataset.NUM_CLASSES
CLASS_NAMES = dataset.CLASS_NAMES
PROSTATE = dataset.PROSTATE


# ---------------------------------------------------------------------------
# model, data, small helpers
# ---------------------------------------------------------------------------

def build_model(name: str, base_channels: Optional[int] = None, depth: Optional[int] = None) -> torch.nn.Module:
    """'unet2d' or 'unet3d' with the defaults of modules.py unless a width / depth is given."""
    kwargs = {k: v for k, v in (("base_channels", base_channels), ("depth", depth)) if v is not None}
    if name == "unet2d":
        return modules.UNet2D(**kwargs)
    if name == "unet3d":
        return modules.UNet3D(**kwargs)
    raise ValueError(f"unknown model {name!r}")


def autocast_dtype(amp: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]


def make_scaler(enabled: bool):
    """Loss scaler for fp16 (bf16 and fp32 need none); works on old and new torch."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def json_safe(obj):
    """NaN -> None (valid JSON), tensors and numpy scalars -> plain numbers, recursively."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2) + "\n")


# ---------------------------------------------------------------------------
# training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp_dtype, grad_clip) -> float:
    """One pass over `loader`; returns the mean training loss per sample."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = loss_fn(model(image), label)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)   # clip the real gradients, not the loss-scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * image.shape[0]
        count += image.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, dim: int, device, amp_dtype) -> List[Dict]:
    """Predict every volume of `loader` whole and score it in 3D.

    Returns one row per volume: name, patient, Dice and IoU per class (NaN for a class absent
    from both maps) and the seconds the prediction took. The 2D model is run slice by slice and
    restacked (modules.SliceWise), so both models are scored on the same 3D volumes.
    """
    net = modules.SliceWise(model, chunk_size=32) if dim == 2 else model
    net.eval()
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        sync(device)
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = net(image).argmax(dim=1)
        sync(device)
        seconds = time.perf_counter() - start
        dice, iou = utils.per_class_scores(pred, label, NUM_CLASSES)
        rows.append({"name": batch["name"][0], "patient": batch["patient"][0],
                     "dice": dice.cpu().tolist(), "iou": iou.cpu().tolist(), "seconds": seconds})
    return rows


def summarise(rows: List[Dict]) -> Dict:
    """Mean and std over volumes of the Dice and IoU of every class (a class absent from both maps is skipped)."""
    out = {}
    for key in ("dice", "iou"):
        values = np.array([[np.nan if v is None else v for v in r[key]] for r in rows], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # a class that is NaN in every volume
            out[f"{key}_mean"] = np.nanmean(values, axis=0).tolist()
            out[f"{key}_std"] = np.nanstd(values, axis=0).tolist()
    out["seconds_per_volume_mean"] = float(np.mean([r["seconds"] for r in rows]))
    out["n_volumes"] = len(rows)
    return out


def plot_history(history: List[Dict], path: Path) -> bool:
    """Training loss and per-class validation Dice against the epoch; False if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: skipping curves.png (history.json still has the numbers)")
        return False
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], color="tab:blue")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Training loss (Dice + cross-entropy)")
    ax_loss.grid(alpha=0.3)
    validated = [h for h in history if h.get("val_dice") is not None]
    for c in range(1, NUM_CLASSES):
        ax_dice.plot([h["epoch"] for h in validated],
                     [np.nan if h["val_dice"][c] is None else h["val_dice"][c] for h in validated],
                     label=CLASS_NAMES[c], linewidth=2.5 if c == PROSTATE else 1.2)
    ax_dice.set(xlabel="epoch", ylabel="Dice", ylim=(0, 1), title="Validation Dice per class (whole volumes)")
    ax_dice.grid(alpha=0.3)
    ax_dice.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# testing
# ---------------------------------------------------------------------------

def run_test(run_dir: Path, args, device, amp_dtype) -> Dict:
    """Score the best saved model on the held-out test patients and write test_metrics.json."""
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    saved = checkpoint["args"]
    model = build_model(saved["model"], saved.get("base_channels"), saved.get("depth")).to(device)
    model.load_state_dict(checkpoint["model"])
    dim = 2 if saved["model"] == "unet2d" else 3
    test_set = dataset.make_datasets(args.root, args.split_file, which=("test",), cache_dir=args.cache_dir,
                                     max_cases_per_split=args.max_cases_per_split, dim=dim)["test"]
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = evaluate(model, loader, dim, device, amp_dtype)
    result = {"model": saved["model"], "checkpoint_epoch": checkpoint["epoch"],
              "summary": summarise(rows), "volumes": rows}
    if device.type == "cuda":
        result["summary"]["peak_inference_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    write_json(run_dir / "test_metrics.json", result)
    s = result["summary"]
    print(f"\nTest split: {s['n_volumes']} volumes, {s['seconds_per_volume_mean']:.3f} s per volume")
    print(f"{'class':12s}{'Dice':>16s}{'IoU':>16s}")
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:12s}{s['dice_mean'][c]:9.4f} +- {s['dice_std'][c]:.3f}"
              f"{s['iou_mean'][c]:9.4f} +- {s['iou_std'][c]:.3f}")
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("unet2d", "unet3d"), required=True)
    ap.add_argument("--name", default=None, help="run name (default: the model name)")
    ap.add_argument("--root", default=dataset.DEFAULT_ROOT)
    ap.add_argument("--split-file", default=str(dataset.DEFAULT_SPLIT_FILE))
    ap.add_argument("--cache-dir", default=None, help="folder for preprocessed volumes (about 25 MB each; never committed)")
    ap.add_argument("--out-dir", default=str(Path.home() / "hipmri_runs"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=None, help="random draws per epoch (default 1000 slices / 250 patches)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 16 slices (2D) or 2 patches (3D)")
    ap.add_argument("--patch", type=int, nargs=3, default=list(dataset.DEFAULT_PATCH), help="3D patch size")
    ap.add_argument("--p-foreground", type=float, default=0.5, help="share of samples centred on a rare organ")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=12.0, help="max gradient norm (0 = off)")
    ap.add_argument("--amp", choices=("bf16", "fp16", "none"), default=None, help="default bf16 on a GPU, none on a CPU")
    ap.add_argument("--val-every", type=int, default=2, help="validate every N epochs")
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--max-cases-per-split", type=int, default=None, help="use only the first N volumes per split (smoke tests)")
    ap.add_argument("--test", action="store_true", help="after training, score the best model on the test split")
    ap.add_argument("--test-only", action="store_true", help="do not train; score the saved best model on the test split")
    return ap.parse_args(argv)


def main(argv=None) -> Dict:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp is None:
        args.amp = "bf16" if device.type == "cuda" else "none"
    amp_dtype = autocast_dtype(args.amp)
    dim = 2 if args.model == "unet2d" else 3
    batch_size = args.batch_size or (16 if dim == 2 else 2)
    run_dir = Path(args.out_dir) / (args.name or args.model)
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.test_only:
        return run_test(run_dir, args, device, amp_dtype)

    sets = dataset.make_datasets(args.root, args.split_file, which=("train", "val"), patch_size=args.patch,
                                 samples_per_epoch=args.samples_per_epoch, cache_dir=args.cache_dir,
                                 max_cases_per_split=args.max_cases_per_split, dim=dim,
                                 p_foreground=args.p_foreground)
    train_loader = DataLoader(sets["train"], batch_size=batch_size, num_workers=args.workers,
                              pin_memory=device.type == "cuda", drop_last=len(sets["train"]) >= batch_size,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(sets["val"], batch_size=1, shuffle=False, num_workers=0)

    model = build_model(args.model, args.base_channels, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = make_scaler(args.amp == "fp16" and device.type == "cuda")
    loss_fn = utils.DiceCELoss()

    config = {"args": vars(args), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "torch": torch.__version__, "parameters": modules.count_parameters(model),
              "batch_size": batch_size, "train_volumes": len(sets["train"].volumes),
              "train_patients": len(sets["train"].patients), "val_volumes": len(sets["val"].volumes)}
    write_json(run_dir / "config.json", config)
    print(f"{args.model}: {config['parameters'] / 1e6:.2f} M parameters, batch {batch_size}, amp {args.amp}, "
          f"device {device}; run directory {run_dir}", flush=True)

    history, best, best_epoch = [], -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, amp_dtype, args.grad_clip)
        entry = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        scheduler.step()
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = summarise(evaluate(model, val_loader, dim, device, amp_dtype))
            entry["val_dice"] = val["dice_mean"]
            entry["val_prostate_dice"] = val["dice_mean"][PROSTATE]
            entry["val_mean_dice"] = float(np.nanmean(val["dice_mean"][1:]))
            score = entry["val_prostate_dice"]
            if not math.isnan(score) and score > best:
                best, best_epoch = score, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_prostate_dice": score,
                            "args": json_safe(vars(args))}, run_dir / "best.pt")
        entry["seconds"] = time.perf_counter() - t0
        history.append(entry)
        write_json(run_dir / "history.json", history)
        val_text = f"val prostate Dice {entry['val_prostate_dice']:.4f}, mean {entry['val_mean_dice']:.4f}  " \
            if "val_dice" in entry else ""
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  {val_text}({entry['seconds']:.1f}s)", flush=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": json_safe(vars(args))}, run_dir / "last.pt")

    summary = {"best_val_prostate_dice": best, "best_epoch": best_epoch,
               "train_seconds": time.perf_counter() - started, "epochs": args.epochs}
    if device.type == "cuda":
        summary["peak_train_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    write_json(run_dir / "summary.json", summary)
    plot_history(history, run_dir / "curves.png")
    print(f"\nbest validation prostate Dice {best:.4f} at epoch {best_epoch}; "
          f"{summary['train_seconds'] / 60:.1f} min of training", flush=True)

    if args.test:
        summary["test"] = run_test(run_dir, args, device, amp_dtype)["summary"]
    return summary


if __name__ == "__main__":
    main()

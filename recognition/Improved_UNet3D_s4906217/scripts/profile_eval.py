"""Where does the time of one validation volume go? (train.evaluate took about 1.9 s per volume in a real training
run on a Rangpur A100, while the GPU forward pass alone is about 0.08 s.)

Synthetic volumes (the default) evaluated in 0.125 s per volume, so the slowness depends on the real data or on the
state of a training process. This script therefore also works on the REAL validation volumes, with the trained
weights, and can recreate the training process (data-loader workers running, one training pass done):

    python profile_eval.py                                       # synthetic 256x256x128 volumes, no dataset needed
    python profile_eval.py --real 6 --checkpoint ~/hipmri_runs/baseline2d/best.pt
    python profile_eval.py --real 6 --checkpoint ... --like-training

For one volume it times each stage, one after the other with a GPU sync in between:

    item      HipMRIVolumeDataset[0]: float16 -> float32 and uint8 -> int64 conversions
    loader    the same through a DataLoader (adds the collate copy)
    to_gpu    host -> device copy of image and label
    forward   modules.SliceWise(UNet2D) under bf16 autocast (or UNet3D with --model unet3d)
    argmax    logits -> class map
    scores    utils.per_class_scores (bincount) and .tolist()
    total     train.evaluate on that volume, for reference

and with --real it also times train.evaluate over all the volumes (seconds per volume, as in a training run).
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import modules  # noqa: E402
import train  # noqa: E402
import utils  # noqa: E402


def synthetic_volume(shape, seed=0) -> dataset.Volume:
    """A volume with the real dtypes and a label layout of large and small organs."""
    rng = np.random.default_rng(seed)
    image = rng.normal(0, 1, size=shape).astype(np.float16)
    label = np.zeros(shape, np.uint8)
    x, y, z = shape
    label[x // 5:4 * x // 5, y // 6:5 * y // 6, z // 12:11 * z // 12] = 1
    label[x // 3:2 * x // 3, y // 3:2 * y // 3, z // 3:2 * z // 3] = 2
    label[2 * x // 5:3 * x // 5, 2 * y // 5:3 * y // 5, 2 * z // 5:3 * z // 5] = 3
    label[x // 2 - 6:x // 2 + 6, y // 2 - 6:y // 2 + 6, z // 2 - 8:z // 2 + 8] = 5
    return dataset.Volume("synthetic", "P", image, label, (1.7, 1.7, 1.56))


def usable_cpus() -> int:
    """CPUs this job may use: what Slurm allocated, else the process's CPU affinity."""
    slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if slurm:
        return slurm
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeats):
    """Median seconds of fn() over `repeats` calls after two warm-up calls."""
    for _ in range(2):
        fn()
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def evaluate_all(model, ds, dim, device, amp_dtype, label):
    """train.evaluate over every volume of `ds`; prints the seconds per volume (after one warm-up volume)."""
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    train.evaluate(model, DataLoader(torch.utils.data.Subset(ds, [0]), batch_size=1), dim, device, amp_dtype)
    sync(device)
    t0 = time.perf_counter()
    rows = train.evaluate(model, loader, dim, device, amp_dtype)
    sync(device)
    total = time.perf_counter() - t0
    print(f"  {label:44s}{total / len(ds):7.3f} s per volume over {len(ds)} volumes "
          f"(inside evaluate: {np.mean([r['seconds'] for r in rows]):.3f} s per volume for the model call)")


def stage_report(model, net, ds, dim, device, amp_dtype, repeats):
    """Time each stage of evaluating volume 0 of `ds`."""
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    stage = {"item": timed(lambda: ds[0], device, repeats),
             "loader": timed(lambda: next(iter(loader)), device, repeats)}
    batch = next(iter(loader))
    image, label = batch["image"], batch["label"]
    stage["to_gpu"] = timed(lambda: (image.to(device), label.to(device)), device, repeats)
    image_d, label_d = image.to(device), label.to(device)

    def forward():
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            return net(image_d)

    stage["forward"] = timed(forward, device, repeats)
    logits = forward()
    stage["argmax"] = timed(lambda: logits.argmax(dim=1), device, repeats)
    pred = logits.argmax(dim=1)
    stage["scores"] = timed(lambda: [t.cpu().tolist() for t in utils.per_class_scores(pred, label_d, 6)], device, repeats)
    stage["total"] = timed(lambda: train.evaluate(model, DataLoader(torch.utils.data.Subset(ds, [0]), batch_size=1),
                                                   dim, device, amp_dtype), device, repeats)
    print(f"  stages for volume 0, median of {repeats} (seconds):")
    for k, v in stage.items():
        print(f"    {k:8s}{v:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("unet2d", "unet3d"), default="unet2d")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--real", type=int, default=0, help="use the first N real validation volumes instead of a synthetic one")
    ap.add_argument("--checkpoint", default=None, help="a best.pt to load the trained weights from")
    ap.add_argument("--like-training", action="store_true",
                    help="with --real: first start the training data loader (4 workers) and run one training pass")
    ap.add_argument("--root", default=dataset.DEFAULT_ROOT)
    ap.add_argument("--split-file", default=str(dataset.DEFAULT_SPLIT_FILE))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="small volumes and a tiny model, to check the script anywhere")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2 if args.model == "unet2d" else 3
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    small = {"base_channels": 4, "depth": 2} if args.smoke else {}
    model = (modules.UNet2D(**small) if dim == 2 else modules.UNet3D(**small)).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(os.path.expanduser(args.checkpoint), map_location=device)["model"])
    model.eval()
    net = modules.SliceWise(model, chunk_size=32) if dim == 2 else model
    print(f"device {device}; torch {torch.__version__}; torch threads {torch.get_num_threads()}; "
          f"os.cpu_count {os.cpu_count()}; CPUs usable by this job {usable_cpus()}; "
          f"weights: {'trained (' + args.checkpoint + ')' if args.checkpoint else 'untrained'}")

    if args.real:
        which = ("train", "val") if args.like_training else ("val",)
        sets = dataset.make_datasets(args.root, args.split_file, which=which, cache_dir=args.cache_dir,
                                     max_cases_per_split=args.real, dim=dim, samples_per_epoch=16 * 20)
        ds = sets["val"]
        v = ds.volumes[0]
        print(f"\nfirst real volume {v.name}: image {v.image.dtype} {v.image.shape} "
              f"C-contiguous={v.image.flags['C_CONTIGUOUS']} F-contiguous={v.image.flags['F_CONTIGUOUS']}; "
              f"label {v.label.dtype} F-contiguous={v.label.flags['F_CONTIGUOUS']}")
    else:
        shape = (64, 64, 16) if args.smoke else (256, 256, 128)
        ds = dataset.HipMRIVolumeDataset([synthetic_volume(shape)])
        print(f"\nsynthetic volume {shape}")

    print("\nBefore any training pass:")
    stage_report(model, net, ds, dim, device, amp_dtype, args.repeats)
    if len(ds) > 1:
        evaluate_all(model, ds, dim, device, amp_dtype, "train.evaluate, plain process")

    if args.like_training and args.real:
        train_loader = DataLoader(sets["train"], batch_size=16, num_workers=4, pin_memory=device.type == "cuda",
                                  persistent_workers=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        t0 = time.perf_counter()
        train.train_one_epoch(model, train_loader, utils.DiceCELoss(), optimizer, train.make_scaler(False),
                              device, amp_dtype, 12.0)
        print(f"\nAfter one training pass ({time.perf_counter() - t0:.1f} s) with 4 persistent data-loader workers alive:")
        model.eval()
        stage_report(model, net, ds, dim, device, amp_dtype, args.repeats)
        if len(ds) > 1:
            evaluate_all(model, ds, dim, device, amp_dtype, "train.evaluate, like a training process")


if __name__ == "__main__":
    main()

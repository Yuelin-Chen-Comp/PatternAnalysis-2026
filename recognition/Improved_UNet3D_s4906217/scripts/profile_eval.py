"""Where does the time of one validation volume go? (train.evaluate took about 1.9 s per volume on a Rangpur
A100, while the GPU forward pass alone is about 0.08 s.)

Times every stage of evaluating one whole volume, one after the other with a GPU sync in between, on synthetic
256 x 256 x 128 volumes shaped like the real ones (no dataset needed):

    item      HipMRIVolumeDataset[0]: float16 -> float32 and uint8 -> int64 conversions
    loader    the same through a DataLoader (adds the collate copy)
    to_gpu    host -> device copy of image and label
    forward   modules.SliceWise(UNet2D) under bf16 autocast (or UNet3D with --model unet3d)
    argmax    logits -> class map
    scores    utils.per_class_scores (bincount) and .tolist()
    total     train.evaluate on one volume, for reference

It also repeats the CPU stages with the torch thread count set to the CPUs Slurm gave the job, to test whether
too many OpenMP threads on a few allocated cores slow the CPU work down.

Usage:
    python profile_eval.py                  # on a GPU node (see profile_job.sh)
    python profile_eval.py --smoke          # small volumes, any machine
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("unet2d", "unet3d"), default="unet2d")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="small volumes and a tiny model, to check the script anywhere")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2 if args.model == "unet2d" else 3
    shape = (64, 64, 16) if args.smoke else (256, 256, 128)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    small = {"base_channels": 4, "depth": 2} if args.smoke else {}
    model = (modules.UNet2D(**small) if dim == 2 else modules.UNet3D(**small)).to(device).eval()
    net = modules.SliceWise(model, chunk_size=32) if dim == 2 else model

    cpus = usable_cpus()
    print(f"device {device}; torch {torch.__version__}; torch threads {torch.get_num_threads()}; "
          f"os.cpu_count {os.cpu_count()}; CPUs usable by this job {cpus}; volume {shape}")

    ds = dataset.HipMRIVolumeDataset([synthetic_volume(shape)])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    r = args.repeats

    def cpu_stages():
        return {"item": timed(lambda: ds[0], device, r),
                "loader": timed(lambda: next(iter(loader)), device, r)}

    default = cpu_stages()
    torch.set_num_threads(cpus)
    limited = cpu_stages()
    print(f"\nCPU stages, median of {r} (seconds)")
    print(f"  {'':8s}{'default threads':>18s}{'threads = job CPUs':>22s}")
    for k in default:
        print(f"  {k:8s}{default[k]:18.3f}{limited[k]:22.3f}")

    batch = next(iter(loader))
    image, label = batch["image"], batch["label"]
    stage = {"to_gpu": timed(lambda: (image.to(device), label.to(device)), device, r)}
    image_d, label_d = image.to(device), label.to(device)

    def forward():
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            return net(image_d)

    stage["forward"] = timed(forward, device, r)
    logits = forward()
    stage["argmax"] = timed(lambda: logits.argmax(dim=1), device, r)
    pred = logits.argmax(dim=1)
    stage["scores"] = timed(lambda: [t.cpu().tolist() for t in utils.per_class_scores(pred, label_d, 6)], device, r)
    stage["total"] = timed(lambda: train.evaluate(model, loader, dim, device, amp_dtype), device, r)

    print(f"\nGPU-side stages, median of {r} (seconds)")
    for k, v in stage.items():
        print(f"  {k:8s}{v:8.3f}")
    print("\n'total' runs the whole of train.evaluate on one volume; compare it with the sum of the stages above.")


if __name__ == "__main__":
    main()

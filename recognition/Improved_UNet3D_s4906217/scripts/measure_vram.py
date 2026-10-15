"""Peak GPU memory, parameter count and speed of the 2D and 3D U-Net (feasibility evidence).

No dataset is needed: every configuration runs on random tensors of the right shape, so this
tells us whether a training step fits and what it costs before any real training. Each row is
one training step (forward + loss + backward + AdamW step) or one inference call:

    train  : the optimiser states, gradients and activations are all counted, because the
             peak of a training run is what has to fit on the GPU
    infer  : model.eval() under torch.inference_mode, the memory of one prediction

Rows that do not fit are reported as OOM instead of stopping the run, so one job answers
"does a whole 256 x 256 x 128 volume fit, or do we need patches?" as well as "how big is a patch?".
The loss is a cross-entropy + soft-Dice stand-in with the same memory footprint as the planned
training loss (softmax and one-hot maps of the logits' size).

Usage:
    python measure_vram.py --json-out vram.json        # the full table, on a GPU
    python measure_vram.py --smoke                     # tiny shapes, any machine (checks the script itself)
    python measure_vram.py --only 3D                   # rows whose name contains '3D'

Peak memory is torch.cuda.max_memory_allocated (tensors actually held); 'reserved' is what the
caching allocator took from the driver and what nvidia-smi shows, so it is always larger.
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modules  # noqa: E402

GIB = 2 ** 30
NUM_CLASSES = 6
FULL_VOLUME = (256, 256, 128)
PATCH = (128, 128, 64)

# (name, model, mode, spatial size, batch size, AMP dtype). 'batch size' is patches for the 3D
# model and slices for the 2D model; an inference row for the 2D model is a whole volume
# run slice by slice (SliceWise), so its size is the volume.
CONFIGS = [
    ("3D train patch bs1 bf16", "3d", "train", PATCH, 1, "bf16"),
    ("3D train patch bs2 bf16", "3d", "train", PATCH, 2, "bf16"),
    ("3D train patch bs1 fp32", "3d", "train", PATCH, 1, "none"),
    ("3D train whole volume bs1 bf16", "3d", "train", FULL_VOLUME, 1, "bf16"),
    ("2D train slice bs16 bf16", "2d", "train", FULL_VOLUME[:2], 16, "bf16"),
    ("2D train slice bs32 bf16", "2d", "train", FULL_VOLUME[:2], 32, "bf16"),
    ("3D infer patch bf16", "3d", "infer", PATCH, 1, "bf16"),
    ("3D infer whole volume bf16", "3d", "infer", FULL_VOLUME, 1, "bf16"),
    ("2D infer whole volume (slice by slice) bf16", "2d", "infer", FULL_VOLUME, 1, "bf16"),
]
# tiny shapes for --smoke, keyed by (model, mode); a 2D inference row is a small volume of 6 slices
SMOKE_SIZES = {("3d", "train"): (32, 32, 16), ("3d", "infer"): (32, 32, 16),
               ("2d", "train"): (32, 32), ("2d", "infer"): (32, 32, 6)}


def stand_in_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + soft Dice over the batch; memory-equivalent to the real training loss."""
    ce = F.cross_entropy(logits, target)
    probs = logits.float().softmax(dim=1)
    onehot = F.one_hot(target, logits.shape[1]).movedim(-1, 1).to(probs.dtype)
    dims = [0] + list(range(2, logits.dim()))
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    return ce + 1.0 - ((2 * inter + 1e-5) / (denom + 1e-5)).mean()


def build_model(kind: str, smoke: bool) -> torch.nn.Module:
    small = {"base_channels": 4, "depth": 2} if smoke else {}
    return modules.UNet3D(**small) if kind == "3d" else modules.UNet2D(**small)


def make_scaler(enabled: bool):
    """Loss scaler for fp16 (bf16 and fp32 need none); works on old and new torch."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(kind, mode, size, batch, amp, device, steps, smoke):
    """Run one configuration; returns the raw numbers (raises OutOfMemoryError if it does not fit)."""
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    model = build_model(kind, smoke).to(device)
    params = modules.count_parameters(model)
    x = torch.randn(batch, 1, *size, device=device)
    times = []

    if mode == "train":
        model.train()
        y = torch.randint(0, NUM_CLASSES, (batch, *size), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = make_scaler(amp == "fp16" and device.type == "cuda")
        for i in range(steps + 1):  # step 0 is a warm-up (cuDNN setup, allocator growth)
            sync(device)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = stand_in_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sync(device)
            if i > 0:
                times.append(time.perf_counter() - t0)
    else:
        model.eval()
        net = modules.SliceWise(model, chunk_size=32) if kind == "2d" else model
        with torch.inference_mode():
            for i in range(steps + 1):
                sync(device)
                t0 = time.perf_counter()
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    net(x)
                sync(device)
                if i > 0:
                    times.append(time.perf_counter() - t0)
    return params, statistics.median(times)


def run_config(cfg, device, steps, smoke):
    name, kind, mode, size, batch, amp = cfg
    if smoke:
        size, batch = SMOKE_SIZES[(kind, mode)], min(batch, 2)
    row = {"name": name, "model": kind.upper(), "mode": mode, "size": "x".join(map(str, size)),
           "batch": batch, "amp": amp, "status": "ok"}
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # reset before the model is built: it counts too
    try:
        row["params"], seconds = measure(kind, mode, size, batch, amp, device, steps, smoke)
        row["seconds"] = round(seconds, 4)
        if device.type == "cuda":
            row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
            row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 2)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    gc.collect()  # frees the tensors of the failed or finished run before the next row
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def markdown_table(rows) -> str:
    head = "| Configuration | Input | Batch | Params (M) | Peak allocated (GiB) | Peak reserved (GiB) | Time per step / call (s) |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['name']} | {r['size']} | {r['batch']} | - | **{r['status']}** | - | - |")
            continue
        alloc = r.get("peak_allocated_gib", "n/a")
        resv = r.get("peak_reserved_gib", "n/a")
        lines.append(f"| {r['name']} | {r['size']} | {r['batch']} | {r['params'] / 1e6:.2f} | {alloc} | {resv} | {r['seconds']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=5, help="timed repetitions per row (one extra warm-up is added)")
    ap.add_argument("--only", default=None, help="run only the rows whose name contains this text")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny shapes and a tiny model, to check the script anywhere")
    ap.add_argument("--cudnn-benchmark", action="store_true", help="let cuDNN autotune (can raise the workspace memory)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(0)
    info = {"torch": torch.__version__, "device": device.type, "smoke": args.smoke,
            "cudnn_benchmark": args.cudnn_benchmark}
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, gpu_total_gib=round(prop.total_memory / GIB, 1),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    else:
        print("No CUDA device: memory columns will be n/a and timings are CPU timings.")
    print(json.dumps(info))

    rows = []
    for cfg in CONFIGS:
        if args.only and args.only not in cfg[0]:
            continue
        print(f"running: {cfg[0]}", flush=True)
        rows.append(run_config(cfg, device, args.steps, args.smoke))
        print("  ->", {k: v for k, v in rows[-1].items() if k in ("status", "peak_allocated_gib", "seconds")}, flush=True)

    print("\n" + markdown_table(rows))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"info": info, "rows": rows}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

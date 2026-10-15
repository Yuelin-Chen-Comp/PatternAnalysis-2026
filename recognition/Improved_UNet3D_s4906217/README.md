# 3D Improved U-Net for HipMRI Prostate Segmentation

COMP3710 Report 2026 — Segmentation project (Hard difficulty). Student: s4906217.
Status: work in progress (feasibility stage).

## Feasibility Review

**Open dilemma (Section 2.2).** A good global Dice can hide failures at the prostate base/apex slices and
leakage into the rectum or bladder. Does 3D volumetric context fix these, or does it only smooth the easy
central slices while costing far more GPU memory and time than a 2D baseline?

### 1. User need, scope and acceptance criteria

- **User:** a radiation oncologist or contouring technician who reviews and edits auto-drafted prostate
  contours on pelvic MRI before radiotherapy planning. The model proposes; a human decides.
- **Prototype purpose:** a 3D Improved U-Net (Isensee et al., 2018) for multi-class HipMRI segmentation,
  compared against a 2D U-Net baseline to decide whether 3D context is worth its compute in a contouring pipeline.
- **Acceptance criteria** (all measured on held-out *patients*, identical for baseline and 3D model):
  1. **Accuracy:** mean test prostate DSC >= 0.70 (course target); per-class DSC and IoU reported for all labels.
  2. **Boundary utility:** prostate DSC reported separately for the base / mid / apex thirds along the slice
     axis. 3D is judged to help only if it beats the baseline on base or apex by >= 0.05 DSC.
  3. **Leakage:** report the share of predicted prostate voxels lying inside ground-truth rectum or bladder,
     for both models.
  4. **Integrity:** patient-level split enforced by an assertion in `dataset.py` (zero shared patients).
  5. **Resource:** trains and infers on one Rangpur GPU; peak VRAM, parameter count and per-volume inference
     time recorded for both models; total budget about 10 GPU-hours.

### 2. Model choice and course concepts

- **Data:** downsampled HipMRI 3D (`semantic_MRs` + `semantic_labels_only`): 38 patients, 211 MR volumes
  (e.g. `B006_Week0_LFOV`) with label volumes (`B006_Week0_SEMANTIC`), 1-8 weekly volumes per patient (mean 5.5),
  so splitting by patient (not volume or slice) is essential to avoid leakage. Labels 0-5: background plus body,
  bone, bladder, rectum, prostate. The label order 1-5 is inferred from voxel fractions and slice extents
  (prostate is the smallest, about 26 slices), to be confirmed by an overlay plot. Prostate is only 0.10% of voxels. All six labels are present in every one of
  the 211 volumes.
- **Model (Hard):** 3D Improved U-Net: residual context modules, localisation modules, and deep supervision
  (segmentation outputs at several scales summed), trained with a multi-class soft Dice loss plus cross-entropy
  to counter the extreme prostate class imbalance.
- **Baseline:** 2D U-Net trained on axial slices (along the last axis) extracted from the *same training
  patients' 3D volumes*, then run slice by slice on the *same test volumes* and restacked into 3D, so Dice is
  computed volumetrically on an identical split. The provided `keras_slices_data` is not used: its slices are
  256 x 128, i.e. cut perpendicular to one of the two 256-voxel axes, so the 128-slice stacking axis lies inside
  each 2D image and the baseline would not show the slice-to-slice discontinuity this dilemma is about; its test
  set also has only 3 patients. (The audit shows its split is patient-disjoint, 31 / 4 / 3, and its 38 case numbers
  equal the 38 numeric parts of the 3D ids `B006`, ..., so it likely covers the same patients; this was not
  verified by content.) Stretch: a plain 3D U-Net to separate the effect of 3D context from the
  Improved-U-Net architecture.
- **Course concepts:** convolution and receptive field; residual learning and skip connections; Dice loss for
  imbalanced segmentation (2D OASIS U-Net, Demo 2); mixed-precision training on the Rangpur A100 (DAWNBench,
  Demo 2). The final model is substantially different from Demo 2: 3D, residual, deep supervision, new data, new
  evaluation protocol.

### 3. Preliminary feasibility evidence

Data audit: [`scripts/data_audit.py`](scripts/data_audit.py), run on Rangpur via
[`scripts/audit_job.sh`](scripts/audit_job.sh) (7 Oct 2026; the first run, job 637740, had a faulty pairing key, so the
numbers below come from the corrected run, job 637802). It reads headers for every MR volume and the full label
volumes.

| Audit item | Result |
|---|---|
| Patients / MR volumes / label volumes | 38 / 211 / 211 |
| Volumes per patient | 1-8 (mean 5.5) |
| Volume shape | 256 x 256 x 128 (210 volumes); 256 x 256 x 144 (1 volume) |
| Voxel spacing (mm) | through-plane 1.56 for all; in-plane 1.41-1.88 (1.68 for 152 of 211) |
| MR/label pairing (names, shapes, affines) | 211 / 211 paired; 0 unmatched files, 0 shape mismatches, 0 affine mismatches |
| Voxel fraction per label | 0: 60.35%, 1: 35.46%, 2: 3.37%, 3: 0.57%, 4: 0.14%, 5 (prostate): 0.10% |
| Label presence | all six labels in all 211 volumes (no empty-class Dice at volume level) |
| Slices per volume containing the label (last axis) | 1 and 2: mean 126.6 (115-143); 3: 36.0 (14-74); 4: 52.1 (39-77); 5 (prostate): 26.0 (17-41) |
| Median first-last slice index (last axis) | bladder [72, 107], prostate [53, 79], rectum [46, 96] |
| Provided 2D splits (train / validate / test) | 31 / 4 / 3 patients, 0 shared (the first run's "1 shared" for `seg_` folders was a filename-parsing artifact); the 38 case numbers equal the 38 numeric parts of the 3D ids |

Orientation check ([`scripts/check_orientation.py`](scripts/check_orientation.py), Rangpur job 662280, 40 label
volumes evenly spaced over the sorted file list). The NIfTI affine gives the axis codes (L, P, S) for all 40 volumes,
and the anatomy agrees with it:

| Offset between organ centroids (voxels, mean +- std over 40 volumes; share of volumes > 0) | axis 0 | axis 1 | axis 2 |
|---|---|---|---|
| rectum - prostate | +0.1 +- 1.2 (55%) | +25.3 +- 2.7 (100%) | +8.2 +- 5.3 (98%) |
| bladder - prostate | +0.1 +- 1.7 (50%) | -15.3 +- 6.5 (2%) | +21.7 +- 4.9 (100%) |

Axis 0 shows no consistent offset, so it is left-right. Axis 1 is anterior-posterior (the rectum lies behind the
prostate, the bladder in front of it; higher index = posterior). Axis 2 is superior-inferior (the bladder lies above
the prostate; higher index = superior, so the prostate base is its higher-index end). The last axis, which is the
slice axis of the 2D baseline, is therefore superior-inferior. Flip augmentation is applied along axis 0 only
(`LEFT_RIGHT_AXIS = 0` in `dataset.py`). That the bladder (label 3) sits above and in front of the prostate and the
rectum (label 4) behind it supports the inferred label order; an overlay plot is still to be made.

Patient-level split (seed 42, stored in [`splits/split_seed42.json`](splits/split_seed42.json)): 26 train / 6
validation / 6 test patients. All volumes of a patient stay in one split, and `dataset.py` raises if any patient
appears in two splits or the stored split no longer matches the data.

Resource profile ([`scripts/measure_vram.py`](scripts/measure_vram.py), Rangpur job 662279). GPU: NVIDIA
A100-PCIE-40GB (39.5 GiB usable); PyTorch 2.13.0+cu130 (CUDA 13.0, cuDNN 9.20); random tensors, so no data is needed.
A training step is forward + a cross-entropy + soft-Dice stand-in loss + backward + one AdamW step, and its peak counts
the weights, gradients and optimiser states. bf16 autocast unless stated, `cudnn.benchmark` off, time is the median of
5 steps after one warm-up. `UNet3D` uses 16 base channels, `UNet2D` 32.

| Configuration | Input | Batch | Params (M) | Peak allocated (GiB) | Peak reserved (GiB) | Time per step / call (s) |
|---|---|---|---|---|---|---|
| 3D train, patch, bf16 | 128x128x64 | 1 | 5.65 | 0.68 | 0.92 | 0.056 |
| 3D train, patch, bf16 | 128x128x64 | 2 | 5.65 | 1.29 | 1.80 | 0.050 |
| 3D train, patch, fp32 | 128x128x64 | 1 | 5.65 | 1.19 | 1.27 | 0.038 |
| 3D train, whole volume, bf16 | 256x256x128 | 1 | 5.65 | 4.93 | 6.99 | 0.270 |
| 2D train, slices, bf16 | 256x256 | 16 | 7.76 | 1.70 | 1.93 | 0.033 |
| 2D train, slices, bf16 | 256x256 | 32 | 7.76 | 3.30 | 3.78 | 0.061 |
| 3D inference, patch, bf16 | 128x128x64 | 1 | 5.65 | 0.28 | 0.34 | 0.014 |
| 3D inference, whole volume, bf16 | 256x256x128 | 1 | 5.65 | 2.11 | 2.36 | 0.111 |
| 2D inference, whole volume slice by slice, bf16 | 256x256x128 | 1 | 7.76 | 1.19 | 1.52 | 0.077 |

Reading the table:
- Everything fits with a wide margin. My earlier estimate that whole-volume training would need tens of GB was wrong
  for this model: a whole 256 x 256 x 128 volume trains in 4.93 GiB. Memory is therefore not what forces
  patch-based training. Patches remain the default because the sampler can oversample the prostate (0.10% of the
  voxels) and a patch step is about 5x faster (0.056 s against 0.270 s); whole-volume training is feasible and stays
  available as a comparison. The headroom (4.9 of 39.5 GiB) also leaves room for the extra width, residual units and
  deep-supervision outputs of the Improved U-Net, which will be measured again once it exists.
- For inference on a whole volume, the 3D U-Net needs about 1.5x the time and 1.8x the peak memory of the 2D baseline
  run slice by slice. That is far from the 5-10x that the open question warns about, but it compares untrained models
  of unequal capacity (7.76 M against 5.65 M parameters), so it is a first data point and not the answer. Training
  time to a given Dice will be compared from the real runs.
- The step times are rough: batch 2 took no longer than batch 1, which suggests that a single patch does not saturate
  the GPU and that overhead and noise matter at this size.

### 4. Risks, budget and fallback

| Risk | Mitigation |
|---|---|
| Only 38 patients, so the test set is about 6-8 patients and results are noisy | Report per-patient Dice distributions, not just the mean; use more than one seed; k-fold if time allows |
| Prostate is only 0.10% of voxels (about 26 slices per volume) | Dice + CE loss; oversample foreground patches |
| In-plane spacing varies (1.41-1.88 mm); one volume (K019_Week1) has 144 slices | No resampling, reported as a limitation; patch training and sliding-window inference accept any depth; per-volume spacing for any distance metric |
| 3D memory | Measured (see the resource profile): 0.68 GiB per 128x128x64 patch and 4.93 GiB per whole volume on a 39.5 GiB A100, so low risk; re-measure after the Improved U-Net's residual units and deep supervision are added |
| Label/MR misalignment or naming mismatch | Checked by the audit (211 / 211 paired, 0 shape or affine mismatches); `dataset.py` raises on any unpaired file |
| Rangpur queue contention near the deadline | Submit early; use `--partition=comp3710 --account=comp3710` |

- **Budget:** about 10 GPU-hours on one A100 (2D baseline about 1 h; 3D model about 3-4 h per run, 2-3 runs).
- **Done:** corrected data audit; `dataset.py` with pairing, patient-level split and leakage check, preprocessing
  (clip + z-score), cached loading, patch sampler with rare-organ oversampling, augmentation and torch Datasets
  (26 unit tests, checked by deliberately breaking the code); `modules.py` with the 2D U-Net baseline, a plain 3D
  U-Net and the slice-by-slice wrapper for the baseline (13 unit tests); left-right axis confirmed
  (`LEFT_RIGHT_AXIS = 0`); peak VRAM measured on the Rangpur A100.
- **Next experiment (by 19 Oct check-off):** a 2D baseline smoke test (a few real training steps on HipMRI slices,
  which needs the loss and `train.py`).
- **Fallback:** if the Improved U-Net does not fit memory or cannot reach Dice 0.7, fall back to a plain 3D
  U-Net (Normal difficulty, capped at 15/20) with patch training. It still answers the dilemma against the 2D baseline.

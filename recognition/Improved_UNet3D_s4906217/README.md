# 3D Improved U-Net for HipMRI Prostate Segmentation

COMP3710 Report 2026, Segmentation project 2.2 (Hard difficulty). Student: s4906217.

> **Draft status.** Sections 1-5 and 7-8 are written; section 6 holds the results on the validation split, the
> resource profile, and placeholders (`TODO(...)`) for the results on the held-out test split, which is evaluated once
> after the protocol below was frozen. No `TODO` may remain at submission.

## Contents

1. [The problem and the algorithm](#1-the-problem-and-the-algorithm)
2. [Feasibility review](#2-feasibility-review-written-for-the-check-off)
3. [Setup, reproducibility and usage](#3-setup-reproducibility-and-usage)
4. [Data and pre-processing](#4-data-and-pre-processing)
5. [Models, training and evaluation protocol](#5-models-training-and-evaluation-protocol)
6. [Results](#6-results)
7. [Artificial Intelligence Usage Disclosure](#7-artificial-intelligence-usage-disclosure)
8. [References](#8-references)

## 1. The problem and the algorithm

**Problem.** In prostate radiotherapy the clinical target volume must be delineated to within a few millimetres,
while the bladder, rectum and femoral heads are spared. Contouring is slow, so an annotation assistant that
drafts the contours for a clinician to review and correct would save time. The task here is multi-class
segmentation of 3D pelvic MRI (HipMRI study, 38 patients, 211 volumes) into background, body, bone, bladder, rectum
and prostate, where the prostate is only 0.10% of the voxels. The engineering question of the course is whether a
good global Dice score hides local failures, namely at the base and apex of the prostate and as leakage into the
rectum or bladder, and whether 3D volumetric context removes them or merely smooths the easy central slices at a
much higher memory and time cost than a 2D model.

**Algorithm.** The prototype is the 3D Improved U-Net of Isensee et al. (2018). An encoder of five levels uses
residual *context modules* (two convolutions with instance normalisation, LeakyReLU and a dropout between them, plus
an identity shortcut) and learned stride-2 convolutions instead of pooling. A decoder upsamples, concatenates the
encoder features of the same level and refines them with *localisation modules*. *Deep supervision*: the last three
decoder levels each produce class scores through a 1x1x1 convolution; the coarse scores are upsampled and added to
the finer ones, so the final prediction is a sum of several scales and the lower levels get a short gradient path
from the loss (their half- and quarter-resolution scores also get a loss of their own). The network is trained on
128 x 128 x 64 patches that are oversampled around the rare organs, with a Dice + cross-entropy loss, and predicts
whole volumes at test time. It is compared against a 2D U-Net baseline (run slice by slice and restacked into 3D) and
a plain 3D U-Net, on identical held-out patients.

**Figure.** **TODO(figure)**: one multi-panel case from the test split (input, ground truth, prediction, difference),
see section 6.5.

**Headline result.** **TODO(test)**: one or two sentences with the test results and the recommendation.

## 2. Feasibility review (written for the check-off)

Written for the compulsory feasibility check-off (Wednesday 21 October 2026); the evidence it points to is in the
sections named in brackets.

**Open dilemma (spec section 2.2).** A good global Dice can hide failures at the prostate base and apex and
leakage into the rectum or bladder. Does 3D context fix these, or does it only smooth the easy central slices
while costing far more GPU memory and time than a 2D baseline?

**1. User need, scope and acceptance criteria.** *User:* a radiation oncologist or contouring technician who
reviews and edits auto-drafted prostate contours before radiotherapy planning: the model proposes, a human decides.
*Purpose:* a 3D Improved U-Net for multi-class HipMRI segmentation, compared against a 2D U-Net baseline, to decide
whether 3D context is worth its compute in a contouring pipeline. *Acceptance criteria*, all measured on held-out
patients and identical for the baseline and the 3D model:

1. **Accuracy:** mean test prostate Dice >= 0.70 (course target); per-class Dice and IoU reported for all labels.
2. **Boundary utility:** prostate Dice reported separately for the base, middle and apex thirds along the slice axis.
   3D is judged to help only if it beats the baseline on the base or the apex by >= 0.05 Dice.
3. **Leakage:** the share of predicted prostate voxels inside the ground-truth rectum or bladder, for both models.
4. **Integrity:** the patient-level split is enforced by an assertion in `dataset.py` (no shared patients).
5. **Resource:** trains and infers on one Rangpur GPU; peak VRAM, parameter count and per-volume inference time
   recorded for both models; total budget about 10 GPU-hours.

**2. Model choice and course concepts.** *Data:* the downsampled HipMRI 3D volumes (`semantic_MRs`,
`semantic_labels_only`); because a patient has 1-8 weekly volumes that are near-copies of each other, the split must
be by patient. *Model (Hard):* the 3D Improved U-Net. *Baseline:* a 2D U-Net trained on axial slices of the same
training volumes and run slice by slice on the same test volumes, so Dice is computed in 3D on an identical split;
the provided `keras_slices_data` is not used because its 256 x 128 slices are cut so that the stacking axis lies
inside each image, which would hide the slice-to-slice discontinuity the dilemma is about, and its test set has only
3 patients. A plain 3D U-Net is added so that the effect of 3D context can be separated from the effect of the
Improved-U-Net architecture. *Course concepts:* convolution and receptive field, residual learning and skip
connections, Dice loss for imbalanced segmentation and mixed-precision training on the Rangpur A100 (Demo 2).

**3. Preliminary feasibility evidence.** A data audit (38 patients, 211 paired volumes, shapes and affines consistent,
prostate 0.10% of the voxels; section 4.1); a patient-level split with a leakage assertion (26 / 6 / 6 patients;
section 4.4); the axis orientation confirmed (section 4.2); a measured resource estimate (0.90 GiB per training patch
and 6.43 GiB per whole volume for the Improved U-Net on a 40 GB A100; section 6.3); and a first 2D baseline and 3D
U-Net trained on the real data (validation prostate Dice 0.84 and 0.87; section 6.1).

**4. Risks, budget and fallback.**

| Risk | Mitigation |
|---|---|
| Only 38 patients: the test set has 6 patients and results are noisy | Report per-patient results, several seeds, and distinguish seed noise from patient-to-patient variation |
| Prostate is only 0.10% of the voxels | Dice + cross-entropy loss; oversample patches and slices around the rare organs |
| 3D memory | Measured: low risk (section 6.3) |
| Label / MR misalignment or naming mismatch | Checked by the audit; `dataset.py` raises on any unpaired file |
| Rangpur queue contention near the deadline | Submit early; one job per user runs at a time, so jobs were queued in batches |

*Budget:* about 10 GPU-hours on one A100 (the nine training runs used 2.6 hours, section 6.1). *Next experiment:*
train the baseline and the 3D models and compare them on the validation split. *Fallback:* if the Improved U-Net
cannot reach Dice 0.7, report the plain 3D U-Net (Normal difficulty, capped at 15/20); it still answers the dilemma
against the 2D baseline. (Outcome: the fallback was not needed.)

## 3. Setup, reproducibility and usage

### 3.1 Dependencies

| Package | Version used for the experiments (Rangpur, conda environment `torch`) |
|---|---|
| Python | 3.11 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0, cuDNN 9.20) |
| NumPy | 2.4.6 |
| nibabel | **TODO(versions)** |
| matplotlib | **TODO(versions)** |
| SciPy | 1.17.1 (optional: connected components in `predict.py`) |

The unit tests were run locally on the CPU (Python 3.13.2, PyTorch 2.8.0, NumPy 2.3.3, matplotlib 3.10.7, SciPy
1.17.1; nibabel is not needed by the tests). `modules.py` imports only PyTorch (and the standard library's `typing`; a test checks this); NumPy is used in
the data pipeline and in `train.py` and `predict.py`.

### 3.2 Reproducibility

- The patient split is fixed in [`splits/split_seed42.json`](splits/split_seed42.json) (seed 42, created once and never
  re-drawn) and is the same for every model and every run.
- Every model was trained three times with `--seed 0, 1, 2` (seed 0 is the default), which changes the weight
  initialisation and the random sampling of patches and slices, not the split. Run names: `baseline2d`,
  `unet3d`, `improved3d` for seed 0 and `<name>_s1`, `<name>_s2` for the others.
- Hardware: one NVIDIA A100-PCIE-40GB per run (Rangpur, partition `comp3710`); `cudnn.benchmark` is off. GPU
  kernels are not bit-wise deterministic, so a re-run gives slightly different numbers; the spread over seeds in
  section 6.1 is the relevant uncertainty.
- Weights and datasets are never committed. A run writes `config.json`, `history.json`, `curves.png`,
  `summary.json`, `best.pt` and `last.pt` to `~/hipmri_runs/<name>/`, outside the repository.

### 3.3 Repository layout

| File | Purpose |
|---|---|
| [`dataset.py`](dataset.py) | pairing of MR and label volumes, patient-level split with a leakage assertion, pre-processing, cached loading, patch and slice samplers, augmentation, torch Datasets |
| [`modules.py`](modules.py) | the models: `UNet2D`, `UNet3D`, `ImprovedUNet3D` and the slice-by-slice wrapper `SliceWise` |
| [`utils.py`](utils.py) | the Dice + cross-entropy loss and the per-class Dice / IoU metrics |
| [`train.py`](train.py) | training, validation and (with `--test`) testing; writes the loss and Dice curves |
| [`predict.py`](predict.py) | inference with a saved model and the boundary analysis with figures |
| [`tests/`](tests) | unit tests on synthetic data (no dataset needed) |
| [`scripts/`](scripts) | Slurm job scripts and the data audit, orientation check, VRAM and timing measurements |

### 3.4 Usage

On the CPU, no data needed (all tests use synthetic data):

```bash
python tests/test_dataset.py && python tests/test_modules.py && python tests/test_utils.py \
  && python tests/test_train.py && python tests/test_predict.py        # or: python -m pytest tests
```

On Rangpur (from the `scripts/` folder; the data is read from `/home/groups/comp3710/HipMRI_Study_open`):

```bash
sbatch train_job.sh unet2d baseline2d --val-every 4 --seed 0          # 2D baseline
sbatch train_job.sh unet3d unet3d --val-every 4 --seed 0              # plain 3D U-Net
sbatch train_job.sh improved3d improved3d --val-every 4 --seed 0      # 3D Improved U-Net
sbatch predict_job.sh improved3d --split test                         # predictions, analysis and figures
sbatch vram_job.sh                                                    # peak memory and timing, random tensors
```

The job scripts ask for one hour; the runs above were submitted with `sbatch --time=02:00:00 ...` for the 3D models.

`predict.py --split test` writes `results.json`, `summary.png` and figures of the worst and a typical case to
`~/hipmri_runs/<name>/predict_test/`. The default split of `predict.py` is `val`; the test split is used only with
`--split test`.

## 4. Data and pre-processing

### 4.1 The data and its audit

HipMRI study, downsampled 3D data (`semantic_MRs`, `semantic_labels_only`): 38 patients with 1-8 weekly MR volumes
each (211 volumes, e.g. `B006_Week0_LFOV`) and a label volume per MR (`B006_Week0_SEMANTIC`) with labels 0-5:
background, body, bone, bladder, rectum, prostate. The label order was inferred from voxel fractions and slice extents
and is supported by the anatomy found in section 4.2. The audit ([`scripts/data_audit.py`](scripts/data_audit.py),
run on Rangpur with [`scripts/audit_job.sh`](scripts/audit_job.sh), job 637802; the first run, job 637740, had a
faulty pairing key and is not used) read the headers of every MR volume and all label volumes:

| Audit item | Result |
|---|---|
| Patients / MR volumes / label volumes | 38 / 211 / 211 |
| Volumes per patient | 1-8 (mean 5.5) |
| Volume shape | 256 x 256 x 128 (210 volumes); 256 x 256 x 144 (1 volume) |
| Voxel spacing (mm) | through-plane 1.56 for all; in-plane 1.41-1.88 (1.68 for 152 of 211) |
| MR / label pairing (names, shapes, affines) | 211 / 211 paired; 0 unmatched files, 0 shape mismatches, 0 affine mismatches |
| Voxel fraction per label | 0: 60.35%, 1: 35.46%, 2: 3.37%, 3: 0.57%, 4: 0.14%, 5 (prostate): 0.10% |
| Label presence | all six labels in all 211 volumes |
| Slices per volume containing the label (last axis) | 1 and 2: mean 126.6 (115-143); 3: 36.0 (14-74); 4: 52.1 (39-77); 5 (prostate): 26.0 (17-41) |
| Median first-last slice index (last axis) | bladder [72, 107], prostate [53, 79], rectum [46, 96] |

The course also provides 2D slices (`keras_slices_data`) with their own patient-disjoint split (31 / 4 / 3 patients);
they are not used (section 2).

### 4.2 Axis orientation

[`scripts/check_orientation.py`](scripts/check_orientation.py) (Rangpur job 662280, 40 label volumes evenly spaced over
the file list) determines the anatomical axes in two independent ways. The NIfTI affine gives the axis codes
(L, P, S) for all 40 volumes, and the offsets between organ centroids agree with it:

| Offset between organ centroids (voxels, mean +- std over 40 volumes; share of volumes > 0) | axis 0 | axis 1 | axis 2 |
|---|---|---|---|
| rectum - prostate | +0.1 +- 1.2 (55%) | +25.3 +- 2.7 (100%) | +8.2 +- 5.3 (98%) |
| bladder - prostate | +0.1 +- 1.7 (50%) | -15.3 +- 6.5 (2%) | +21.7 +- 4.9 (100%) |

Axis 0 shows no consistent offset, so it is left-right. Axis 1 is anterior-posterior (rectum behind the prostate,
bladder in front; higher index = posterior). Axis 2 is superior-inferior (bladder above the prostate; higher index =
superior), so the last axis, which is the slice axis of the 2D baseline, runs from the apex (low index) to the
base (high index, next to the bladder). Consequences: flips are applied along axis 0 only, and the thirds of the prostate
are named accordingly (section 5.4).

### 4.3 Pre-processing

Implemented in [`dataset.py`](dataset.py), applied identically to every volume and every model:

- **Intensity normalisation.** Intensities above the 99.5th percentile of the non-zero voxels are clipped, then the
  volume is z-scored with the mean and standard deviation of the non-zero (in-body) voxels only. The audit showed many
  exact zeros (1st percentile 0, mean about 30 against a 99th percentile of about 250), so statistics over the whole
  volume would be dominated by the empty background; the zeros map to one constant value. Volumes are stored as
  float16.
- **Labels** are checked to lie in 0-5 (anything else raises) and stored as uint8. Image and label always go through
  the same loader and the same crop and flip code, so they cannot drift apart.
- **No resampling or cropping:** the volumes are used at the resolution provided (256 x 256 x 128, one volume with 144
  slices); the model accepts any size. The in-plane spacing varies (1.41-1.88 mm) and is not corrected, which is a
  limitation.
- **Axes** are kept in the file order (X, Y, Z) = (left-right, anterior-posterior, superior-inferior).

### 4.4 Patient-level split and its justification

A patient has up to 8 weekly volumes that are near-copies of each other. Splitting by volume or by slice would put the
same anatomy in training and test and inflate the Dice, so every volume of a patient lives in exactly one split.
Patients were assigned at random with seed 42 and the split was written once to
[`splits/split_seed42.json`](splits/split_seed42.json); `dataset.py` raises a `LeakageError` if a patient appears in
two splits or if the stored split no longer matches the data on disk (both are unit-tested).

| Split | Patients | Volumes | Used for |
|---|---|---|---|
| train | 26 | 136 | learning the weights |
| validation | 6 | 48 | choosing the epoch (best prostate Dice) |
| test | 6 | 27 | the final evaluation, read once after the protocol was frozen |

The validation split has the same number of patients as the test split; with only 6 patients
each, a single difficult patient moves a mean visibly (section 6), which is why results are also given per patient.
The proportions follow the usual 70 / 15 / 15; k-fold cross-validation would give less noisy estimates but was not
affordable in the available time.

### 4.5 Patch and slice sampling, augmentation

- **Training samples.** The 3D models are trained on random 128 x 128 x 64 patches, the 2D baseline on random axial
  slices of the same training volumes. With probability 0.5 a patch is centred on (a slice passes through) a random
  voxel of a randomly chosen rare organ (bladder, rectum or prostate), otherwise its position is uniform, so the
  prostate (0.10% of the voxels, about 26 of 128 slices) is seen often enough while the background and body stay in the
  training signal. A patch centre is jittered by up to a quarter of the patch size.
- **Patient balancing.** A training sample picks a patient uniformly and then one of its volumes, so a patient with
  eight weekly scans is not drawn eight times as often as one with a single scan.
- **Augmentation** (training only; the spatial warps of the original paper are left out): a left-right flip along
  axis 0 (probability 0.5; flipping another axis would put the bladder behind the rectum), a random scale of 0.9-1.1
  and shift of up to 0.1 in z-score units (probability 0.5), and Gaussian noise with a standard deviation of up to
  0.05 (probability 0.2). Labels only receive the flip.
- **Validation and test** use whole volumes, no randomness (inference on an entire volume needs 1.2-3.0 GiB, so no
  sliding window is needed).

## 5. Models, training and evaluation protocol

### 5.1 The models

All models are in [`modules.py`](modules.py); they use instance normalisation (a batch of one or two large 3D patches
makes batch statistics noisy; instance norm makes the 2D baseline and the 3D models normalise the same way),
LeakyReLU (slope 0.01), 'same' padding and Kaiming initialisation, and return raw logits.

| Model | Description | Parameters |
|---|---|---|
| `UNet2D` (baseline) | 2D U-Net (Ronneberger et al., 2015): four down-steps, 32-512 channels, two 3x3 convolutions per level, max-pooling, transposed-convolution upsampling, skips; runs on 256 x 256 axial slices and is restacked into a volume | 7.76 M |
| `UNet3D` | the same design in 3D (Cicek et al., 2016), 16-256 channels | 5.65 M |
| `ImprovedUNet3D` | Isensee et al. (2018): residual context modules with dropout 0.3, stride-2 convolutions, localisation modules, three segmentation layers summed (deep supervision), 16-256 channels | 8.27 M |

`UNet2D` and `UNet3D` are one class (`UNet`, `dim` = 2 or 3) and share all building blocks, so a difference between
them comes from the dimensionality alone. The 2D model is wider (32 against 16 base channels) because a slice is cheap;
the models therefore do not have equal capacity, which is stated wherever they are compared. Differences from the
original Improved U-Net: a Dice + cross-entropy loss instead of Dice alone, AdamW with a cosine schedule, and no spatial
augmentation. The code was written for this project from the descriptions in the papers; no code was copied from
other repositories.

### 5.2 Loss, optimiser and training protocol

- **Loss:** cross-entropy plus soft Dice (`utils.DiceCELoss`; Milletari et al., 2016, and the combination of nnU-Net,
  Isensee et al., 2021). The Dice is computed per class over the whole batch (so a patch without prostate stays
  defined), averaged over the five foreground classes, and the background only enters the cross-entropy. For the
  Improved U-Net the half- and quarter-resolution score maps are also scored against labels subsampled to their
  resolution, with weights 0.5 and 0.25, and the weighted sum is divided by 1.75 so that the loss keeps its scale.
- **Optimiser:** AdamW (Loshchilov and Hutter, 2019), learning rate 1e-3, weight decay 1e-5, cosine decay to 0,
  gradient-norm clipping at 12, bf16 mixed precision.
- **Budget per run:** 100 epochs; the 3D models see 250 patches per epoch (batch 2, 12 500 steps), the 2D baseline 1000
  slices per epoch (batch 16, 6 200 steps). The numbers of samples were set once and not tuned.
- **Model selection:** the validation set is scored every 4 epochs and at the last one, and the epoch with the highest
  mean prostate Dice is kept (`best.pt`). The test split is never read during training.
- **Hyper-parameters** were not tuned; they were fixed before the first run of each model (a smoke test of the 2D
  baseline only checked that the pipeline works), so that no model had more tuning than another.

### 5.3 Evaluation protocol (frozen before the test split was evaluated)

Every volume is predicted whole (the 2D baseline slice by slice, restacked into a volume) and scored in 3D. Dice and
IoU of each class are computed per volume and averaged over the volumes; because the weekly volumes of a patient are not
independent, results are also averaged per patient first and then over the 6 patients. For each model the three seeds
are evaluated separately on the test split and reported as mean +- standard deviation over seeds.

### 5.4 Boundary measures (the open question)

Computed by [`predict.py`](predict.py) from the prediction and the ground truth of each volume:

- **Thirds.** The slices from the first to the last one containing true prostate are split into three near-equal
  slabs. By section 4.2 the high-index end is the base (next to the bladder) and the low-index end the apex. The Dice
  of the prostate is computed inside each slab.
- **Reach.** How far the predicted prostate extends beyond the true base and apex, in slices and mm (positive = too
  far), measured on the largest connected component of the prediction so that a stray fragment does not count.
- **Leakage** (over-contouring): the share of the predicted prostate lying inside the true rectum or the true bladder.
- **Where the true prostate goes** (under-contouring): the share of the true prostate voxels that are kept as
  prostate (recall), called bladder, called rectum or called something else (background, body or bone), for the whole
  prostate and for each third. This measure was added after the first analysis of the 2D baseline showed that the
  leakage number, which looks in the opposite direction, cannot see a base that is read as bladder.
- **Slice continuity:** the mean Dice between neighbouring slices of the prostate mask, for the prediction and for the
  ground truth (a model that works slice by slice does not enforce smoothness along the slice axis).
- **Connected components** of the predicted prostate and the share outside the largest one (small disconnected
  artifacts).

## 6. Results

### 6.1 Training results on the validation split

Nine runs (three models, seeds 0, 1 and 2) on the validation split; the best epoch of each run, chosen on this same
split, so the numbers are slightly optimistic. Mean prostate Dice over the 48 validation volumes:

| Model | Seed 0 | Seed 1 | Seed 2 | Mean +- std over seeds | Training time per run |
|---|---|---|---|---|---|
| 2D U-Net | 0.8418 | 0.8494 | 0.8411 | **0.8441 +- 0.0046** | 6.3 min (seed 0: 41 min, before a speed-up of the validation step) |
| 3D U-Net | 0.8720 | 0.8720 | 0.8683 | **0.8707 +- 0.0021** | 14.5 min |
| Improved 3D U-Net | 0.8697 | 0.8763 | 0.8773 | **0.8744 +- 0.0041** | 19.5 min |

The nine runs used 2.6 GPU-hours of training in total. The 3D models are about 0.027 Dice above the 2D baseline, and
every 3D run is above every 2D run, so that difference is well outside the seed noise. The Improved U-Net is 0.004
above the plain 3D U-Net on average (two of its three seeds are above all plain 3D seeds), but with three seeds this
is within the noise (Welch t about 1.4) and is not claimed as an improvement. **TODO(test)**: the same table on the
test split.

### 6.2 Results on the test split

**TODO(test)**: per-class Dice and IoU (mean +- std over seeds, and over the 6 test patients) for the three models;
the per-patient table; acceptance criterion 1.

### 6.3 Resource profile

[`scripts/measure_vram.py`](scripts/measure_vram.py), run on Rangpur twice (job 662279 and a second run on 19 Oct 2026
after the Improved U-Net was added). The peak memory of the rows that both runs have is identical to the last digit; the
table is the second run. GPU: NVIDIA A100-PCIE-40GB (39.5 GiB usable); PyTorch 2.13.0+cu130; random tensors, so no data
is needed. A training step is forward + a cross-entropy + soft-Dice stand-in loss + backward + one AdamW step, and its
peak counts the weights, gradients and optimiser states; the Improved U-Net's step also includes the loss on its two
auxiliary outputs. bf16 autocast unless stated, `cudnn.benchmark` off, time is the median of 5 steps after one warm-up.

| Configuration | Input | Batch | Params (M) | Peak allocated (GiB) | Peak reserved (GiB) | Time per step / call (s) |
|---|---|---|---|---|---|---|
| 3D U-Net train, patch, bf16 | 128x128x64 | 1 | 5.65 | 0.68 | 0.92 | 0.050 |
| 3D U-Net train, patch, bf16 | 128x128x64 | 2 | 5.65 | 1.29 | 1.80 | 0.050 |
| 3D U-Net train, patch, fp32 | 128x128x64 | 1 | 5.65 | 1.19 | 1.27 | 0.038 |
| 3D U-Net train, whole volume, bf16 | 256x256x128 | 1 | 5.65 | 4.93 | 6.99 | 0.269 |
| Improved 3D train, patch, bf16 | 128x128x64 | 1 | 8.27 | 0.90 | 1.16 | 0.049 |
| Improved 3D train, patch, bf16 | 128x128x64 | 2 | 8.27 | 1.69 | 1.96 | 0.069 |
| Improved 3D train, whole volume, bf16 | 256x256x128 | 1 | 8.27 | 6.43 | 7.66 | 0.369 |
| 2D U-Net train, slices, bf16 | 256x256 | 16 | 7.76 | 1.70 | 1.93 | 0.033 |
| 2D U-Net train, slices, bf16 | 256x256 | 32 | 7.76 | 3.30 | 3.78 | 0.061 |
| 3D U-Net inference, patch, bf16 | 128x128x64 | 1 | 5.65 | 0.28 | 0.34 | 0.014 |
| 3D U-Net inference, whole volume, bf16 | 256x256x128 | 1 | 5.65 | 2.11 | 2.36 | 0.111 |
| Improved 3D inference, whole volume, bf16 | 256x256x128 | 1 | 8.27 | 2.98 | 3.63 | 0.150 |
| 2D U-Net inference, whole volume slice by slice, bf16 | 256x256x128 | 1 | 7.76 | 1.19 | 1.52 | 0.076 |

- Everything fits with a wide margin: a whole 256 x 256 x 128 volume trains in 4.93 GiB (3D U-Net) or 6.43 GiB
  (Improved U-Net) of 39.5 GiB, so memory is not what forces patch-based training. Patches are used because the
  sampler can oversample the prostate and a patch step is about 5x faster (0.050 s against 0.269 s). (An early estimate
  that whole volumes would need tens of GB was wrong and is corrected here.)
- The Improved U-Net costs about 1.3x the memory and 1.35-1.4x the time of the plain 3D U-Net.
- For inference on a whole volume, the 3D U-Net needs 1.8x the peak memory and 1.5x the time of the 2D baseline run
  slice by slice, the Improved U-Net 2.5x and 2.0x: far from the 5-10x that the open question warns about, though the
  models have unequal capacity.
- Step times are rough: at a batch of 1 a patch does not saturate the GPU, so overhead and noise matter.
- Validation speed: the validation of 48 volumes took 90 s at first and 8-10 s after
  [`dataset.py`](dataset.py) converted the Fortran-ordered NIfTI arrays to C order once (a
  [profile](scripts/profile_eval.py) showed that the DataLoader's collate copy of such arrays was the main cost).
  **TODO(test)**: per-volume inference times from the test run.

### 6.4 Boundary analysis

**TODO(test)**: the table of the boundary measures of section 5.4 for the three models on the test split (mean +- std
over seeds, per third), the paired per-patient differences between the models, and the verdict on acceptance criteria 2
and 3.

### 6.5 Failure cases

**TODO(test)**: 3-5 failure cases of the test split as multi-panel figures (axial base, middle and apex slices and a
sagittal plane; input, ground truth, prediction, difference; false negatives, false positives and leakage marked),
with the diagnosed trigger of each, and one typical case. For each of the three models if they differ.

### 6.6 Recommendation to the project manager

**TODO(test)**: where 3D context delivers a measurable gain and where it does not, whether the extra memory and time
(sections 6.3) are justified in an operational contouring pipeline, and which anatomical cases remain unsuitable for
automated drafting.

### 6.7 Limitations

- Six test patients: differences smaller than the patient-to-patient spread (about 0.02-0.04 Dice) cannot be
  established; seed noise (0.002-0.005) and patient variation are different things.
- One split and no cross-validation; the test patients were not chosen for being representative.
- The ground truth comes from a single annotator, so part of what is counted as an error may be annotation variability.
- No hyper-parameter tuning for any model; the Improved U-Net in particular may profit from tuning of the dropout,
  the learning rate and the weights of the auxiliary losses, which was not tried.
- The in-plane spacing (1.41-1.88 mm) was not corrected, and no distance-based measure (Hausdorff distance, surface
  distance) is reported.

## 7. Artificial Intelligence Usage Disclosure

**Tools.** Claude (Anthropic), through the Claude Code desktop application (model Claude Sonnet 5.5).
**TODO(student)**: add any other AI tool you used and for what (for example translation or summarising).

**What the AI assisted with.** Reading the assessment specification and proposing a plan; writing the first
versions of [`dataset.py`](dataset.py), [`modules.py`](modules.py), [`utils.py`](utils.py), [`train.py`](train.py),
[`predict.py`](predict.py), the Slurm job scripts, the audit, orientation, VRAM and profiling scripts, and all unit
tests; analysing the numbers and figures of the experiments with me; drafting the commit messages and the text of this
README. **TODO(student)**: state in your own words what you decided and did yourself (the choice of project, the design
and protocol decisions, running every job on Rangpur, reading the results), and what you changed.

**How the AI-written code was audited, verified and tested.**

- 87 unit tests on synthetic data (data pipeline, models, loss and metrics, the training pipeline end to end, the boundary
  analysis), including tests that predict.py and train.py give the same Dice for the same checkpoint.
- *Deliberate-breakage checks:* for the models, the loss, the training script and the analysis, bugs were inserted on
  purpose (for example no padding, an ignored skip connection, swapped bladder and rectum, patient-level averaging
  replaced by volume-level averaging, the test set read during training) and the tests had to fail. This found two
  weak tests, which were rewritten (a test that did not distinguish a real skip connection from any other input of the
  concatenation, and a test that did not notice the test split being loaded during training).
- *Measurements instead of estimates:* an AI estimate that whole volumes would need tens of GB was replaced by a
  measurement (4.9-6.4 GiB); a claim made from two figures that the base is "read as bladder" was checked against
  a new measure on all 48 volumes and turned out to hold for only 2.5% of the base voxels on average (the dominant error is
  a boundary that is too tight, section 6.4); a slow validation step was traced with a profiler to a memory-layout
  problem instead of guessing.
- *Checks against the real data:* the audit, the orientation check, the nine training runs and the predictions were run on
  Rangpur, and the figures were inspected by eye.
- Limits of this verification: the unit tests ran locally with a different PyTorch version (2.8.0) from the
  experiments (2.13.0). **TODO(student)**: confirm that you read the code you submit and can explain every equation
  and design choice.

## 8. References

1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation", MICCAI 2015.
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, "Brain tumor segmentation and radiomics
   survival prediction: Contribution to the BRATS 2017 challenge", Brainlesion 2018, pp. 287-297.
3. O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation
   from sparse annotation", MICCAI 2016, pp. 424-432.
4. F. Milletari, N. Navab, S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image
   segmentation", 3DV 2016.
5. F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, K. H. Maier-Hein, "nnU-Net: a self-configuring method for deep
   learning-based biomedical image segmentation", Nature Methods 18, 2021.
6. I. Loshchilov, F. Hutter, "Decoupled weight decay regularization", ICLR 2019.
7. D. Ulyanov, A. Vedaldi, V. Lempitsky, "Instance normalization: the missing ingredient for fast stylization",
   arXiv:1607.08022, 2016.
8. K. He, X. Zhang, S. Ren, J. Sun, "Delving deep into rectifiers: surpassing human-level performance on ImageNet
   classification", ICCV 2015 (the initialisation).
9. HipMRI study data and the course specification (COMP3710 Report 2026, Pattern Analysis, The University of
   Queensland).

"""Model components for HipMRI prostate segmentation (PyTorch only, no NumPy).

This file holds the three models the study compares:

    UNet2D           the baseline: a 2D U-Net (Ronneberger et al., 2015) run on axial slices
    UNet3D           a plain 3D U-Net (Cicek et al., 2016) run on 3D patches
    ImprovedUNet3D   the 3D Improved U-Net (Isensee et al., 2018): residual context modules, localisation
                     modules and deep supervision (see ImprovedUNet below)

UNet2D and UNet3D are the same class, UNet, with dim = 2 or 3. They share ConvBlock, Down and Up,
so the only thing that differs between the baseline and the 3D model is the dimensionality:
a difference in Dice or memory cannot come from different normalisation or activations.

Tensor layout (matches dataset.py):
    3D model  input (B, 1, X, Y, Z)   output logits (B, num_classes, X, Y, Z)
    2D model  input (B, 1, X, Y)      output logits (B, num_classes, X, Y)
A volume is (X, Y, Z) = (256, 256, 128) and Z is the slice axis, so the 2D baseline sees
256 x 256 axial slices. The models return raw logits; softmax lives in the loss.

Differences from the original papers, chosen on purpose:
* 'same' padding (padding=1) instead of 'valid', so the output has the input's size and the
  Dice is computed on every voxel.
* Instance normalisation by default instead of batch normalisation. A 3D patch is large, so
  only a batch of 1-2 fits in memory, and batch statistics from one or two patches are noisy.
  Instance norm does not depend on the batch, so the 2D baseline (batch of many slices) and
  the 3D model (batch of 1-2 patches) are normalised the same way.
* The channel count doubles at every level starting from base_channels, which is smaller
  than the originals so a 128 x 128 x 64 patch trains on one GPU.
"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

_CONV = {2: nn.Conv2d, 3: nn.Conv3d}
_CONV_TRANSPOSE = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
_MAX_POOL = {2: nn.MaxPool2d, 3: nn.MaxPool3d}
_INSTANCE_NORM = {2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
_BATCH_NORM = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
_DROPOUT = {2: nn.Dropout2d, 3: nn.Dropout3d}


def _check_dim(dim: int) -> None:
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")


def init_weights(module: nn.Module) -> None:
    """Kaiming-normal initialisation for every convolution (use with model.apply).

    The gain matches the LeakyReLU(0.01) that follows each convolution, which keeps the size
    of the activations roughly constant from the first level to the deepest one.
    """
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters, for the resource table in the README."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class ConvBlock(nn.Module):
    """Two times (3x3(x3) convolution -> normalisation -> LeakyReLU): the unit of one U-Net level.

    The convolutions have no bias because the normalisation that follows would cancel it
    (instance norm brings its own learnable shift).
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int = 3, norm: str = "instance"):
        super().__init__()
        _check_dim(dim)
        if norm == "instance":
            make_norm = lambda channels: _INSTANCE_NORM[dim](channels, affine=True)  # noqa: E731
        elif norm == "batch":
            make_norm = _BATCH_NORM[dim]
        else:
            raise ValueError(f"norm must be 'instance' or 'batch', got {norm!r}")
        layers = []
        for cin in (in_channels, out_channels):
            layers += [_CONV[dim](cin, out_channels, kernel_size=3, padding=1, bias=False),
                       make_norm(out_channels),
                       nn.LeakyReLU(0.01, inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder step: halve every spatial size with a max-pool, then a ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, dim: int = 3, norm: str = "instance"):
        super().__init__()
        _check_dim(dim)
        self.pool = _MAX_POOL[dim](kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels, dim, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder step: double the spatial sizes, concatenate the encoder skip, then a ConvBlock.

    The upsampling is a learned 2x2(x2) transposed convolution with stride 2. Max-pooling
    rounds odd sizes down (15 -> 7), so the upsampled map (14) can be smaller than the skip
    (15); it is zero-padded at the end to match, which makes any input size work.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 dim: int = 3, norm: str = "instance"):
        super().__init__()
        _check_dim(dim)
        self.up = _CONV_TRANSPOSE[dim](in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dim, norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        missing = [s - u for s, u in zip(skip.shape[2:], x.shape[2:])]
        if any(missing):
            pad = []
            for m in reversed(missing):  # F.pad lists the last axis first
                pad += [0, m]
            x = F.pad(x, pad)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net with `depth` down-steps; dim = 2 gives the 2D model, dim = 3 the 3D model.

    Level i has base_channels * 2**i channels, so the default (base 16, depth 4) goes
    16, 32, 64, 128, 256 from the first level to the bottleneck. Every spatial size of the
    input should be at least 2**depth, and the bottleneck must hold more than one voxel
    (instance norm cannot normalise a single value).
    """

    def __init__(self, dim: int = 3, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 16, depth: int = 4, norm: str = "instance"):
        super().__init__()
        _check_dim(dim)
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.dim = dim
        channels = [base_channels * 2 ** level for level in range(depth + 1)]
        self.first = ConvBlock(in_channels, channels[0], dim, norm)
        self.downs = nn.ModuleList(Down(channels[i], channels[i + 1], dim, norm) for i in range(depth))
        self.ups = nn.ModuleList(Up(channels[i + 1], channels[i], channels[i], dim, norm)
                                 for i in reversed(range(depth)))
        self.head = _CONV[dim](channels[0], num_classes, kernel_size=1)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != self.dim + 2:
            raise ValueError(f"a {self.dim}D UNet expects (B, C{', X, Y, Z' if self.dim == 3 else ', X, Y'}) "
                             f"input, got shape {tuple(x.shape)}")
        skips = [self.first(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()  # the bottleneck
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


class UNet2D(UNet):
    """The baseline: a 2D U-Net. Base 32 channels, since a 256 x 256 slice is cheap."""

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 32,
                 depth: int = 4, norm: str = "instance"):
        super().__init__(2, in_channels, num_classes, base_channels, depth, norm)


class UNet3D(UNet):
    """A plain 3D U-Net. Base 16 channels, so a 128 x 128 x 64 patch fits on one GPU."""

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 16,
                 depth: int = 4, norm: str = "instance"):
        super().__init__(3, in_channels, num_classes, base_channels, depth, norm)


# ---------------------------------------------------------------------------
# The Improved U-Net (Isensee et al., 2018)
# ---------------------------------------------------------------------------

def _conv_norm_act(in_channels: int, out_channels: int, dim: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    """Convolution -> instance norm -> LeakyReLU; the stride-2 version is the learned downsampling."""
    return nn.Sequential(
        _CONV[dim](in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
        _INSTANCE_NORM[dim](out_channels, affine=True),
        nn.LeakyReLU(0.01, inplace=True))


class ContextModule(nn.Module):
    """Pre-activation residual block: (norm, LeakyReLU, 3x3(x3) conv) twice with a dropout in between, and the
    input added to the result.

    The identity path lets the gradient reach the early layers directly, which is what makes a deeper network
    trainable on so little data; the dropout (over whole feature maps) regularises it. The module keeps the
    number of channels and the spatial size.
    """

    def __init__(self, channels: int, dim: int = 3, dropout: float = 0.3):
        super().__init__()
        _check_dim(dim)
        norm = _INSTANCE_NORM[dim]
        self.block = nn.Sequential(
            norm(channels, affine=True), nn.LeakyReLU(0.01, inplace=True),
            _CONV[dim](channels, channels, kernel_size=3, padding=1, bias=False),
            _DROPOUT[dim](dropout),
            norm(channels, affine=True), nn.LeakyReLU(0.01, inplace=True),
            _CONV[dim](channels, channels, kernel_size=3, padding=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ImprovedUNet(nn.Module):
    """The Improved U-Net of Isensee et al. (2018), for dim = 3 (the model of this study) or dim = 2.

    Context pathway (encoder). A 3x3x3 convolution, then at every level a ContextModule; between levels a
    3x3x3 convolution with stride 2 (a learned downsampling, instead of max-pooling) that doubles the channels.

    Localisation pathway (decoder). Upsample by repeating the voxels (nearest neighbour), a 3x3x3 convolution that
    halves the channels, concatenate the context features of that level, then a localisation module: a 3x3x3
    convolution followed by a 1x1x1 convolution.

    Deep supervision. The last `deep_levels` localisation outputs each get a 1x1x1 convolution to class scores
    (a segmentation layer). The coarsest of them is upsampled and added to the next one, and so on up to the full
    resolution, so the final logits are the sum of the scores of several scales and the lower levels receive a
    short gradient path straight from the loss. With return_aux = True the lower-resolution score maps are also
    returned (highest resolution first, strides in self.aux_strides) so that they can be given a loss of their own.

    Every spatial size of the input should be at least 2**depth, and the bottleneck must hold more than one
    voxel (instance norm cannot normalise a single value). Other sizes work: the upsampling is done to the
    exact size of the skip connection.
    """

    def __init__(self, dim: int = 3, in_channels: int = 1, num_classes: int = 6, base_channels: int = 16,
                 depth: int = 4, deep_levels: int = 3, dropout: float = 0.3):
        super().__init__()
        _check_dim(dim)
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if deep_levels < 1:
            raise ValueError(f"deep_levels must be at least 1, got {deep_levels}")
        self.dim = dim
        self.depth = depth
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.stem = _CONV[dim](in_channels, channels[0], kernel_size=3, padding=1, bias=False)
        self.context = nn.ModuleList(ContextModule(c, dim, dropout) for c in channels)
        self.downs = nn.ModuleList(_conv_norm_act(channels[i], channels[i + 1], dim, stride=2) for i in range(depth))

        levels = list(reversed(range(depth)))   # the decoder works from level depth-1 up to level 0
        self.ups = nn.ModuleList(_conv_norm_act(channels[i + 1], channels[i], dim) for i in levels)
        self.localise = nn.ModuleList(
            nn.Sequential(_conv_norm_act(2 * channels[i], channels[i], dim),
                          _conv_norm_act(channels[i], channels[i], dim, kernel=1)) for i in levels)

        # segmentation layers on the last `deep_levels` localisation outputs, coarsest first
        self.head_levels = list(range(min(deep_levels, depth) - 1, -1, -1))
        self.heads = nn.ModuleList(_CONV[dim](channels[i], num_classes, kernel_size=1) for i in self.head_levels)
        self.aux_strides = [2 ** i for i in sorted(self.head_levels) if i > 0]   # of the maps returned as aux
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.dim() != self.dim + 2:
            raise ValueError(f"a {self.dim}D ImprovedUNet expects (B, C{', X, Y, Z' if self.dim == 3 else ', X, Y'}) "
                             f"input, got shape {tuple(x.shape)}")
        h = self.context[0](self.stem(x))
        skips = [h]
        for down, context in zip(self.downs, self.context[1:]):
            h = context(down(h))
            skips.append(h)

        localised = {}
        for up, localise, level in zip(self.ups, self.localise, reversed(range(self.depth))):
            h = F.interpolate(h, size=skips[level].shape[2:], mode="nearest")
            h = localise(torch.cat([skips[level], up(h)], dim=1))
            localised[level] = h

        total, aux = None, []
        for head, level in zip(self.heads, self.head_levels):   # from the coarsest to the full resolution
            scores = head(localised[level])
            if level > 0:
                aux.append(scores)
            total = scores if total is None else F.interpolate(total, size=scores.shape[2:], mode="nearest") + scores
        return (total, aux[::-1]) if return_aux else total


class ImprovedUNet3D(ImprovedUNet):
    """The 3D Improved U-Net of this study: 16 base channels, five levels, three segmentation layers."""

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 16, depth: int = 4,
                 deep_levels: int = 3, dropout: float = 0.3):
        super().__init__(3, in_channels, num_classes, base_channels, depth, deep_levels, dropout)


# ---------------------------------------------------------------------------
# Running the 2D baseline on a 3D volume
# ---------------------------------------------------------------------------
# The baseline is trained on single slices but evaluated on the same test volumes as the 3D
# model: every axial slice goes through the 2D network and the logits are stacked back into
# a volume, so Dice is computed in 3D, voxel for voxel, on an identical split. Stacking
# independent slices is also what makes the baseline's boundaries jagged between slices.

def volume_to_slices(volume: torch.Tensor) -> torch.Tensor:
    """(B, C, X, Y, Z) -> (B*Z, C, X, Y): every axial slice becomes one 2D sample."""
    b, c, x, y, z = volume.shape
    return volume.permute(0, 4, 1, 2, 3).reshape(b * z, c, x, y)


def slices_to_volume(slices: torch.Tensor, batch_size: int) -> torch.Tensor:
    """(B*Z, K, X, Y) -> (B, K, X, Y, Z): the inverse of volume_to_slices."""
    n, k, x, y = slices.shape
    if n % batch_size:
        raise ValueError(f"{n} slices cannot be split into {batch_size} volumes")
    z = n // batch_size
    return slices.reshape(batch_size, z, k, x, y).permute(0, 2, 3, 4, 1).contiguous()


class SliceWise(nn.Module):
    """Apply a 2D model to every axial slice of a volume and stack the logits into a volume.

    For evaluation, not for training: the slices are run in chunks of `chunk_size` to bound
    the memory (a whole volume is 128 slices). With instance norm a chunk gives exactly the
    same result as one big batch; with batch norm only in eval mode.
    """

    def __init__(self, model2d: nn.Module, chunk_size: Optional[int] = 32):
        super().__init__()
        self.model2d = model2d
        self.chunk_size = chunk_size

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        slices = volume_to_slices(volume)
        chunks = slices.split(self.chunk_size) if self.chunk_size else (slices,)
        return slices_to_volume(torch.cat([self.model2d(c) for c in chunks], dim=0), volume.shape[0])

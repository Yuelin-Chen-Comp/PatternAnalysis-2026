"""Tests for utils.py: the Dice + cross-entropy loss and the per-class Dice / IoU metrics.

Run from the project folder:  python -m pytest tests     (or: python tests/test_utils.py)
Tiny hand-checkable examples; skipped when torch is not installed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    from torch.nn import functional as F
    import utils
except ImportError:  # pragma: no cover
    torch = utils = None


class Skip(Exception):
    """Raised by a test that cannot run here (used when pytest is not installed)."""


def need_torch():
    if torch is not None:
        return
    try:
        import pytest
    except ImportError:
        raise Skip("torch not installed")
    pytest.skip("torch not installed")


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def confident_logits(target, num_classes, scale=30.0):
    """Logits that put (almost) all the probability on `target` (B, ...) -> (B, C, ...)."""
    return F.one_hot(target, num_classes).movedim(-1, 1).float() * scale


def random_target(shape, num_classes=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_classes, shape, generator=g)


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------

def test_loss_is_near_zero_for_a_perfect_prediction_and_large_for_a_wrong_one():
    need_torch()
    target = random_target((2, 8, 8, 4))
    loss = utils.DiceCELoss()
    perfect = float(loss(confident_logits(target, 6), target))
    wrong = float(loss(confident_logits((target + 1) % 6, 6), target))
    assert perfect < 1e-3, perfect
    assert wrong > 10, wrong


def test_dice_term_matches_a_hand_computation():
    need_torch()
    # 1D-like toy: 3 classes on 8 voxels; with confident logits the soft Dice equals the hard Dice
    target = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
    pred = torch.tensor([[0, 0, 1, 1, 1, 2, 0, 2]])
    # class 1: TP=2, FP=1, FN=0 -> 4/5; class 2: TP=2, FP=0, FN=1 -> 4/5
    expected = 1 - (0.8 + 0.8) / 2
    got = float(utils.DiceCELoss(ce_weight=0.0, smooth=0.0)(confident_logits(pred, 3), target))
    assert abs(got - expected) < 1e-4, (got, expected)
    # with the background included, class 0 (TP=2, FP=1, FN=1 -> 4/6) joins the average
    expected_bg = 1 - (4 / 6 + 0.8 + 0.8) / 3
    got_bg = float(utils.DiceCELoss(ce_weight=0.0, smooth=0.0, include_background=True)(confident_logits(pred, 3), target))
    assert abs(got_bg - expected_bg) < 1e-4, (got_bg, expected_bg)


def test_ce_only_equals_cross_entropy_and_weights_scale_the_terms():
    need_torch()
    torch.manual_seed(0)
    logits, target = torch.randn(2, 6, 8, 8), random_target((2, 8, 8))
    ce = F.cross_entropy(logits, target)
    assert torch.allclose(utils.DiceCELoss(dice_weight=0.0)(logits, target), ce)
    both = utils.DiceCELoss()(logits, target)
    dice_only = utils.DiceCELoss(ce_weight=0.0)(logits, target)
    assert torch.allclose(both, ce + dice_only, atol=1e-6)
    assert torch.allclose(utils.DiceCELoss(ce_weight=2.0, dice_weight=0.5)(logits, target),
                          2 * ce + 0.5 * dice_only, atol=1e-6)


def test_absent_class_does_not_produce_nan_and_costs_nothing():
    need_torch()
    target = torch.zeros(1, 8, 8, dtype=torch.long)            # only background in this patch
    value = utils.DiceCELoss()(confident_logits(target, 6), target)
    assert torch.isfinite(value) and float(value) < 1e-3


def test_a_missed_class_is_punished_even_when_it_is_tiny():
    need_torch()
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[0, :2, :2] = 5                                         # 4 of 1024 pixels are 'prostate'
    miss_it = torch.zeros_like(target)                            # predicts background everywhere
    loss_dice = utils.DiceCELoss(ce_weight=0.0)(confident_logits(miss_it, 6), target)
    loss_ce = utils.DiceCELoss(dice_weight=0.0)(confident_logits(miss_it, 6), target)
    # the Dice term sees the missing class (1 class of 5 has Dice 0 -> loss 0.2); the CE of 4 wrong
    # pixels is only 4/1024 of a large per-pixel error, which is the weakness Dice compensates
    assert abs(float(loss_dice) - 0.2) < 1e-3
    assert float(loss_dice) > 0.1


def test_gradients_flow_in_2d_and_3d_and_for_bf16_logits():
    need_torch()
    for shape in ((2, 6, 8, 8), (1, 6, 8, 8, 4)):
        logits = torch.randn(shape, requires_grad=True)
        utils.DiceCELoss()(logits, random_target((shape[0],) + shape[2:])).backward()
        assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
        assert float(logits.grad.abs().sum()) > 0
    value = utils.DiceCELoss()(torch.randn(1, 6, 8, 8).bfloat16(), random_target((1, 8, 8)))
    assert value.dtype == torch.float32 and torch.isfinite(value)


# ---------------------------------------------------------------------------
# the metrics
# ---------------------------------------------------------------------------

def test_per_class_scores_hand_example():
    need_torch()
    target = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    pred = torch.tensor([0, 0, 1, 1, 1, 2, 0, 2])
    dice, iou = utils.per_class_scores(pred, target, 3)
    # class 0: TP=2 FP=1 FN=1; class 1: TP=2 FP=1 FN=0; class 2: TP=2 FP=0 FN=1
    assert torch.allclose(dice, torch.tensor([4 / 6, 4 / 5, 4 / 5]))
    assert torch.allclose(iou, torch.tensor([2 / 4, 2 / 3, 2 / 3]))


def test_per_class_scores_perfect_and_absent_classes():
    need_torch()
    target = random_target((4, 8, 8), num_classes=3)
    dice, iou = utils.per_class_scores(target, target, 6)         # classes 3-5 never occur
    assert torch.allclose(dice[:3], torch.ones(3)) and torch.allclose(iou[:3], torch.ones(3))
    assert bool(torch.isnan(dice[3:]).all()) and bool(torch.isnan(iou[3:]).all())
    # a class that is in the ground truth but never predicted scores 0, not NaN
    dice, _ = utils.per_class_scores(torch.zeros_like(target), target, 3)
    assert float(dice[1]) == 0.0 and float(dice[2]) == 0.0


def test_per_class_scores_work_on_a_3d_volume_and_reject_shape_mismatch():
    need_torch()
    target = random_target((16, 16, 8))
    dice, iou = utils.per_class_scores(target, target, 6)
    assert tuple(dice.shape) == (6,) and bool((dice == 1).all())
    assert raises(ValueError, utils.per_class_scores, target, target[:, :, :4], 6)


def test_dice_is_never_below_iou_and_both_are_in_range():
    need_torch()
    pred, target = random_target((32, 32, 8), seed=1), random_target((32, 32, 8), seed=2)
    dice, iou = utils.per_class_scores(pred, target, 6)
    assert bool(((dice >= 0) & (dice <= 1)).all()) and bool((dice >= iou).all())
    assert bool(torch.allclose(dice, 2 * iou / (1 + iou)))        # the identity linking the two


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    skipped = 0
    for name, fn in tests:
        try:
            fn()
        except Skip as why:
            skipped += 1
            print(f"skip {name} ({why})")
        else:
            print(f"ok   {name}")
    print(f"{len(tests) - skipped} passed, {skipped} skipped")

"""Tests for modules.py: shapes, skip connections, gradient flow, slice handling, no NumPy.

Run from the project folder:  python -m pytest tests     (or: python tests/test_modules.py)
The models are tiny and run on random tensors on the CPU, so no dataset or GPU is needed.
All tests are skipped when torch is not installed.
"""
import ast
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

try:
    import torch
    import modules
except ImportError:  # pragma: no cover
    torch = modules = None


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


def tiny(dim, **kwargs):
    """A small model so every test runs in milliseconds."""
    torch.manual_seed(0)
    return modules.UNet(dim=dim, base_channels=4, depth=2, **kwargs)


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------

def test_output_shapes_2d_and_3d():
    need_torch()
    torch.manual_seed(0)
    y2 = modules.UNet2D(base_channels=4, depth=2)(torch.randn(3, 1, 32, 32))
    assert tuple(y2.shape) == (3, 6, 32, 32)
    y3 = modules.UNet3D(base_channels=4, depth=2)(torch.randn(2, 1, 16, 16, 8))
    assert tuple(y3.shape) == (2, 6, 16, 16, 8)


def test_num_classes_and_input_channels_are_respected():
    need_torch()
    y = modules.UNet3D(in_channels=2, num_classes=3, base_channels=4, depth=2)(torch.randn(1, 2, 16, 16, 8))
    assert tuple(y.shape) == (1, 3, 16, 16, 8)


def test_odd_input_sizes_keep_their_size():
    need_torch()
    # sizes that are not divisible by 2**depth: the decoder pads the upsampled map to the skip
    assert tuple(tiny(2)(torch.randn(1, 1, 30, 22)).shape) == (1, 6, 30, 22)
    assert tuple(tiny(3)(torch.randn(1, 1, 18, 14, 10)).shape) == (1, 6, 18, 14, 10)


def test_wrong_input_rank_and_bad_arguments_are_rejected():
    need_torch()
    assert raises(ValueError, tiny(3), torch.randn(1, 1, 16, 16))        # 3D model, 2D input
    assert raises(ValueError, tiny(2), torch.randn(1, 1, 16, 16, 8))     # 2D model, 3D input
    assert raises(ValueError, modules.UNet, dim=4)
    assert raises(ValueError, modules.UNet, dim=3, depth=0)
    assert raises(ValueError, modules.UNet, dim=3, norm="layer")


def test_gradients_reach_every_parameter():
    need_torch()
    for dim, shape in ((2, (2, 1, 16, 16)), (3, (1, 1, 16, 16, 8))):
        model = tiny(dim)
        x = torch.randn(*shape)
        target = torch.randint(0, 6, (shape[0],) + shape[2:])
        torch.nn.functional.cross_entropy(model(x), target).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), f"{dim}D: bad gradient for {name}"
        # the gradient has travelled the whole way from the head back to the first convolution
        assert float(model.first.block[0].weight.grad.abs().sum()) > 0


def test_skip_connections_change_the_output():
    need_torch()
    for dim, shape in ((2, (1, 1, 16, 16)), (3, (1, 1, 16, 16, 8))):
        model = tiny(dim).eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            normal = model(x)
            handles = [up.register_forward_pre_hook(lambda mod, args: (args[0], torch.zeros_like(args[1])))
                       for up in model.ups]
            without_skips = model(x)
            for h in handles:
                h.remove()
        assert not torch.allclose(normal, without_skips), f"{dim}D: the skips have no effect on the output"


def test_eval_mode_is_deterministic():
    need_torch()
    model = tiny(3).eval()
    x = torch.randn(1, 1, 16, 16, 8)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_batch_norm_option_runs():
    need_torch()
    y = tiny(2, norm="batch")(torch.randn(4, 1, 16, 16))
    assert tuple(y.shape) == (4, 6, 16, 16)


def test_count_parameters():
    need_torch()
    model = tiny(3)
    assert modules.count_parameters(model) == sum(p.numel() for p in model.parameters())
    for p in model.head.parameters():
        p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    assert modules.count_parameters(model) == total - head
    assert modules.count_parameters(model, trainable_only=False) == total


# ---------------------------------------------------------------------------
# the 2D baseline on 3D volumes
# ---------------------------------------------------------------------------

def test_volume_slices_roundtrip_and_ordering():
    need_torch()
    vol = torch.randn(2, 3, 5, 6, 4)
    slices = modules.volume_to_slices(vol)
    assert tuple(slices.shape) == (2 * 4, 3, 5, 6)
    assert torch.equal(slices[1 * 4 + 2], vol[1, :, :, :, 2]), "sample b*Z+z must be slice z of volume b"
    assert torch.equal(modules.slices_to_volume(slices, 2), vol)
    assert raises(ValueError, modules.slices_to_volume, slices, 3)


def test_slicewise_equals_running_each_slice():
    need_torch()
    model2d = tiny(2).eval()
    vol = torch.randn(2, 1, 16, 16, 5)
    with torch.no_grad():
        out = modules.SliceWise(model2d, chunk_size=3)(vol)
        assert tuple(out.shape) == (2, 6, 16, 16, 5)
        for b in range(2):
            for z in range(5):
                one = model2d(vol[b:b + 1, :, :, :, z])[0]
                assert torch.allclose(out[b, :, :, :, z], one, atol=1e-5), f"slice b={b} z={z} differs"


def test_slicewise_chunk_size_does_not_change_the_result():
    need_torch()
    model2d = tiny(2).eval()
    vol = torch.randn(1, 1, 16, 16, 7)
    with torch.no_grad():
        reference = modules.SliceWise(model2d, chunk_size=None)(vol)
        for chunk in (1, 2, 7):
            assert torch.allclose(modules.SliceWise(model2d, chunk_size=chunk)(vol), reference, atol=1e-5)


# ---------------------------------------------------------------------------
# course requirement
# ---------------------------------------------------------------------------

def test_modules_py_does_not_import_numpy():
    """The assessment spec forbids NumPy (and other non-torch libraries) in modules.py."""
    tree = ast.parse((PROJECT / "modules.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"torch", "typing"}, f"unexpected imports in modules.py: {sorted(imported)}"


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

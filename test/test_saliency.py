import itertools

import numpy as np
import pytest

from seam_carving import resize, saliency_mask, spectral_residual
from seam_carving.carve import _get_backward_seam


def test_dp_matches_exhaustive_paths():
    energy = np.random.default_rng(12).uniform(0, 10, (5, 4)).astype(np.float32)
    paths = [
        p
        for p in itertools.product(range(4), repeat=5)
        if all(abs(a - b) <= 1 for a, b in zip(p, p[1:]))
    ]
    expected = min(sum(float(energy[r, c]) for r, c in enumerate(p)) for p in paths)
    seam = _get_backward_seam(energy)
    assert sum(float(energy[r, c]) for r, c in enumerate(seam)) == expected


def test_spectral_residual():
    assert not spectral_residual(np.ones((20, 30))).any()
    # An impulse has constant Fourier amplitude; its residual reconstruction
    # must stay localized. A filled rectangle need not have a salient interior.
    src = np.zeros((64, 64, 3), dtype=np.uint8)
    src[25, 35] = 255
    sal = spectral_residual(src)
    assert sal.shape == src.shape[:2] and sal.dtype == np.float32
    assert np.isfinite(sal).all() and sal.min() == 0 and sal.max() == 1
    assert np.unravel_index(sal.argmax(), sal.shape) == (25, 35)
    assert sal[22:29, 32:39].mean() > sal[:5].mean()


@pytest.mark.parametrize("order", ["width-first", "height-first"])
@pytest.mark.parametrize("size", [(6, 7), (12, 13)])
def test_resize_dimensions(order, size):
    src = np.random.default_rng(4).integers(0, 256, (10, 9, 3), dtype=np.uint8)
    out = resize(src, size, energy_mode="saliency", order=order)
    assert out.shape == (size[1], size[0], 3) and out.dtype == src.dtype


def test_protect_flat_foreground():
    src = np.zeros((12, 16, 3), dtype=np.uint8)
    src[:, :5] = [120, 0, 0]
    sal = np.zeros((12, 16), dtype=np.float32)
    sal[:, :5] = 1
    out = resize(src, (8, 12), energy_mode="saliency", saliency_map=sal)
    np.testing.assert_array_equal(out[:, :5], src[:, :5])
    # A zero barrier must recover the unmodified baseline exactly.
    np.testing.assert_array_equal(
        resize(src, (8, 12), energy_mode="saliency", saliency_weight=0),
        resize(src, (8, 12)),
    )


@pytest.mark.parametrize("sal", [np.array([[np.nan]]), np.array([[2.0]]), np.zeros(3)])
def test_invalid_map(sal):
    with pytest.raises(ValueError):
        saliency_mask(sal)


def test_invalid_options():
    src = np.zeros((10, 10), dtype=np.uint8)
    for options in [
        dict(saliency_threshold=-0.1),
        dict(saliency_weight=-1),
        dict(saliency_map=np.zeros((3, 3))),
        dict(drop_mask=np.ones((10, 10), dtype=bool)),
    ]:
        with pytest.raises(ValueError):
            resize(src, (8, 10), energy_mode="saliency", **options)

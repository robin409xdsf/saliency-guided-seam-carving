"""Spectral-residual saliency, without pretrained models or manual object masks."""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, zoom


def spectral_residual(src: np.ndarray, analysis_size: int = 64) -> np.ndarray:
    """Return an H x W float32 saliency map in [0, 1].

    FFT -> log amplitude minus 3x3 local average -> inverse FFT with original
    phase -> squared magnitude -> Gaussian smoothing -> resize and normalize.
    RGB inputs use RGB order. Analysis uses a square low-resolution image.
    """
    src = np.asarray(src, dtype=np.float64)
    if src.ndim == 3 and src.shape[2] == 3:
        src = src @ np.array([0.2125, 0.7154, 0.0721])
    if src.ndim != 2 or src.size == 0 or not np.isfinite(src).all():
        raise ValueError("src must be a nonempty finite grayscale or RGB image")
    if not isinstance(analysis_size, int) or analysis_size < 2:
        raise ValueError("analysis_size must be an integer >= 2")
    h, w = src.shape
    if np.ptp(src) == 0:
        return np.zeros((h, w), dtype=np.float32)
    small = zoom(src, (analysis_size / h, analysis_size / w), order=1)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.maximum(np.abs(spectrum), 1e-12))
    residual = log_amplitude - uniform_filter(log_amplitude, size=3, mode="reflect")
    reconstruction = np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))
    sal = gaussian_filter(np.abs(reconstruction) ** 2, sigma=2.5, mode="reflect")
    sal = zoom(sal, (h / analysis_size, w / analysis_size), order=1)
    sal -= sal.min()
    peak = sal.max()
    if peak > 0:
        sal /= peak
    return sal.astype(np.float32)


def saliency_mask(saliency_map: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Binarize a normalized saliency map using S > threshold."""
    sal = np.asarray(saliency_map)
    if sal.ndim != 2 or sal.size == 0 or not np.isfinite(sal).all():
        raise ValueError("saliency_map must be a nonempty finite 2D array")
    if np.any(sal < 0) or np.any(sal > 1):
        raise ValueError("saliency_map must be normalized to [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("saliency_threshold must be in [0, 1]")
    return sal > threshold

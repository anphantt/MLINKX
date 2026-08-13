from __future__ import annotations
import os
from typing import Any, Dict, List, Mapping, Tuple
import numpy as np
from scipy import signal, stats
from .utils import require_2d_window
# -----------------------------------------------------------------------------
# Spectral helpers
# -----------------------------------------------------------------------------

def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freqs >= lo) & (freqs < hi)

def _welch_psd(window: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    x = require_2d_window(window)
    nperseg = min(x.shape[-1], int(sfreq * 2))
    noverlap = nperseg // 2
    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        axis=-1,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return freqs.astype(np.float32), psd.astype(np.float32)



def _band_power_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Mapping[str, Tuple[float, float]],
    log_scale: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    band_names = list(bands.keys())
    out = np.zeros((psd.shape[0], len(band_names)), dtype=np.float32)

    for b_idx, band_name in enumerate(band_names):
        mask = _band_mask(freqs, bands[band_name])
        if not np.any(mask):
            continue
        power = np.trapz(psd[:, mask], freqs[mask], axis=-1)
        # power = np.trapezoid(psd[:, mask], freqs[mask], axis=-1)
        if log_scale:
            power = np.log1p(np.maximum(power, 0.0))
        out[:, b_idx] = power.astype(np.float32)
    return out, band_names



def _relative_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Mapping[str, Tuple[float, float]],
) -> Tuple[np.ndarray, List[str]]:
    abs_power, band_names = _band_power_from_psd(freqs, psd, bands, log_scale=False)
    total_power = abs_power.sum(axis=1, keepdims=True)
    rel = abs_power / np.clip(total_power, 1e-8, None)
    return rel.astype(np.float32), band_names



def _bandpass_filter(sig_1d: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    nyq = sfreq / 2.0
    hi = min(hi, nyq - 1e-3)
    if lo <= 0 or hi <= lo:
        raise ValueError(f"Invalid band {band} for sfreq={sfreq}")
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, sig_1d).astype(np.float32)



def _analytic_phase(window: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    x = require_2d_window(window)
    phases = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[0]):
        xf = _bandpass_filter(x[ch], sfreq, band)
        phases[ch] = np.angle(signal.hilbert(xf)).astype(np.float32)
    return phases

# -----------------------------------------------------------------------------
# Node feature extractors
# -----------------------------------------------------------------------------

def feature_relative_band_power(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    values, band_names = _relative_band_power(freqs, psd, bands)
    return values, {
        "feature_names": [f"rbp_{b}" for b in band_names],
        "description": "Relative band power per channel.",
    }



def feature_absolute_band_power(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    values, band_names = _band_power_from_psd(freqs, psd, bands, log_scale=False)
    return values, {
        "feature_names": [f"abs_power_{b}" for b in band_names],
        "description": "Absolute band power per channel.",
    }



def feature_log_band_power(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    values, band_names = _band_power_from_psd(freqs, psd, bands, log_scale=True)
    return values, {
        "feature_names": [f"log_power_{b}" for b in band_names],
        "description": "Log-transformed band power per channel.",
    }



def feature_hjorth(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)

    var_x = np.var(x, axis=-1)
    var_dx = np.var(dx, axis=-1)
    var_ddx = np.var(ddx, axis=-1)

    activity = var_x
    mobility = np.sqrt(var_dx / np.clip(var_x, 1e-8, None))
    complexity = np.sqrt(var_ddx / np.clip(var_dx, 1e-8, None)) / np.clip(mobility, 1e-8, None)

    values = np.stack([activity, mobility, complexity], axis=-1).astype(np.float32)
    return values, {
        "feature_names": ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"],
        "description": "Hjorth activity, mobility, complexity per channel.",
    }



def feature_statistical(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    mean = np.mean(x, axis=-1)
    std = np.std(x, axis=-1)
    skew = stats.skew(x, axis=-1, bias=False)
    kurt = stats.kurtosis(x, axis=-1, fisher=True, bias=False)
    min_v = np.min(x, axis=-1)
    max_v = np.max(x, axis=-1)
    ptp = np.ptp(x, axis=-1)

    values = np.stack([mean, std, skew, kurt, min_v, max_v, ptp], axis=-1).astype(np.float32)
    return values, {
        "feature_names": ["mean", "std", "skew", "kurtosis", "min", "max", "ptp"],
        "description": "Basic statistical features per channel.",
    }



def feature_spectral_entropy(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    p = psd / np.clip(psd.sum(axis=-1, keepdims=True), 1e-8, None)
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1) / np.log(p.shape[-1])
    values = entropy[:, None].astype(np.float32)
    return values, {
        "feature_names": ["spectral_entropy"],
        "description": "Normalized spectral entropy per channel.",
    }



def _higuchi_fd_1d(x: np.ndarray, kmax: int = 8) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 4:
        return float("nan")

    lk = []
    xk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            ll = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (((n - m - 1) // k) * k)
            lm.append((ll * norm) / k)
        if len(lm) == 0:
            continue
        lk.append(np.mean(lm))
        xk.append(1.0 / k)

    if len(lk) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(xk), np.log(lk), deg=1)
    return float(coeffs[0])



def feature_hfd(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    values = np.array([_higuchi_fd_1d(ch) for ch in x], dtype=np.float32)[:, None]
    return values, {
        "feature_names": ["higuchi_fd"],
        "description": "Higuchi fractal dimension per channel.",
    }



def feature_wavelet_energy(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise ImportError("wavelet features require PyWavelets (pywt). Install it or remove 'wavelet_energy'.") from exc

    x = require_2d_window(window)
    level = 5
    values = []
    for ch in x:
        coeffs = pywt.wavedec(ch, wavelet="db4", level=level)
        energies = [float(np.sum(np.square(c))) for c in coeffs]
        values.append(energies)
    values = np.asarray(values, dtype=np.float32)
    names = [f"wavelet_energy_a{level}"] + [f"wavelet_energy_d{i}" for i in range(level, 0, -1)]
    return values, {
        "feature_names": names,
        "description": "Wavelet sub-band energies per channel.",
    }


# -----------------------------------------------------------------------------
# Connectivity extractors
# -----------------------------------------------------------------------------

def _symmetrize(mat: np.ndarray) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float32)



def conn_pearson(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    mat = np.corrcoef(x)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return _symmetrize(mat), {"description": "Pearson correlation connectivity.", "band_names": None}



def conn_spearman(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    ranks = np.apply_along_axis(stats.rankdata, 1, x)
    mat = np.corrcoef(ranks)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return _symmetrize(mat), {"description": "Spearman correlation connectivity.", "band_names": None}



def conn_coherence(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    band_names = list(bands.keys())
    n_channels = x.shape[0]
    out = np.zeros((len(band_names), n_channels, n_channels), dtype=np.float32)
    nperseg = min(x.shape[-1], int(sfreq * 2))
    noverlap = nperseg // 2

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh = signal.coherence(x[i], x[j], fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            for b_idx, band_name in enumerate(band_names):
                mask = _band_mask(freqs, bands[band_name])
                val = float(np.mean(coh[mask])) if np.any(mask) else 0.0
                out[b_idx, i, j] = val
                out[b_idx, j, i] = val
    return out, {"description": "Band-averaged magnitude-squared coherence.", "band_names": band_names}



def _phase_connectivity(
    window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, Tuple[float, float]],
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = require_2d_window(window)
    band_names = list(bands.keys())
    n_channels = x.shape[0]
    out = np.zeros((len(band_names), n_channels, n_channels), dtype=np.float32)

    for b_idx, band_name in enumerate(band_names):
        phases = _analytic_phase(x, sfreq, bands[band_name])
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                dphi = phases[i] - phases[j]
                if mode == "plv":
                    val = np.abs(np.mean(np.exp(1j * dphi)))
                elif mode == "pli":
                    val = np.abs(np.mean(np.sign(np.sin(dphi))))
                elif mode == "wpli":
                    im = np.sin(dphi)
                    denom = np.mean(np.abs(im))
                    val = np.abs(np.mean(im)) / max(denom, 1e-8)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                out[b_idx, i, j] = float(val)
                out[b_idx, j, i] = float(val)
    return out, {"description": f"Band-wise {mode.upper()} connectivity.", "band_names": band_names}



def conn_plv(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _phase_connectivity(window, sfreq, bands, mode="plv")



def conn_pli(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _phase_connectivity(window, sfreq, bands, mode="pli")



def conn_wpli(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _phase_connectivity(window, sfreq, bands, mode="wpli")


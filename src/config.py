from typing import Any, Dict, Tuple
from .preprocessing import (
    feature_relative_band_power,
    feature_hjorth,
    feature_statistical,
    conn_coherence,
    conn_wpli,
    # feature_spectral_entropy,
    # feature_hfd,
    # feature_wavelet_energy,
    # conn_pearson,
    # conn_spearman,
    # conn_plv,
    # conn_pli
)

# -----------------------------------------------------------------------------
# Default EEG bands
# -----------------------------------------------------------------------------
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BANDS_DICT = DEFAULT_BANDS.copy()
BANDS = list(DEFAULT_BANDS.values())
NUM_BANDS = len(BANDS)

FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "relative_band_power": {
        "fn": feature_relative_band_power,
        "description": "Relative band power per channel.",
        "kind": "node",
    },
    "hjorth": {
        "fn": feature_hjorth,
        "description": "Hjorth parameters per channel.",
        "kind": "node",
    },
    "statistical": {
        "fn": feature_statistical,
        "description": "Statistical per-channel features.",
        "kind": "node",
    },
    # "spectral_entropy": {
    #     "fn": feature_spectral_entropy,
    #     "description": "Spectral entropy per channel.",
    #     "kind": "node",
    # },
    # "higuchi_fd": {
    #     "fn": feature_hfd,
    #     "description": "Higuchi fractal dimension per channel.",
    #     "kind": "node",
    # },
    # "wavelet_energy": {
    #     "fn": feature_wavelet_energy,
    #     "description": "Wavelet energies per channel.",
    #     "kind": "node",
    # },
}


CONNECTIVITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    # "pearson": {"fn": conn_pearson, "description": "Pearson correlation.", "kind": "connectivity"},
    # "spearman": {"fn": conn_spearman, "description": "Spearman correlation.", "kind": "connectivity"},
    "coherence": {"fn": conn_coherence, "description": "Band-wise coherence.", "kind": "connectivity"},
    # "plv": {"fn": conn_plv, "description": "Band-wise phase-locking value.", "kind": "connectivity"},
    # "pli": {"fn": conn_pli, "description": "Band-wise phase-lag index.", "kind": "connectivity"},
    "wpli": {"fn": conn_wpli, "description": "Band-wise weighted phase-lag index.", "kind": "connectivity"},
}


# FEATURE_DIM_DICT = {
#                     "rbp": 5,       # Relative Band Power
#                     "hjorth": 3,    # Hjorth parameters
#                     "stats": 4,     # Statistical features
#                     "energies": 6,  # Wavelet energies
#                     "svd": 1,       # SVD entropy
#                     "zero": 1,      # Zero-crossing rate
#                     "hfd": 1        # Higuchi fractal dimension
#                     }


MONO_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]

MONOFIXEDGES = [
        ("F7", "Fp1"),
        ("F3", "Fp1"),
        ("Fp1", "Fz"), 
        ("F8", "Fp2"), 
        ("F4", "Fp2"),
        ("Fz", "Fp2"),
        ("Fp1", "Fp2"), 
        ("F7", "F3"),
        ("F3", "Fz"),
        ("F4", "Fz"), 
        ("F8", "F4"), 
        ("F3", "C3"),
        ("Fz", "Cz"),
        ("F4", "C4"), 
        ("C3", "T3"),
        ("C3", "Cz"),
        ("C4", "Cz"),
        ("C4", "T4"), 
        ("T3", "T5"),
        ("C3", "P3"),
        ("Cz", "Pz"),
        ("C4", "P4"), 
        ("T4", "T6"), 
        ("P3", "Pz"),
        ("P4", "Pz"),
        ("P4", "O2"), 
        ("P3", "O1"),
        ("T6", "O2"), 
        ("T5", "O1"),
        ("O1", "O2")
]



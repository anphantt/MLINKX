from __future__ import annotations
import argparse
import glob
import json
import logging
import math
import os
import re

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from fractions import Fraction
from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import h5py
import mne
import numpy as np
import pandas as pd
import yaml

from scipy import signal

from .utils import set_global_seed, require_2d_window, as_numpy_float32
from .config import (
    DEFAULT_BANDS,
    FEATURE_REGISTRY,
    CONNECTIVITY_REGISTRY,
)

from .caueeg.caueeg_script import (
    load_caueeg_config,
    load_caueeg_task_datasets,
)

from .caueeg.pipeline import (
    EegRandomCrop,
    EegDropChannels,
    EegToTensor,
    eeg_collate_fn,
)


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_CONFIG_PATHS = {
    "aheap": CONFIG_DIR / "aheap.yaml",
    "caueeg": CONFIG_DIR / "caueeg.yaml",
}

def resolve_project_path(path_value: str | os.PathLike) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()

def build_stats_transform(crop_length: int, latency: int, drop_idx):
    return transforms.Compose([
        EegRandomCrop(
            crop_length=crop_length,
            length_limit=10**7,
            multiple=1,
            latency=latency,
            segment_simulation=False,
            return_timing=False,
        ),
        EegDropChannels(drop_idx),
        EegToTensor(),
    ])

def compute_train_signal_stats(
    dataset_path: str,
    task: str,
    file_format: str,
    crop_length: int,
    latency: int,
    seed: int,
):
    set_global_seed(seed)
    _, _, _, drop_idx = get_caueeg_channel_info(dataset_path)

    stats_transform = build_stats_transform(
        crop_length=crop_length,
        latency=latency,
        drop_idx=drop_idx,
    )

    _, train_set, _, _ = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=stats_transform,
        verbose=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=eeg_collate_fn,
    )

    mean_sum = None
    std_sum = None
    n_count = 0

    for _ in range(5):
        for sample in train_loader:
            signal = sample["signal"]                      # [N, C, T]
            std, mean = torch.std_mean(signal, dim=-1, keepdim=True)   # [N, C, 1]

            mean_batch = mean.sum(dim=0, keepdim=True)    # [1, C, 1]
            std_batch = std.sum(dim=0, keepdim=True)      # [1, C, 1]

            if mean_sum is None:
                mean_sum = torch.zeros_like(mean_batch)
                std_sum = torch.zeros_like(std_batch)

            mean_sum += mean_batch
            std_sum += std_batch
            n_count += signal.shape[0]

    signal_mean = mean_sum / n_count
    signal_std = std_sum / n_count

    return signal_mean.detach().cpu().numpy(), signal_std.detach().cpu().numpy()

def load_or_compute_norm_stats(
    dataset_path: str,
    task: str,
    file_format: str,
    crop_length: int,
    latency: int,
    seed: int,
    norm_stats_npz: Optional[str] = None,
):
    if norm_stats_npz is not None and os.path.exists(norm_stats_npz):
        stats = np.load(norm_stats_npz)
        signal_mean = stats["signal_mean"]
        signal_std = stats["signal_std"]
        print(f"Loaded existing norm stats: {norm_stats_npz}")
        return signal_mean, signal_std

    print("Norm stats file not provided/found. Recomputing stats.")
    return compute_train_signal_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
    )


def _normalize_target_sfreq(target_sfreq: Optional[float]) -> Optional[float]:
    if target_sfreq is None:
        return None
    target = float(target_sfreq)
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"target_sampling_rate must be a positive finite number, got {target_sfreq!r}")
    return target



def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)



def _resample_window(window: np.ndarray, orig_sfreq: float, target_sfreq: Optional[float]) -> np.ndarray:
    x = require_2d_window(window)
    target = _normalize_target_sfreq(target_sfreq)
    orig = float(orig_sfreq)
    if target is None or math.isclose(orig, target, rel_tol=1e-9, abs_tol=1e-9):
        return x.astype(np.float32, copy=False)

    ratio = Fraction(target / orig).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    y = signal.resample_poly(x, up=up, down=down, axis=-1, padtype="line")

    expected_len = int(round(x.shape[-1] * target / orig))
    if y.shape[-1] > expected_len:
        y = y[..., :expected_len]
    elif y.shape[-1] < expected_len:
        pad_width = [(0, 0)] * y.ndim
        pad_width[-1] = (0, expected_len - y.shape[-1])
        y = np.pad(y, pad_width, mode="edge")

    return y.astype(np.float32, copy=False)

def _json_dumps(obj: Any) -> str:
    return json.dumps(_as_jsonable(obj), ensure_ascii=False)

def _window_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    # Chunk one window at a time to support selective loading.
    return (1,) + tuple(shape[1:])


def _to_hdf5_attr_value(value):
    if value is None:
        return _json_dumps(None)
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_hdf5_attr_value(value.item())
        return _json_dumps(value.tolist())
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _to_hdf5_attr_value(value.item())
        return _json_dumps(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def _ensure_window_dataset(
    parent: h5py.Group,
    name: str,
    n_windows: int,
    per_window_shape: Tuple[int, ...],
    attrs: Optional[Mapping[str, Any]] = None,
) -> h5py.Dataset:
    if name not in parent:
        ds = parent.create_dataset(
            name,
            shape=(n_windows,) + tuple(per_window_shape),
            dtype=np.float32,
            chunks=_window_chunks((n_windows,) + tuple(per_window_shape)),
            compression="gzip",
            compression_opts=4,
        )
        if attrs:
            for k, v in attrs.items():
                ds.attrs[k] = _to_hdf5_attr_value(v)
        return ds
    return parent[name]


def _truncate_subject_group(grp: h5py.Group, new_size: int) -> None:
    for subpath in ["windows/raw", "windows/qc", "windows/features", "windows/connectivity"]:
        parent = grp[subpath]
        for name in list(parent.keys()):
            ds = parent[name]
            data = ds[:new_size]
            attrs = dict(ds.attrs)
            del parent[name]
            if data.dtype.kind in {"S", "U", "O"}:
                new_ds = parent.create_dataset(name, data=data, dtype=_vlen_str_dtype())
            else:
                new_ds = parent.create_dataset(
                    name,
                    data=data,
                    compression="gzip" if data.ndim >= 2 else None,
                    compression_opts=4 if data.ndim >= 2 else None,
                    chunks=_window_chunks(data.shape) if data.ndim >= 2 else None,
                )
            for k, v in attrs.items():
                new_ds.attrs[k] = v

def _create_subject_group(
    h5f: h5py.File,
    subject_record: Mapping[str, Any],
    n_windows: int,
    window_shape: Tuple[int, int],
    feature_names: Sequence[str],
    connectivity_names: Sequence[str],
    bands: Mapping[str, Tuple[float, float]],
) -> h5py.Group:
    subject_id = str(subject_record["subject_id"]).replace("/", "__")
    grp = h5f.require_group(f"subjects/{subject_id}")

    meta = grp.require_group("metadata")
    meta.attrs["subject_id"] = str(subject_record["subject_id"])
    meta.attrs["label"] = int(subject_record.get("label", subject_record.get("class_id")))
    meta.attrs["class_id"] = int(subject_record.get("class_id", subject_record.get("label")))
    meta.attrs["sampling_rate"] = float(subject_record["sampling_rate"])
    meta.attrs["stored_sampling_rate"] = float(subject_record.get("stored_sampling_rate", subject_record["sampling_rate"]))
    meta.attrs["original_sampling_rate"] = float(subject_record.get("original_sampling_rate", subject_record["sampling_rate"]))
    meta.attrs["num_windows"] = int(n_windows)
    meta.attrs["num_channels"] = int(window_shape[0])
    meta.attrs["num_timepoints"] = int(window_shape[1])
    meta.attrs["stored_num_timepoints"] = int(window_shape[1])
    meta.attrs["montage_type"] = "" if subject_record.get("montage_type") is None else str(subject_record.get("montage_type"))
    meta.attrs["session_info_json"] = _json_dumps(subject_record.get("session_info"))
    meta.attrs["recording_info_json"] = _json_dumps(subject_record.get("recording_info"))

    if "channel_names" in meta:
        del meta["channel_names"]
    _write_string_dataset(meta, "channel_names", [str(x) for x in subject_record["channel_names"]])

    windows_grp = grp.require_group("windows")
    raw_grp = windows_grp.require_group("raw")
    feat_grp = windows_grp.require_group("features")
    conn_grp = windows_grp.require_group("connectivity")

    if "eeg" in raw_grp:
        del raw_grp["eeg"]
    raw_grp.create_dataset(
        "eeg",
        shape=(n_windows, window_shape[0], window_shape[1]),
        dtype=np.float32,
        chunks=_window_chunks((n_windows, window_shape[0], window_shape[1])),
        compression="gzip",
        compression_opts=4,
    )

    for name in ["segment_id", "start_sample", "end_sample"]:
        if name in raw_grp:
            del raw_grp[name]
    raw_grp.create_dataset("segment_id", shape=(n_windows,), dtype=np.int64)
    raw_grp.create_dataset("start_sample", shape=(n_windows,), dtype=np.int64)
    raw_grp.create_dataset("end_sample", shape=(n_windows,), dtype=np.int64)
    feat_grp.attrs["requested_families_json"] = _json_dumps(list(feature_names))
    conn_grp.attrs["requested_metrics_json"] = _json_dumps(list(connectivity_names))
    conn_grp.attrs["bands_json"] = _json_dumps({k: list(v) for k, v in bands.items()})

    return grp

def build_master_eeg_dataset(
    subject_records: Iterable[Mapping[str, Any]],
    output_h5_path: str | os.PathLike,
    *,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
    overwrite: bool = False,
    target_sampling_rate: Optional[float] = None,
) -> str:
    output_h5_path = str(output_h5_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)

    feature_families = list(feature_families or FEATURE_REGISTRY.keys())
    connectivity_metrics = list(connectivity_metrics or CONNECTIVITY_REGISTRY.keys())
    bands = dict(bands or DEFAULT_BANDS)
    target_sampling_rate = _normalize_target_sfreq(target_sampling_rate)
    unknown_features = sorted(set(feature_families) - set(FEATURE_REGISTRY.keys()))
    unknown_conn = sorted(set(connectivity_metrics) - set(CONNECTIVITY_REGISTRY.keys()))
    
    if unknown_features:
        raise ValueError(f"Unknown feature families: {unknown_features}")
    if unknown_conn:
        raise ValueError(f"Unknown connectivity metrics: {unknown_conn}")

    if overwrite and os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    mode = "a" if os.path.exists(output_h5_path) else "w"

    with h5py.File(output_h5_path, mode) as h5f:
        h5f.attrs["bands_json"] = _json_dumps({k: list(v) for k, v in bands.items()})
        h5f.attrs["feature_registry_json"] = _json_dumps({
            k: {kk: vv for kk, vv in v.items() if kk != "fn"}
            for k, v in FEATURE_REGISTRY.items()
        })
        h5f.attrs["connectivity_registry_json"] = _json_dumps({
            k: {kk: vv for kk, vv in v.items() if kk != "fn"}
            for k, v in CONNECTIVITY_REGISTRY.items()
        })
        h5f.attrs["storage_format"] = "HDF5"
        h5f.attrs["target_sampling_rate"] = -1.0 if target_sampling_rate is None else float(target_sampling_rate)

        for subject_record in subject_records:
            subject_id = str(subject_record["subject_id"]).replace("/", "__")
            subject_path = f"subjects/{subject_id}"
            if subject_path in h5f and not overwrite:
                continue
            if subject_path in h5f and overwrite:
                del h5f[subject_path]

            source_windows = subject_record["windows"]
            if len(source_windows) == 0:
                continue

            original_sampling_rate = float(subject_record.get("sampling_rate", 500.0))
            stored_sampling_rate = original_sampling_rate if target_sampling_rate is None else float(target_sampling_rate)
            channel_names = list(subject_record.get("channel_names") or [f"ch_{i}" for i in range(require_2d_window(source_windows[0]).shape[0])])

            processed_windows: List[Dict[str, np.ndarray]] = []
            for w in source_windows:
                qc_window = require_2d_window(w)
                stored_window = _resample_window(qc_window, original_sampling_rate, stored_sampling_rate)
                processed_windows.append({"qc_window": qc_window, "stored_window": stored_window})

            first_window = require_2d_window(processed_windows[0]["stored_window"])
            n_channels, n_time = first_window.shape
            n_windows = len(processed_windows)

            segment_ids = list(subject_record.get("segment_ids") or range(n_windows))
            start_samples = list(subject_record.get("start_samples") or [0] * n_windows)
            default_window_len = int(round(n_time * original_sampling_rate / stored_sampling_rate))
            end_samples = list(subject_record.get("end_samples") or [s + default_window_len for s in start_samples])
            if len(segment_ids) != n_windows or len(start_samples) != n_windows or len(end_samples) != n_windows:
                raise ValueError(f"Metadata lengths do not match number of windows for subject {subject_record['subject_id']}")

            subject_record = dict(subject_record)
            subject_record["sampling_rate"] = stored_sampling_rate
            subject_record["stored_sampling_rate"] = stored_sampling_rate
            subject_record["original_sampling_rate"] = original_sampling_rate
            subject_record["channel_names"] = channel_names
            grp = _create_subject_group(
                h5f,
                subject_record,
                n_windows=n_windows,
                window_shape=(n_channels, n_time),
                feature_names=feature_families,
                connectivity_names=connectivity_metrics,
                bands=bands,
            )

            raw_grp = grp["windows/raw"]
            feat_grp = grp["windows/features"]
            conn_grp = grp["windows/connectivity"]

            written_idx = 0
            for idx, prepared in enumerate(processed_windows):
                x = require_2d_window(prepared["stored_window"])
                qc_source = require_2d_window(prepared["qc_window"])
                if x.shape != (n_channels, n_time):
                    raise ValueError(
                        f"Inconsistent stored window shape for subject {subject_record['subject_id']}: expected {(n_channels, n_time)}, got {x.shape}"
                    )
                raw_grp["eeg"][written_idx] = x
                raw_grp["segment_id"][written_idx] = int(segment_ids[idx])
                raw_grp["start_sample"][written_idx] = int(start_samples[idx])
                raw_grp["end_sample"][written_idx] = int(end_samples[idx])

                for family in feature_families:
                    spec = FEATURE_REGISTRY[family]
                    values, meta = spec["fn"](x, stored_sampling_rate, bands)
                    values = as_numpy_float32(values)
                    ds = _ensure_window_dataset(
                        feat_grp,
                        family,
                        n_windows=n_windows,
                        per_window_shape=tuple(values.shape),
                        attrs={
                            "description": meta.get("description", spec["description"]),
                            "feature_names": meta.get("feature_names", []),
                            "shape_description": ["num_windows", "num_channels", "num_features"],
                        },
                    )
                    ds[written_idx] = values

                for metric in connectivity_metrics:
                    spec = CONNECTIVITY_REGISTRY[metric]
                    values, meta = spec["fn"](x, stored_sampling_rate, bands)
                    values = as_numpy_float32(values)
                    shape_desc = ["num_windows"]
                    if values.ndim == 2:
                        shape_desc += ["num_channels", "num_channels"]
                    elif values.ndim == 3:
                        shape_desc += ["num_bands", "num_channels", "num_channels"]
                    else:
                        shape_desc += [f"dim_{i}" for i in range(values.ndim)]
                    ds = _ensure_window_dataset(
                        conn_grp,
                        metric,
                        n_windows=n_windows,
                        per_window_shape=tuple(values.shape),
                        attrs={
                            "description": meta.get("description", spec["description"]),
                            "band_names": meta.get("band_names"),
                            "shape_description": shape_desc,
                        },
                    )
                    ds[written_idx] = values

                written_idx += 1

            if written_idx != n_windows:
                _truncate_subject_group(grp, written_idx)

            grp["metadata"].attrs["num_windows"] = int(written_idx)

    return output_h5_path

def extract_subject_id_from_set_path(file_path: str | os.PathLike) -> str:
    m = re.search(r"(sub-\d+)", str(file_path))
    if m is None:
        raise ValueError(f"Could not extract subject_id from path: {file_path}")
    return m.group(1)

def read_eeglab_set(
    file_path: str | os.PathLike,
    *,
    preload: bool = True,
    eeg_only: bool = True,
) -> mne.io.BaseRaw:

    logging.getLogger("mne").setLevel(logging.ERROR)
    raw = mne.io.read_raw_eeglab(str(file_path), preload=preload, verbose="ERROR")

    if eeg_only:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw.pick(picks)

    return raw

def sliding_window_indices(
    n_samples: int,
    window_samples: int,
    step_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if window_samples <= 0:
        raise ValueError(f"window_samples must be > 0, got {window_samples}")
    if step_samples <= 0:
        raise ValueError(f"step_samples must be > 0, got {step_samples}")
    if n_samples < window_samples:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)
    ends = starts + window_samples
    return starts, ends

def segment_continuous_eeg(
    eeg: np.ndarray,
    sfreq: float,
    *,
    window_sec: float,
    overlap: float,
) -> Tuple[List[np.ndarray], List[int], List[int], List[int]]:
    eeg = require_2d_window(eeg)

    if not (0.0 <= float(overlap) < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    if float(window_sec) <= 0:
        raise ValueError(f"window_sec must be > 0, got {window_sec}")

    window_samples = int(round(float(window_sec) * float(sfreq)))
    step_samples = int(round(window_samples * (1.0 - float(overlap))))

    if step_samples < 1:
        raise ValueError(
            f"Overlap too large: window_sec={window_sec}, overlap={overlap}, "
            f"window_samples={window_samples}, step_samples={step_samples}"
        )

    starts, ends = sliding_window_indices(
        n_samples=eeg.shape[1],
        window_samples=window_samples,
        step_samples=step_samples,
    )

    windows: List[np.ndarray] = []
    segment_ids: List[int] = []
    start_samples: List[int] = []
    end_samples: List[int] = []

    for seg_id, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        windows.append(eeg[:, s:e].astype(np.float32, copy=False))
        segment_ids.append(int(seg_id))
        start_samples.append(int(s))
        end_samples.append(int(e))

    return windows, segment_ids, start_samples, end_samples

def build_subject_record_from_set_file(
    file_path: str | os.PathLike,
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float,
    overlap: float,
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
    session_info: Optional[Mapping[str, Any]] = None,
    extra_recording_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    file_path = str(file_path)
    subject_id = extract_subject_id_from_set_path(file_path)

    if subject_id not in subject_label_map:
        raise KeyError(f"Missing label for subject {subject_id}")

    raw = read_eeglab_set(file_path, preload=True, eeg_only=True)

    sfreq = float(raw.info["sfreq"])
    if expected_sfreq is not None and not math.isclose(sfreq, float(expected_sfreq), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Unexpected sfreq for {subject_id}: got {sfreq}, expected {expected_sfreq}"
        )

    ch_names = list(raw.ch_names)
    if rename_channels is not None:
        ch_names = [rename_channels.get(ch, ch) for ch in ch_names]

    eeg = raw.get_data().astype(np.float32, copy=False)   # usually Volts from MNE
    windows, segment_ids, start_samples, end_samples = segment_continuous_eeg(
        eeg,
        sfreq,
        window_sec=window_sec,
        overlap=overlap,
    )

    recording_info = {
        "source_file": file_path,
        "n_channels_raw": int(eeg.shape[0]),
        "n_samples_raw": int(eeg.shape[1]),
        "duration_sec_raw": float(eeg.shape[1] / sfreq),
        "window_sec": float(window_sec),
        "overlap": float(overlap),
        "task_name": Path(file_path).stem,
        "mne_info_description": str(raw.info.get("description", "")),
    }
    if extra_recording_info is not None:
        recording_info.update({str(k): _as_jsonable(v) for k, v in extra_recording_info.items()})

    label_int = int(subject_label_map[subject_id])

    return {
        "subject_id": subject_id,
        "label": label_int,
        "class_id": label_int,
        "sampling_rate": sfreq,
        "channel_names": ch_names,
        "montage_type": "referential",
        "session_info": None if session_info is None else dict(session_info),
        "recording_info": recording_info,
        "windows": windows,
        "segment_ids": segment_ids,
        "start_samples": start_samples,
        "end_samples": end_samples,
    }

def iter_subject_records_from_set_files(
    file_paths: Sequence[str | os.PathLike],
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float,
    overlap: float,
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:

    for file_path in file_paths:
        yield build_subject_record_from_set_file(
            file_path=file_path,
            subject_label_map=subject_label_map,
            window_sec=window_sec,
            overlap=overlap,
            rename_channels=rename_channels,
            expected_sfreq=expected_sfreq,
        )


def find_set_files_under_derivatives(
    derivatives_root: str | os.PathLike,
    *,
    pattern: str = "sub-*/eeg/*.set",
) -> List[str]:
    derivatives_root = str(derivatives_root)
    paths = sorted(glob.glob(os.path.join(derivatives_root, pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No .set files found under {derivatives_root} with pattern {pattern}"
        )
    return paths

def build_master_eeg_dataset_from_set_files(
    file_paths: Sequence[str | os.PathLike],
    output_h5_path: str | os.PathLike,
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float = 4.0,
    overlap: float = 0.5,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
    overwrite: bool = False,
    target_sampling_rate: Optional[float] = None,
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
) -> str:

    subject_records = iter_subject_records_from_set_files(
        file_paths=file_paths,
        subject_label_map=subject_label_map,
        window_sec=window_sec,
        overlap=overlap,
        rename_channels=rename_channels,
        expected_sfreq=expected_sfreq,
    )

    return build_master_eeg_dataset(
        subject_records=subject_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=bands,
        overwrite=overwrite,
        target_sampling_rate=target_sampling_rate,
    )

def strip_avg_suffix(ch: str) -> str:
    ch = str(ch).strip()
    return ch[:-4] if ch.endswith("-AVG") else ch

def get_caueeg_channel_info(dataset_path: str):
    caueeg_cfg = load_caueeg_config(dataset_path)
    signal_header = list(caueeg_cfg["signal_header"])

    idx_ekg = signal_header.index("EKG")
    idx_photic = signal_header.index("Photic")
    drop_idx = [idx_ekg, idx_photic]

    eeg19_raw = [ch for ch in signal_header if ch not in ["EKG", "Photic"]]
    eeg19_clean = [strip_avg_suffix(ch) for ch in eeg19_raw]

    return caueeg_cfg, signal_header, eeg19_clean, drop_idx

def sample_sliding_window_starts(
    signal_len: int,
    crop_length: int,
    latency: int,
    overlap: float,
):
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    step = int(round(crop_length * (1.0 - overlap)))
    if step <= 0:
        raise ValueError(f"Invalid step={step}. Check crop_length={crop_length}, overlap={overlap}")

    last_start = signal_len - crop_length
    if last_start < latency:
        return []

    starts = list(range(int(latency), int(last_start) + 1, step))
    return [int(s) for s in starts]


def normalize_crop_dataset_mode(x: np.ndarray, signal_mean: np.ndarray, signal_std: np.ndarray):
    return ((x - signal_mean.squeeze(0)) / (signal_std.squeeze(0) + 1e-8)).astype(np.float32)


def build_sliding_subject_records_from_split(
    dataset,
    split_name: str,
    eeg19_clean,
    drop_idx,
    signal_mean,
    signal_std,
    crop_length: int,
    latency: int,
    overlap: float,
):

    records = []
    kept_ids = []

    for sample in dataset:
        signal = np.asarray(sample["signal"], dtype=np.float32)  # [21, T]
        serial = str(sample["serial"])
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        signal = np.delete(signal, drop_idx, axis=0)  # [19, T]

        starts = sample_sliding_window_starts(
            signal_len=signal.shape[-1],
            crop_length=crop_length,
            latency=latency,
            overlap=overlap,
        )

        if len(starts) == 0:
            continue

        windows = []
        segment_ids = []
        segment_meta = []

        for i, st in enumerate(starts):
            crop = signal[:, st:st + crop_length].astype(np.float32, copy=False)

            crop = normalize_crop_dataset_mode(crop, signal_mean, signal_std)

            windows.append(crop)
            segment_ids.append(i)
            segment_meta.append({
                "serial": serial,
                "split": split_name,
                "crop_start": int(st),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
                "step": int(round(crop_length * (1.0 - overlap))),
                "window_mode": "sliding",
                "normalization": "dataset",
            })

        sid = f"{split_name}_{serial}"

        rec = {
            "subject_id": sid,
            "label": label,
            "class_id": label,
            "sampling_rate": 200.0,
            "channel_names": eeg19_clean,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": segment_ids,
            "segment_metadata": segment_meta,
            "recording_info": {
                "serial": serial,
                "age": age,
                "split": split_name,
                "window_mode": "sliding",
                "n_sliding_windows": len(windows),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
            },
        }

        records.append(rec)
        kept_ids.append(sid)

    return records, kept_ids


def build_caueeg_sliding_master_compatible(
    dataset_path: str,
    task: str = "dementia",
    file_format: str = "feather",
    output_h5_path: str = "/data/CAUEEG/caueeg_master.h5",
    seed: int = 42,
    crop_length: int = 2000,
    latency: int = 2000,
    overlap: float = 0.5,
    feature_families=None,
    connectivity_metrics=None,
    norm_stats_npz: Optional[str] = None,
    overwrite: bool = False,
    bands=None,
    target_sampling_rate: Optional[int] = None,
):
    set_global_seed(seed)

    if feature_families is None:
        feature_families = list(FEATURE_REGISTRY.keys())

    if connectivity_metrics is None:
        connectivity_metrics = list(CONNECTIVITY_REGISTRY.keys())

    _, _, eeg19_clean, drop_idx = get_caueeg_channel_info(dataset_path)

    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    signal_mean, signal_std = load_or_compute_norm_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
        norm_stats_npz=norm_stats_npz,
    )

    train_records, train_ids = build_sliding_subject_records_from_split(
        train_set,
        "train",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
    )

    val_records, val_ids = build_sliding_subject_records_from_split(
        val_set,
        "val",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
    )

    test_records, test_ids = build_sliding_subject_records_from_split(
        test_set,
        "test",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
    )

    all_records = train_records + val_records + test_records
    resolved_bands = dict(bands or DEFAULT_BANDS)
    meta_json = output_h5_path.replace(".h5", "_meta.json")
    with open(meta_json, "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "task": task,
                "file_format": file_format,
                "seed": seed,
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
                "step": int(round(crop_length * (1.0 - overlap))),
                "window_mode": "sliding",
                "input_norm": "dataset",
                "norm_stats_npz_source": norm_stats_npz,
                "bands": {k: list(v) for k, v in resolved_bands.items()},
                "feature_families": list(feature_families),
                "connectivity_metrics": list(connectivity_metrics),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            f,
            indent=2,
        )

    np.savez(
        output_h5_path.replace(".h5", "_dataset_norm_stats.npz"),
        signal_mean=signal_mean,
        signal_std=signal_std,
    )

    build_master_eeg_dataset(
        subject_records=all_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=resolved_bands,
        overwrite=overwrite,
        target_sampling_rate=target_sampling_rate,
    )

    total_train = sum(len(r["windows"]) for r in train_records)
    total_val = sum(len(r["windows"]) for r in val_records)
    total_test = sum(len(r["windows"]) for r in test_records)
    total_all = total_train + total_val + total_test

    print(f"Saved sliding H5: {output_h5_path}")
    print(f"Saved meta: {meta_json}")
    print(f"Saved norm stats: {output_h5_path.replace('.h5', '_dataset_norm_stats.npz')}")
    print(f"Train records: {len(train_records)} | sliding windows: {total_train}")
    print(f"Val records:   {len(val_records)} | sliding windows: {total_val}")
    print(f"Test records:  {len(test_records)} | sliding windows: {total_test}")
    print(f"Total sliding windows stored: {total_all}")




def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must contain a YAML dictionary: {config_path}"
        )

    return config


def parse_name_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def get_cli_or_config(
    cli_value: Any,
    dataset_cfg: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value

    return dataset_cfg.get(key, default)


def validate_registry_names(
    names: list[str],
    registry: dict,
    argument_name: str,
) -> None:
    unknown = sorted(set(names) - set(registry.keys()))

    if unknown:
        raise ValueError(
            f"Unknown {argument_name}: {unknown}. "
            f"Available values: {sorted(registry.keys())}"
        )


def resolve_output_path(
    dataset: str,
    output_argument: str | None,
    config: dict[str, Any],
    window_sec: float,
    overlap: float,
    target_sampling_rate: int,
) -> Path:

    if output_argument is not None:
        output_path = Path(output_argument).expanduser()
    else:
        # output_root = dataset_cfg.get("output_root")
        output_root = resolve_project_path(
            config["output_root"]
        )

        if output_root is None:
            raise ValueError(
                "Either provide --output or define output_root "
                "in the dataset YAML file."
            )

        experiment_folder = (
            f"window{window_sec:g}_"
            f"overlap{overlap:g}_"
            f"fs{target_sampling_rate}"
        )

        output_path = (
            Path(output_root).expanduser()
            / experiment_folder
            / "master.h5"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path




def _encode_label_value(
    label_value: Any,
    label_to_int: Optional[Mapping[Any, int]] = None,
) -> int:

    if label_to_int is not None:
        if label_value not in label_to_int:
            raise KeyError(f"Label {label_value!r} not found in label_to_int.")
        return int(label_to_int[label_value])

    if isinstance(label_value, (int, np.integer)):
        return int(label_value)

    raise ValueError(
        "Label must already be an integer, or you must provide label_to_int "
        f"to encode non-integer labels. Got {label_value!r}"
    )

def load_subject_label_map_from_tsv(
    tsv_path: str | os.PathLike,
    *,
    subject_col: str = "participant_id",
    label_col: str = "Group",
    sep: str = "\t",
    label_to_int: Optional[Mapping[Any, int]] = None,
) -> Tuple[Dict[str, int], Dict[str, Any]]:

    df = pd.read_csv(tsv_path, sep=sep)

    if subject_col not in df.columns:
        raise KeyError(f"Missing subject column '{subject_col}' in {tsv_path}")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}' in {tsv_path}")

    raw_labels = df[label_col].tolist()

    if label_to_int is None:
        label_to_int = {"C": 0, "A": 1, "F": 2}
        
    subject_to_label: Dict[str, int] = {}
    for _, row in df.iterrows():
        sid = str(row[subject_col])
        subject_to_label[sid] = _encode_label_value(row[label_col], label_to_int)

    meta = {
        "subject_col": subject_col,
        "label_col": label_col,
        "label_to_int": dict(label_to_int),
    }
    return subject_to_label, meta


def _write_string_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    arr = np.asarray(list(values), dtype=object)
    if name in group:
        del group[name]
    group.create_dataset(name, data=arr, dtype=h5py.string_dtype(encoding="utf-8"))


def build_aheap_from_config(
    dataset_cfg: dict[str, Any],
    output_h5_path: Path,
    window_sec: float,
    overlap: float,
    feature_families: list[str],
    connectivity_metrics: list[str],
    target_sampling_rate: int,
    overwrite: bool,
) -> None:
    derivatives_root = resolve_project_path(
        dataset_cfg["derivatives_root"]
    )

    participants_tsv = resolve_project_path(
        dataset_cfg["participants_tsv"]
    )

    if not derivatives_root.exists():
        raise FileNotFoundError(
            f"AHEAP derivatives directory not found: "
            f"{derivatives_root}"
        )

    if not participants_tsv.exists():
        raise FileNotFoundError(
            f"AHEAP participants TSV not found: "
            f"{participants_tsv}"
        )

    subject_label_map, label_meta = (
        load_subject_label_map_from_tsv(
            participants_tsv,
            subject_col=dataset_cfg.get(
                "subject_col",
                "participant_id",
            ),
            label_col=dataset_cfg.get(
                "label_col",
                "Group",
            ),
            sep=dataset_cfg.get("separator", "\t"),
            label_to_int=dataset_cfg["label_to_int"],
        )
    )

    file_paths = find_set_files_under_derivatives(
        derivatives_root=derivatives_root,
        pattern=dataset_cfg.get(
            "file_pattern",
            "sub-*/eeg/*.set",
        ),
    )

    if not file_paths:
        raise RuntimeError(
            f"No .set files found under {derivatives_root}"
        )

    print(f"Dataset: AHEAP")
    print(f"Number of EEG files: {len(file_paths)}")
    print(f"Number of labeled subjects: {len(subject_label_map)}")
    print(f"Label metadata: {label_meta}")

    build_master_eeg_dataset_from_set_files(
        file_paths=file_paths,
        output_h5_path=str(output_h5_path),
        subject_label_map=subject_label_map,
        window_sec=window_sec,
        overlap=overlap,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=DEFAULT_BANDS,
        overwrite=overwrite,
        target_sampling_rate=target_sampling_rate,
        rename_channels=dataset_cfg.get("rename_channels"),
        expected_sfreq=dataset_cfg.get("source_sampling_rate"),
    )


def build_caueeg_from_config(
    dataset_cfg: dict[str, Any],
    output_h5_path: Path,
    window_sec: float,
    overlap: float,
    feature_families: list[str],
    connectivity_metrics: list[str],
    target_sampling_rate: int,
    overwrite: bool,
    seed: int,
) -> None:

    dataset_path = resolve_project_path(
        dataset_cfg["dataset_path"]
    )
    norm_stats_npz = dataset_cfg.get("norm_stats_npz")

    if norm_stats_npz is not None:
        norm_stats_npz = str(
            resolve_project_path(norm_stats_npz)
        )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"CAUEEG dataset directory not found: "
            f"{dataset_path}"
        )

    source_sampling_rate = int(
        dataset_cfg["source_sampling_rate"]
    )

    crop_length_samples = int(
        round(window_sec * source_sampling_rate)
    )

    latency_sec = float(
        dataset_cfg.get("latency_sec", 0.0)
    )

    latency_samples = int(
        round(latency_sec * source_sampling_rate)
    )

    print("Dataset: CAUEEG")
    print(f"Window: {window_sec:g} seconds")
    print(f"Window samples: {crop_length_samples}")
    print(f"Initial latency: {latency_sec:g} seconds")
    print(f"Output: {output_h5_path}")

    build_caueeg_sliding_master_compatible(
        dataset_path=str(dataset_path),
        task=dataset_cfg["task"],
        file_format=dataset_cfg.get("file_format","edf"),
        output_h5_path=str(output_h5_path),
        seed=seed,
        crop_length=crop_length_samples,
        latency=latency_samples,
        overlap=overlap,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        norm_stats_npz=norm_stats_npz,
        overwrite=overwrite,
        bands=DEFAULT_BANDS,
        target_sampling_rate=target_sampling_rate,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an EEG HDF5 file from AHEAP or CAUEEG."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DEFAULT_CONFIG_PATHS.keys()),
        help="Dataset to preprocess.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional custom YAML file. By default, the script "
            "uses configs/<dataset>.yaml."
        ),
    )

    parser.add_argument(
        "--window-sec",
        "--window_sec",
        dest="window_sec",
        type=float,
        default=None,
        help="Segment duration in seconds.",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Overlap ratio, such as 0.5.",
    )

    parser.add_argument(
        "--feature-families",
        "--feature_families",
        dest="feature_families",
        type=str,
        default=None,
        help=(
            "Comma-separated feature families, for example "
            "'relative_band_power,hjorth'."
        ),
    )

    parser.add_argument(
        "--connectivity-metrics",
        "--connectivity_metrics",
        dest="connectivity_metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated connectivity metrics, for example "
            "'coherence,wpli'."
        ),
    )

    parser.add_argument(
        "--target-sampling-rate",
        "--target_sampling_rate",
        dest="target_sampling_rate",
        type=int,
        default=None,
        help="Target sampling rate after resampling.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Exact output HDF5 path. If omitted, output_root "
            "from the YAML file is used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite an existing HDF5 file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config_path = (
        Path(args.config).expanduser()
        if args.config is not None
        else DEFAULT_CONFIG_PATHS[args.dataset]
    )

    dataset_cfg = load_yaml_config(config_path)

    configured_dataset = dataset_cfg.get("dataset")

    if configured_dataset is not None:
        configured_dataset = str(configured_dataset).lower()

        if configured_dataset != args.dataset:
            raise ValueError(
                f"CLI dataset is '{args.dataset}', but config "
                f"declares dataset '{configured_dataset}'."
            )

    window_sec = float(
        get_cli_or_config(
            args.window_sec,
            dataset_cfg,
            "window_sec",
            4.0,
        )
    )

    overlap = float(
        get_cli_or_config(
            args.overlap,
            dataset_cfg,
            "overlap",
            0.5,
        )
    )

    source_sampling_rate = int(
        dataset_cfg["source_sampling_rate"]
    )

    target_sampling_rate = int(
        get_cli_or_config(
            args.target_sampling_rate,
            dataset_cfg,
            "target_sampling_rate",
            source_sampling_rate,
        )
    )

    feature_families = parse_name_list(
        args.feature_families
        if args.feature_families is not None
        else dataset_cfg.get(
            "feature_families",
            list(FEATURE_REGISTRY.keys()),
        )
    )

    connectivity_metrics = parse_name_list(
        args.connectivity_metrics
        if args.connectivity_metrics is not None
        else dataset_cfg.get(
            "connectivity_metrics",
            list(CONNECTIVITY_REGISTRY.keys()),
        )
    )

    overwrite = bool(
        get_cli_or_config(
            args.overwrite,
            dataset_cfg,
            "overwrite",
            False,
        )
    )

    seed = int(
        get_cli_or_config(
            args.seed,
            dataset_cfg,
            "seed",
            42,
        )
    )

    if window_sec <= 0:
        raise ValueError("window_sec must be positive.")

    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    if target_sampling_rate <= 0:
        raise ValueError(
            "target_sampling_rate must be positive."
        )

    validate_registry_names(
        feature_families,
        FEATURE_REGISTRY,
        "feature families",
    )

    validate_registry_names(
        connectivity_metrics,
        CONNECTIVITY_REGISTRY,
        "connectivity metrics",
    )

    output_h5_path = resolve_output_path(
        dataset=args.dataset,
        output_argument=args.output,
        config=dataset_cfg,
        window_sec=window_sec,
        overlap=overlap,
        target_sampling_rate=target_sampling_rate,
    )

    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Window length: {window_sec:g} seconds")
    print(f"Overlap: {overlap:g}")
    print(f"Target sampling rate: {target_sampling_rate} Hz")
    print(f"Feature families: {feature_families}")
    print(f"Connectivity metrics: {connectivity_metrics}")
    print(f"Output: {output_h5_path}")
    print("=" * 60)

    if output_h5_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_h5_path}. "
            "Use --overwrite to replace it."
        )

    if args.dataset == "aheap":
        build_aheap_from_config(
            dataset_cfg=dataset_cfg,
            output_h5_path=output_h5_path,
            window_sec=window_sec,
            overlap=overlap,
            feature_families=feature_families,
            connectivity_metrics=connectivity_metrics,
            target_sampling_rate=target_sampling_rate,
            overwrite=overwrite,
        )

    elif args.dataset == "caueeg":
        build_caueeg_from_config(
            dataset_cfg=dataset_cfg,
            output_h5_path=output_h5_path,
            window_sec=window_sec,
            overlap=overlap,
            feature_families=feature_families,
            connectivity_metrics=connectivity_metrics,
            target_sampling_rate=target_sampling_rate,
            overwrite=overwrite,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}"
        )

    print(f"Finished building: {output_h5_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import torch
import json
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import h5py
import warnings
from collections import Counter, defaultdict


BANK_ENCODERS = {
    "linkx_bank",
    "cnn_bank",
    "gnn_bank"
}

MULTIBAND_ENCODERS = {}

def nullable_int(val):
    if val.lower() == "none":
        return None
    return int(val)


def as_numpy_float32(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)

def require_2d_window(window: np.ndarray) -> np.ndarray:
    arr = as_numpy_float32(window)
    if arr.ndim != 2:
        raise ValueError(f"EEG window must have shape [channels, time], got {arr.shape}")
    return arr

def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seeds for python, numpy, torch, and optionally enforce deterministic behavior.
    Call this once at the beginning of each run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For CUDA matmul determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")


# =========================================================
# Basic H5 helpers
# =========================================================

def _safe_subject_id(subject_id: Any) -> str:
    return str(subject_id).replace("/", "__")


def _decode_str_list(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _iter_existing_subject_ids(h5f: h5py.File, subject_ids: Optional[Sequence[str]]) -> List[str]:
    all_ids = list(h5f["subjects"].keys())
    if subject_ids is None:
        return all_ids
    wanted = [_safe_subject_id(sid) for sid in subject_ids]
    return [sid for sid in wanted if sid in all_ids]


def _require_3d_feature_tensor(x: np.ndarray, name: str = "feature tensor") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [num_windows, num_channels, num_features], got {x.shape}")
    return x


# =========================================================
# H5 payload loader
# =========================================================

def load_h5_payload_for_subjects(
    h5_path: str,
    subject_ids: Optional[Sequence[str]] = None,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    connectivity_band: Optional[int | str] = None,
    load_raw_for_alignment: bool = False,
    load_bad_segment_flag: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load a subject-indexed payload from the master H5.

    Returns
    -------
    payload : dict
        payload[sid] = {
            "label": int,
            "segment_id": np.ndarray [W],
            "start_sample": np.ndarray [W],
            "end_sample": np.ndarray [W],
            "channel_names": list[str],
            "features": {family: np.ndarray [W, C, F_family]},
            "connectivity": {metric: np.ndarray [W, C, C] or [W, B, C, C]},
            "raw_eeg": None or np.ndarray [W, C, T],
            "bad_segment_flag": optional np.ndarray [W],
            "aligned_raw_eeg": None,
            "aligned_adj": None,
        }
    """
    payload: Dict[str, Dict[str, Any]] = {}

    with h5py.File(h5_path, "r") as h5f:
        for sid in _iter_existing_subject_ids(h5f, subject_ids):
            # if sid in {"train_00587", "train_00781", "train_01301"}:
            #     continue
            grp = h5f[f"subjects/{sid}"]

            # print("feature keys:", list(grp["windows/features"].keys()))
            # print("connectivity keys:", list(grp["windows/connectivity"].keys()))

            entry: Dict[str, Any] = {
                "label": int(grp["metadata"].attrs["label"]),
                "segment_id": grp["windows/raw/segment_id"][:].astype(np.int64),
                "start_sample": grp["windows/raw/start_sample"][:].astype(np.int64),
                "end_sample": grp["windows/raw/end_sample"][:].astype(np.int64),
                "channel_names": _decode_str_list(grp["metadata/channel_names"][:]),
                "features": {},
                "connectivity": {},
                "raw_eeg": None,
                "aligned_raw_eeg": None,
                "aligned_adj": None,
            }
            # print("feature_families", feature_families)
            for family in feature_families or []:
                # print(family)
                x = np.asarray(grp[f"windows/features/{family}"][:], dtype=np.float32)
                entry["features"][family] = x

            for metric in connectivity_metrics or []:
                adj = np.asarray(grp[f"windows/connectivity/{metric}"][:], dtype=np.float32)

                # Optional band slicing for banded connectivity:
                # stored shape is often [W, B, C, C]
                if connectivity_band is not None and adj.ndim == 4:
                    ds = grp[f"windows/connectivity/{metric}"]
                    band_names = list(ds.attrs.get("band_names", []))
                    if isinstance(connectivity_band, str):
                        decoded_band_names = _decode_str_list(band_names)
                        if connectivity_band not in decoded_band_names:
                            raise KeyError(
                                f"Band '{connectivity_band}' not found for metric '{metric}'. "
                                f"Available: {decoded_band_names}"
                            )
                        band_idx = decoded_band_names.index(connectivity_band)
                    else:
                        band_idx = int(connectivity_band)
                    adj = adj[:, band_idx]  # -> [W, C, C]

                entry["connectivity"][metric] = adj

            if load_raw_for_alignment:
                entry["raw_eeg"] = np.asarray(grp["windows/raw/eeg"][:], dtype=np.float32)

            if load_bad_segment_flag:
                qpath = "windows/qc/bad_segment_flag"
                if qpath in grp:
                    entry["bad_segment_flag"] = grp[qpath][:].astype(np.int64)

            payload[sid] = entry

    return payload


def load_bank_specs(bank_specs_json: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if bank_specs_json is None or str(bank_specs_json).strip() == "":
        return None
    text = str(bank_specs_json).strip()
    if os.path.isfile(text):
        with open(text, "r") as f:
            obj = json.load(f)
    else:
        obj = json.loads(text)
    if not isinstance(obj, list):
        raise ValueError("bank_specs_json must decode to a list of candidate spec dictionaries.")
    return [dict(x) for x in obj]

def aheap_get_paths(data_dir, tsv_path, num_class = 'all3'):
    df = pd.read_table(tsv_path)

    if num_class == 'all2':
        label_mapping = {"A": 1, "F":1, "C": 0}
    elif num_class == 'all3':
        label_mapping = {"A": 1, "F":2, "C": 0}
    elif num_class == 'adhc':
        label_mapping = {"A": 1, "C": 0}
        df = df[~(df['Group'] == 'F')]
    elif num_class == 'ftdhc':
        label_mapping = {"F":1, "C": 0}
        df = df[~(df['Group'] == 'A')]
    elif num_class == 'adftd':
        label_mapping = {"A": 1, "F":0}
        df = df[~(df['Group'] == 'C')]
    else:
        raise ValueError(f"Invalid num_class '{num_class}'.")

    participant_labels = {row['participant_id']: label_mapping[row['Group']] for _, row in df.iterrows()}
    data_paths = []
    labels = []
    sub_id_list = []

    for sub_id in df['participant_id'].tolist():
        sub_path = os.path.join(data_dir, sub_id, 'eeg', f"{sub_id}_task-eyesclosed_eeg.set")
        if os.path.exists(sub_path):
            label = participant_labels.get(sub_id, -1)
            if label != -1:
                data_paths.append(sub_path)
                labels.append(label)
                sub_id_list.append(sub_id)
            else:
                print(f"Warning: Label for {sub_id} not found. Skipping.")
    # print(f"Total number of file paths: {len(data_paths)}")
    return data_paths, labels, sub_id_list

def get_class(class_set, dataset):
    if dataset == 'aheap':
        if class_set == 'all2':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["Healthy", "AD"]

        elif class_set == 'all3':
            num_classes = 3
            class_labels = [0,1,2]
            class_names = ["Healthy", "AD", "FTD"]

        elif class_set == 'adhc':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["Healthy", "AD"]

        elif class_set == 'ftdhc':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["Healthy", "FTD"]

        elif class_set == 'adftd':
            num_classes = 2
            class_labels = [0,1]
            class_names = ["FTD", "AD"]
        else:
            raise ValueError(f"Invalid set_class '{class_set}', only [all2, all3, adhc, ftdhc, adftd]")
    elif dataset == 'caueeg':
        num_classes = 3
        class_labels = [0,1,2]
        class_names = ["Healthy", "Dementia", "MCI"]
    
    elif dataset == 'dryad':
        num_classes = 3
        class_labels = [0,1,2]
        class_names = ["Healthy", "AD", "MCI"]
    else:
        raise ValueError(f"Invalid dataset!")
    return num_classes, class_labels, class_names


def set_reproducible(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g    

def _safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_")

def _as_float_cpu_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)

def is_bank_encoder(encoder_type: str) -> bool:
    return str(encoder_type).lower() in BANK_ENCODERS


def is_multiband_encoder(encoder_type: str) -> bool:
    return str(encoder_type).lower() in MULTIBAND_ENCODERS


def collect_required_connectivity_metrics(
    bank_specs,
    default_connectivity_metric: str,
):
    metrics = {str(default_connectivity_metric)}
    if bank_specs is None:
        return sorted(metrics)

    for spec in bank_specs:
        m = spec.get("connectivity_metric", default_connectivity_metric)
        metrics.add(str(m))
    return sorted(metrics)
    

def normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set:
    """
    Convert fixed_edges into a set of sorted integer node pairs.
    Supports:
      - integer edges: [(0,1), (1,2)]
      - channel-name edges: [("Fp1","F3"), ("F3","C3")]
    """
    if fixed_edges is None:
        return set()

    fixed_pairs = set()
    name_to_idx = None

    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names has length {len(channel_names)} but n_channels={n_channels}"
            )
        name_to_idx = {name: i for i, name in enumerate(channel_names)}

    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            if name_to_idx is None:
                raise ValueError(
                    "fixed_edges contains channel names, but channel_names was not provided."
                )
            if u not in name_to_idx or v not in name_to_idx:
                continue
            i, j = name_to_idx[u], name_to_idx[v]
        else:
            i, j = int(u), int(v)

        if i == j:
            continue
        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(u, v)} is out of range for {n_channels} nodes.")

        fixed_pairs.add(tuple(sorted((i, j))))

    return fixed_pairs


def balanced_kfold_split(sub_id_list, labels, random_term = 42, k=10):
    sub_id_list = np.array(sub_id_list)
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_term)
    all_folds = []
    for fold_idx, (_, test_idx) in enumerate(skf.split(sub_id_list, labels)):
        fold_subjects = sub_id_list[test_idx]
        fold_labels = labels[test_idx]
        all_folds.append(fold_subjects.tolist())
        class_counts = Counter(fold_labels)
        class_str = ", ".join([f"Class {cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
    return all_folds


def stratified_split_subjects(train_subjects, subject_label_map, val_ratio=0.1, seed=42):
    sids = np.array(list(train_subjects))
    y = np.array([subject_label_map[sid] for sid in sids])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(sids, y))

    new_train_subjects = set(sids[train_idx].tolist())
    val_subjects = set(sids[val_idx].tolist())
    return new_train_subjects, val_subjects


# def save_history_csv(history, csv_path):
#     df = pd.DataFrame(history)
#     df.to_csv(csv_path, index=False)
#     print(f"Saved history: {csv_path}")
#     return df

def make_jsonable(x):
    """Convert numpy/torch objects into JSON-safe Python objects."""
    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [make_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x

def normalize_summary_row(row):
    """Prepare one summary row for CSV storage."""
    row = make_jsonable(row)

    # Store list-like config as a stable string
    if isinstance(row.get("feature_families"), list):
        row["feature_families"] = ",".join(map(str, row["feature_families"]))

    # Store confusion matrix as JSON string in CSV
    if "confusion_matrix" in row:
        row["confusion_matrix_json"] = json.dumps(row["confusion_matrix"])
        del row["confusion_matrix"]

    return row

def aggregate_fold_summaries_to_seed(fold_rows, seed=None):
    """
    Convert k fold-level summary_test rows into one seed-level row.

    This is a fallback when you only have fold summaries, not raw OOF predictions.
    """
    rows = [normalize_summary_row(r) for r in fold_rows]
    df = pd.DataFrame(rows)

    # In case your fold rows use test_accuracy style names
    rename_map = {
        "test_accuracy": "accuracy",
        "test_balanced_accuracy": "balanced_accuracy",
        "test_macro_f1": "macro_f1",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Prefer weighted mean if you have fold test size.
    # Otherwise normal mean is fine if folds are approximately equal.
    possible_weight_cols = [
        "num_subjects",
        "test_num_subjects",
        "n_test",
        "num_samples",
    ]
    weight_col = next((c for c in possible_weight_cols if c in df.columns), None)

    seed_row = {}

    # Copy stable config columns from the first row
    skip_cols = set(metric_cols + [
        "fold",
        "confusion_matrix",
        "confusion_matrix_json",
    ])

    for c in df.columns:
        if c in skip_cols:
            continue
        vals = df[c].dropna().unique()
        if len(vals) == 1:
            seed_row[c] = vals[0]

    if seed is not None:
        seed_row["seed"] = int(seed)
        seed_row["split_seed"] = int(seed)
    elif "split_seed" in df.columns and df["split_seed"].nunique() == 1:
        seed_row["seed"] = int(df["split_seed"].iloc[0])

    seed_row["num_folds"] = int(len(df))

    for m in metric_cols:
        values = df[m].astype(float).to_numpy()

        if weight_col is not None:
            weights = df[weight_col].astype(float).to_numpy()
            seed_row[m] = float(np.average(values, weights=weights))
        else:
            seed_row[m] = float(np.mean(values))

        seed_row[f"{m}_fold_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        seed_row[f"{m}_fold_min"] = float(np.min(values))
        seed_row[f"{m}_fold_max"] = float(np.max(values))

    # Sum fold confusion matrices into one seed-level confusion matrix
    if "confusion_matrix_json" in df.columns:
        cms = []
        for s in df["confusion_matrix_json"].dropna():
            cms.append(np.asarray(json.loads(s), dtype=int))

        if len(cms) > 0:
            seed_row["confusion_matrix"] = np.stack(cms, axis=0).sum(axis=0).tolist()

    elif "confusion_matrix" in df.columns:
        cms = []
        for cm in df["confusion_matrix"].dropna():
            cms.append(np.asarray(cm, dtype=int))

        if len(cms) > 0:
            seed_row["confusion_matrix"] = np.stack(cms, axis=0).sum(axis=0).tolist()

    return seed_row

def save_seed_aggregation(summary_rows, output_dir):
    """
    Save:
      1) all_seed_results.csv      : one row per seed
      2) aggregate_seed_results.csv: mean/std/min/max/count across seeds
      3) aggregate_confusion_matrix.json
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = [normalize_summary_row(r) for r in summary_rows]
    df = pd.DataFrame(rows)

    raw_path = os.path.join(output_dir, "all_seed_results.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # These define one experimental variant.
    variant_cols = [
        "encoder_type",
        "training_approach",
        "mil_pool_type",
        "feature_families",
        "topology",
        "connectivity_metric",
        "connectivity_band",
        "edge_mode",
        "base_k",
        "batch_size",
        "epochs",
        "patience",
        "start_epoch",
        "lr",
        "dropout",
        "weight_decay",
        "graph_emb_dim",
        "attn_dim",
    ]
    variant_cols = [c for c in variant_cols if c in df.columns]

    agg = (
        df.groupby(variant_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    # Flatten multi-index columns
    agg.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in agg.columns
    ]

    # Add readable mean ± std columns
    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_mean_std"] = agg.apply(
                lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
                if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
                axis=1,
            )

    agg_path = os.path.join(output_dir, "aggregate_seed_results.csv")
    agg.to_csv(agg_path, index=False)

    # Aggregate confusion matrices separately
    cm_path = None
    if "confusion_matrix_json" in df.columns:
        cms = []
        for s in df["confusion_matrix_json"].dropna():
            cms.append(np.asarray(json.loads(s), dtype=float))

        if len(cms) > 0:
            cm_stack = np.stack(cms, axis=0)
            cm_info = {
                "num_seeds": int(len(cms)),
                "confusion_matrix_sum": cm_stack.sum(axis=0).astype(int).tolist(),
                "confusion_matrix_mean": cm_stack.mean(axis=0).tolist(),
                "confusion_matrix_std": cm_stack.std(axis=0, ddof=1).tolist()
                if len(cms) > 1 else np.zeros_like(cm_stack[0]).tolist(),
            }

            cm_path = os.path.join(output_dir, "aggregate_confusion_matrix.json")
            with open(cm_path, "w") as f:
                json.dump(cm_info, f, indent=2)

    print(f"Saved per-seed results: {raw_path}")
    print(f"Saved aggregate results: {agg_path}")
    if cm_path is not None:
        print(f"Saved aggregate confusion matrix: {cm_path}")

    return df, agg
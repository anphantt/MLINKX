from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from typing import Any, Dict, List, Optional, Sequence, Tuple
import h5py
import warnings
from collections import Counter, defaultdict
import re

EdgeSpec = Sequence[Tuple[int | str, int | str]]

BANK_ENCODERS = {
    "linkx_bank",
    "cnn_bank",
    "gnn_bank"
}

# MULTIBAND_ENCODERS = {}

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

def set_global_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )

# def set_global_seed(seed: int = 42, deterministic: bool = True):
#     """
#     Set seeds for python, numpy, torch, and optionally enforce deterministic behavior.
#     Call this once at the beginning of each run.
#     """
#     os.environ["PYTHONHASHSEED"] = str(seed)

#     # For CUDA matmul determinism
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#     random.seed(seed)
#     np.random.seed(seed)

#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         torch.use_deterministic_algorithms(True)
#     else:
#         torch.backends.cudnn.deterministic = False
#         torch.backends.cudnn.benchmark = True

#     np.seterr(all="ignore")
#     warnings.filterwarnings("ignore")


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

# def aheap_get_paths(data_dir, tsv_path, num_class = 'all3'):
#     df = pd.read_table(tsv_path)

#     if num_class == 'all2':
#         label_mapping = {"A": 1, "F":1, "C": 0}
#     elif num_class == 'all3':
#         label_mapping = {"A": 1, "F":2, "C": 0}
#     elif num_class == 'adhc':
#         label_mapping = {"A": 1, "C": 0}
#         df = df[~(df['Group'] == 'F')]
#     elif num_class == 'ftdhc':
#         label_mapping = {"F":1, "C": 0}
#         df = df[~(df['Group'] == 'A')]
#     elif num_class == 'adftd':
#         label_mapping = {"A": 1, "F":0}
#         df = df[~(df['Group'] == 'C')]
#     else:
#         raise ValueError(f"Invalid num_class '{num_class}'.")

#     participant_labels = {row['participant_id']: label_mapping[row['Group']] for _, row in df.iterrows()}
#     data_paths = []
#     labels = []
#     sub_id_list = []

#     for sub_id in df['participant_id'].tolist():
#         sub_path = os.path.join(data_dir, sub_id, 'eeg', f"{sub_id}_task-eyesclosed_eeg.set")
#         if os.path.exists(sub_path):
#             label = participant_labels.get(sub_id, -1)
#             if label != -1:
#                 data_paths.append(sub_path)
#                 labels.append(label)
#                 sub_id_list.append(sub_id)
#             else:
#                 print(f"Warning: Label for {sub_id} not found. Skipping.")
#     # print(f"Total number of file paths: {len(data_paths)}")
#     return data_paths, labels, sub_id_list

# def get_class(class_set, dataset):
#     if dataset == 'aheap':
#         if class_set == 'all2':
#             num_classes = 2
#             class_labels = [0,1]
#             class_names = ["Healthy", "AD"]

#         elif class_set == 'all3':
#             num_classes = 3
#             class_labels = [0,1,2]
#             class_names = ["Healthy", "AD", "FTD"]

#         elif class_set == 'adhc':
#             num_classes = 2
#             class_labels = [0,1]
#             class_names = ["Healthy", "AD"]

#         elif class_set == 'ftdhc':
#             num_classes = 2
#             class_labels = [0,1]
#             class_names = ["Healthy", "FTD"]

#         elif class_set == 'adftd':
#             num_classes = 2
#             class_labels = [0,1]
#             class_names = ["FTD", "AD"]
#         else:
#             raise ValueError(f"Invalid set_class '{class_set}', only [all2, all3, adhc, ftdhc, adftd]")
#     elif dataset == 'caueeg':
#         num_classes = 3
#         class_labels = [0,1,2]
#         class_names = ["Healthy", "Dementia", "MCI"]
    
#     else:
#         raise ValueError(f"Invalid dataset!")
#     return num_classes, class_labels, class_names


# def set_reproducible(seed: int):
#     seed = int(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False
#     torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def summarize_graph_pool(graphs: Sequence[Data], name: str) -> Dict[str, Any]:
    subject_to_count: Dict[str, int] = defaultdict(int)
    label_to_subjects: Dict[int, set] = defaultdict(set)
    for g in graphs:
        sid = str(g.subject_id)
        y = int(g.y.view(-1)[0].item())
        subject_to_count[sid] += 1
        label_to_subjects[y].add(sid)
    counts = np.asarray(list(subject_to_count.values()), dtype=np.int64)
    info = {
        "num_graphs": int(len(graphs)),
        "num_subjects": int(len(subject_to_count)),
        "segments_min": int(counts.min()) if len(counts) else 0,
        "segments_mean": float(counts.mean()) if len(counts) else 0.0,
        "segments_max": int(counts.max()) if len(counts) else 0,
        "subjects_per_label": {int(k): int(len(v)) for k, v in label_to_subjects.items()},
    }
    print(f"\n[{name}] {info}")
    return info

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


# def is_multiband_encoder(encoder_type: str) -> bool:
#     return str(encoder_type).lower() in MULTIBAND_ENCODERS
def resolve_feature_families(
    dataset: str,
    value: str | None,
) -> list[str]:
    if value:
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    if dataset == "aheap":
        return [
            "relative_band_power",
            "hjorth",
        ]

    if dataset == "caueeg":
        return [
            "relative_band_power",
            "statistical",
        ]

    raise ValueError(f"Unsupported dataset: {dataset}")

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

    # new_train_subjects = set(sids[train_idx].tolist())
    # val_subjects = set(sids[val_idx].tolist())

    new_train_subjects = sids[train_idx].tolist()
    val_subjects = sids[val_idx].tolist()

    return new_train_subjects, val_subjects


def make_jsonable(x):
    """Convert numpy/torch objects into JSON-safe Python objects."""
    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [make_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x

def save_json(path: str | os.PathLike, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            make_jsonable(payload),
            file,
            indent=2,
            ensure_ascii=False,
        )

# def normalize_summary_row(row):
#     """Prepare one summary row for CSV storage."""
#     row = make_jsonable(row)
#     # Store list-like config as a stable string
#     if isinstance(row.get("feature_families"), list):
#         row["feature_families"] = ",".join(map(str, row["feature_families"]))
#     # Store confusion matrix as JSON string in CSV
#     if "confusion_matrix" in row:
#         row["confusion_matrix_json"] = json.dumps(row["confusion_matrix"])
#         del row["confusion_matrix"]
#     return row

def normalize_summary_row(row):
    normalized = {}

    for key, value in row.items():
        if key == "confusion_matrix":
            normalized["confusion_matrix_json"] = json.dumps(
                make_jsonable(value)
            )
        elif isinstance(value, (list, tuple, dict)):
            normalized[key] = json.dumps(
                make_jsonable(value),
                sort_keys=True,
            )
        elif isinstance(value, np.generic):
            normalized[key] = value.item()
        elif torch.is_tensor(value):
            normalized[key] = json.dumps(
                value.detach().cpu().tolist()
            )
        else:
            normalized[key] = value

    return normalized

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

# def save_seed_aggregation(summary_rows, output_dir):
#     """
#     Save:
#       1) all_seed_results.csv      : one row per seed
#       2) aggregate_seed_results.csv: mean/std/min/max/count across seeds
#       3) aggregate_confusion_matrix.json
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     rows = [normalize_summary_row(r) for r in summary_rows]
#     df = pd.DataFrame(rows)

#     raw_path = os.path.join(output_dir, "all_seed_results.csv")
#     df.to_csv(raw_path, index=False)

#     metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"]
#     metric_cols = [c for c in metric_cols if c in df.columns]

#     # These define one experimental variant.
#     variant_cols = [
#         "encoder_type",
#         "training_approach",
#         "mil_pool_type",
#         "feature_families",
#         "topology",
#         "connectivity_metric",
#         "connectivity_band",
#         "edge_mode",
#         "base_k",
#         "batch_size",
#         "epochs",
#         "patience",
#         "start_epoch",
#         "lr",
#         "dropout",
#         "weight_decay",
#         "graph_emb_dim",
#         "attn_dim",
#     ]
#     variant_cols = [c for c in variant_cols if c in df.columns]

#     agg = (
#         df.groupby(variant_cols, dropna=False)[metric_cols]
#         .agg(["mean", "std", "min", "max", "count"])
#         .reset_index()
#     )

#     # Flatten multi-index columns
#     agg.columns = [
#         col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
#         for col in agg.columns
#     ]

#     # Add readable mean ± std columns
#     for m in metric_cols:
#         mean_col = f"{m}_mean"
#         std_col = f"{m}_std"
#         if mean_col in agg.columns and std_col in agg.columns:
#             agg[f"{m}_mean_std"] = agg.apply(
#                 lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
#                 if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
#                 axis=1,
#             )

#     agg_path = os.path.join(output_dir, "aggregate_seed_results.csv")
#     agg.to_csv(agg_path, index=False)

#     # Aggregate confusion matrices separately
#     cm_path = None
#     if "confusion_matrix_json" in df.columns:
#         cms = []
#         for s in df["confusion_matrix_json"].dropna():
#             cms.append(np.asarray(json.loads(s), dtype=float))

#         if len(cms) > 0:
#             cm_stack = np.stack(cms, axis=0)
#             cm_info = {
#                 "num_seeds": int(len(cms)),
#                 "confusion_matrix_sum": cm_stack.sum(axis=0).astype(int).tolist(),
#                 "confusion_matrix_mean": cm_stack.mean(axis=0).tolist(),
#                 "confusion_matrix_std": cm_stack.std(axis=0, ddof=1).tolist()
#                 if len(cms) > 1 else np.zeros_like(cm_stack[0]).tolist(),
#             }

#             cm_path = os.path.join(output_dir, "aggregate_confusion_matrix.json")
#             with open(cm_path, "w") as f:
#                 json.dump(cm_info, f, indent=2)

#     print(f"Saved per-seed results: {raw_path}")
#     print(f"Saved aggregate results: {agg_path}")
#     if cm_path is not None:
#         print(f"Saved aggregate confusion matrix: {cm_path}")

#     return df, agg
# def save_seed_aggregation(
#     seed_summaries: list[dict],
#     output_dir: str | os.PathLike,
# ):
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     flat_rows = []

#     for row in seed_summaries:
#         flat_rows.append({
#             "dataset": row["dataset"],
#             "evaluation_protocol": row["evaluation_protocol"],
#             "seed": row["seed"],
#             "num_folds": row.get("num_folds", 1),
#             "accuracy": row["accuracy"],
#             "balanced_accuracy": row["balanced_accuracy"],
#             "macro_f1": row["macro_f1"],
#             "segment_accuracy": row.get(
#                 "segment_accuracy"
#             ),
#             "segment_balanced_accuracy": row.get(
#                 "segment_balanced_accuracy"
#             ),
#             "segment_macro_f1": row.get(
#                 "segment_macro_f1"
#             ),
#         })

#     seed_df = pd.DataFrame(flat_rows)

#     seed_df.to_csv(
#         output_dir / "seed_results.csv",
#         index=False,
#     )

#     save_json(
#         output_dir / "seed_results.json",
#         seed_summaries,
#     )

#     aggregate_payload = {
#         "num_seeds": int(len(seed_df)),
#         "std_ddof": 0,
#         "metrics": {},
#     }

#     aggregate_row = {
#         "num_seeds": int(len(seed_df)),
#     }

#     for metric in METRIC_KEYS:
#         if metric not in seed_df.columns:
#             continue

#         mean_value = float(seed_df[metric].mean())
#         std_value = float(
#             seed_df[metric].std(ddof=0)
#         )

#         aggregate_payload["metrics"][metric] = {
#             "mean": mean_value,
#             "std": std_value,
#         }

#         aggregate_row[f"{metric}_mean"] = mean_value
#         aggregate_row[f"{metric}_std"] = std_value
#         aggregate_row[f"{metric}_mean_std"] = (
#             f"{mean_value:.4f} ± {std_value:.4f}"
#         )

#     aggregate_df = pd.DataFrame([aggregate_row])

#     aggregate_df.to_csv(
#         output_dir / "aggregate_results.csv",
#         index=False,
#     )

#     save_json(
#         output_dir / "aggregate_results.json",
#         aggregate_payload,
#     )

#     return seed_df, aggregate_df

def save_seed_aggregation(summary_rows, output_dir):
    """
    Save per-seed and aggregate results for one experimental variant.
    """
    if not summary_rows:
        raise ValueError("summary_rows is empty.")

    os.makedirs(output_dir, exist_ok=True)

    normalized_rows = [
        normalize_summary_row(row)
        for row in summary_rows
    ]

    df = pd.DataFrame(normalized_rows)

    # ---------------------------------------------------------
    # Per-seed results
    # ---------------------------------------------------------
    raw_csv_path = os.path.join(
        output_dir,
        "all_seed_results.csv",
    )
    df.to_csv(raw_csv_path, index=False)

    raw_json_path = os.path.join(
        output_dir,
        "all_seed_results.json",
    )
    with open(raw_json_path, "w", encoding="utf-8") as file:
        json.dump(
            make_jsonable(summary_rows),
            file,
            indent=2,
        )

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "segment_accuracy",
        "segment_balanced_accuracy",
        "segment_macro_f1",
    ]
    metric_cols = [
        column
        for column in metric_cols
        if column in df.columns
    ]

    num_runs = len(df)
    std_ddof = 1

    aggregate_json = {
        "dataset": df["dataset"].iloc[0]
        if "dataset" in df.columns else None,
        "evaluation_protocol": (
            df["evaluation_protocol"].iloc[0]
            if "evaluation_protocol" in df.columns
            else None
        ),
        "encoder_type": (
            df["encoder_type"].iloc[0]
            if "encoder_type" in df.columns
            else None
        ),
        "num_runs": int(num_runs),
        "seeds": (
            sorted(df["seed"].astype(int).tolist())
            if "seed" in df.columns
            else []
        ),
        "std_ddof": std_ddof,
        "metrics": {},
    }

    aggregate_csv_row = {
        "dataset": aggregate_json["dataset"],
        "evaluation_protocol": (
            aggregate_json["evaluation_protocol"]
        ),
        "encoder_type": aggregate_json["encoder_type"],
        "num_runs": int(num_runs),
    }

    for metric in metric_cols:
        values = pd.to_numeric(
            df[metric],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        mean_value = float(values.mean())

        std_value = (
            float(values.std(ddof=std_ddof))
            if len(values) > 1
            else 0.0
        )

        min_value = float(values.min())
        max_value = float(values.max())

        aggregate_json["metrics"][metric] = {
            "mean": mean_value,
            "std": std_value,
            "min": min_value,
            "max": max_value,
            "count": int(len(values)),
        }

        aggregate_csv_row[f"{metric}_mean"] = mean_value
        aggregate_csv_row[f"{metric}_std"] = std_value
        aggregate_csv_row[f"{metric}_min"] = min_value
        aggregate_csv_row[f"{metric}_max"] = max_value
        aggregate_csv_row[f"{metric}_count"] = int(
            len(values)
        )
        aggregate_csv_row[f"{metric}_mean_std"] = (
            f"{mean_value:.4f} ± {std_value:.4f}"
        )

    # ---------------------------------------------------------
    # Confusion matrices
    # ---------------------------------------------------------
    confusion_matrices = []

    if "confusion_matrix_json" in df.columns:
        for value in df["confusion_matrix_json"].dropna():
            confusion_matrices.append(
                np.asarray(
                    json.loads(value),
                    dtype=float,
                )
            )

    if confusion_matrices:
        shapes = {
            matrix.shape
            for matrix in confusion_matrices
        }

        if len(shapes) != 1:
            raise ValueError(
                "Confusion matrices have inconsistent shapes: "
                f"{sorted(shapes)}"
            )

        cm_stack = np.stack(
            confusion_matrices,
            axis=0,
        )

        aggregate_json["confusion_matrix"] = {
            "count": int(len(cm_stack)),
            "sum": cm_stack.sum(axis=0).tolist(),
            "mean": cm_stack.mean(axis=0).tolist(),
            "std": (
                cm_stack.std(axis=0, ddof=std_ddof).tolist()
                if len(cm_stack) > 1
                else np.zeros_like(cm_stack[0]).tolist()
            ),
        }

    # ---------------------------------------------------------
    # Aggregate CSV and JSON
    # ---------------------------------------------------------
    aggregate_df = pd.DataFrame(
        [aggregate_csv_row]
    )

    aggregate_csv_path = os.path.join(
        output_dir,
        "aggregate_seed_results.csv",
    )
    aggregate_df.to_csv(
        aggregate_csv_path,
        index=False,
    )

    aggregate_json_path = os.path.join(
        output_dir,
        "aggregate_seed_results.json",
    )
    with open(
        aggregate_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_jsonable(aggregate_json),
            file,
            indent=2,
        )

    print(f"Saved per-seed CSV: {raw_csv_path}")
    print(f"Saved per-seed JSON: {raw_json_path}")
    print(f"Saved aggregate CSV: {aggregate_csv_path}")
    print(f"Saved aggregate JSON: {aggregate_json_path}")

    return df, aggregate_df
def save_summary_metrics_csv(summary_rows: List[Dict[str, Any]], csv_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([normalize_summary_row(r) for r in summary_rows])
    df.to_csv(csv_path, index=False)
    return df

# def make_split_subject_id(serial: str, split_name: str, use_split_prefix: bool = True) -> str:
#     serial = str(serial)
#     if not use_split_prefix:
#         return serial
#     if serial.startswith(("train_", "val_", "test_")):
#         return serial
#     return f"{split_name}_{serial}"

# def segment_recording(
#     signal: np.ndarray,
#     crop_len: int = CROP_LEN,
#     step: int = STEP,
#     latency: int = LATENCY,
# ) -> Tuple[List[np.ndarray], List[int]]:
#     x = np.asarray(signal, dtype=np.float32)
#     x = x[:19]  # drop EKG + photic
#     total_len = x.shape[-1]
#     starts = list(range(int(latency), total_len - int(crop_len) + 1, int(step)))
#     windows = [x[:, s:s + int(crop_len)].astype(np.float32, copy=False) for s in starts]
#     return windows, starts

# def dataset_to_subject_records(
#     dataset,
#     split_name: str,
#     use_split_prefix: bool = True,
#     crop_len: int = CROP_LEN,
#     step: int = STEP,
#     latency: int = LATENCY,
#     limit: Optional[int] = None,
# ) -> Tuple[List[Dict[str, Any]], List[str]]:
#     records: List[Dict[str, Any]] = []
#     subject_ids: List[str] = []

#     for sample in dataset:
#         serial = str(sample["serial"])
#         subject_id = make_split_subject_id(serial, split_name, use_split_prefix=use_split_prefix)

#         signal = sample["signal"]
#         label = int(sample["class_label"])
#         age = float(sample.get("age", np.nan))

#         windows, starts = segment_recording(signal, crop_len=crop_len, step=step, latency=latency)
#         if len(windows) == 0:
#             continue

#         rec = {
#             "subject_id": subject_id,
#             "label": label,
#             "class_id": label,
#             "sampling_rate": SFREQ,
#             "channel_names": CAUEEG_EEG19,
#             "windows": windows,
#             "start_samples": starts,
#             "segment_ids": list(range(len(windows))),
#             "recording_info": {
#                 "serial": serial,
#                 "age": age,
#                 "split_name": split_name,
#             },
#         }
#         records.append(rec)
#         subject_ids.append(subject_id)

#         if limit is not None and len(subject_ids) >= int(limit):
#             break

#     return records, subject_ids



def load_subject_index_from_h5(h5_path: str) -> pd.DataFrame:
    rows = []

    with h5py.File(h5_path, "r") as h5f:
        if "subjects" not in h5f:
            raise KeyError(
                f"Missing 'subjects' group in H5 file: {h5_path}"
            )

        for subject_id, subject_group in h5f["subjects"].items():
            metadata = subject_group["metadata"]

            rows.append({
                "subject_id": str(subject_id),
                "label": int(metadata.attrs["label"]),
            })

    if not rows:
        raise RuntimeError(f"No subjects found in H5 file: {h5_path}")

    subject_df = pd.DataFrame(rows)
    subject_df = subject_df.sort_values("subject_id").reset_index(drop=True)

    return subject_df

# def split_caueeg_subjects(
#     subject_df: pd.DataFrame,
# ) -> tuple[list[str], list[str], list[str]]:
    # subject_ids = subject_df["subject_id"].tolist()
def get_caueeg_predefined_split(subject_ids):
    train_subjects = [
        sid for sid in subject_ids
        if sid.startswith("train_")
    ]
    val_subjects = [
        sid for sid in subject_ids
        if sid.startswith("val_")
    ]
    test_subjects = [
        sid for sid in subject_ids
        if sid.startswith("test_")
    ]

    if not train_subjects:
        raise RuntimeError("No CAUEEG train subjects found in H5.")

    if not val_subjects:
        raise RuntimeError("No CAUEEG validation subjects found in H5.")

    if not test_subjects:
        raise RuntimeError("No CAUEEG test subjects found in H5.")

    return train_subjects, val_subjects, test_subjects

def resolve_bank_specs(
    bank_specs_json: str | None,
):
    bank_specs = load_bank_specs(
        bank_specs_json
    )

    if bank_specs is not None:
        return bank_specs

    return [
        {
            "name": "wpli_theta_full",
            "connectivity_metric": "wpli",
            "connectivity_band": 1,
            "filter_method": "full",
        },
        {
            "name": "wpli_alpha_fixed",
            "connectivity_metric": "wpli",
            "connectivity_band": 2,
            "filter_method": "fixed",
        },
        {
            "name": "coherence_alpha_combined",
            "connectivity_metric": "coherence",
            "connectivity_band": 2,
            "filter_method": "combined",
        },
        {
            "name": "coherence_theta_topk4",
            "connectivity_metric": "coherence",
            "connectivity_band": 1,
            "filter_method": "topk",
        },
    ]
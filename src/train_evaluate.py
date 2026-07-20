
import os
import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, f1_score, recall_score

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _move_to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors in nested structures to CPU so checkpoints are portable.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj

def move_batch_to_device(batch: Dict[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to"):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: Optional[int] = None) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(int(num_classes))) if num_classes is not None else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "conf_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def train_segment_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
) -> Dict[str, Any]:
    model.train()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        logits = out["logits"]
        labels = batch["labels"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


class EarlyStopping:
    """
    Early stopping driven by LOWER validation loss.

    Features:
    - warmup via start_epoch
    - min_delta for meaningful improvement
    - keeps top-k checkpoints with lowest val_loss
    - exposes selected checkpoint using:
        1) max val_bal_acc
        2) max val_macro_f1
        3) min val_loss
    - deletes checkpoints that fall out of the top-k set
    """

    def __init__(
        self,
        patience: int,
        start_epoch: int = 0,
        min_delta: float = 0.0,
        top_k: int = 1,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        file_prefix: str = "checkpoint",
    ):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {start_epoch}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.top_k = int(top_k)
        self.save_dir = save_dir
        self.verbose = verbose
        self.file_prefix = file_prefix

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch_by_loss = None
        self.counter = 0
        self.should_stop = False
        self.stop_epoch = None

        # sorted by storage rule: lowest val_loss first
        self.checkpoints: List[Dict[str, Any]] = []

    def _improved(self, val_loss: float) -> bool:
        return val_loss < (self.best_val_loss - self.min_delta)

    @staticmethod
    def _storage_sort_key(meta: Dict[str, Any]):
        """
        How checkpoints are ranked for staying in top-k.
        Primary goal: lowest validation loss.
        Ties are broken deterministically.
        """
        return (
            float(meta["val_loss"]),
            -float(meta["val_bal_acc"]),
            -float(meta["val_macro_f1"]),
            int(meta["epoch"]),
        )

    @staticmethod
    def _selection_sort_key(meta: Dict[str, Any]):
        """
        Final checkpoint selection rule requested by user:
          1) max val_bal_acc
          2) max val_macro_f1
          3) min val_loss
        """
        return (
            -float(meta["val_bal_acc"]),
            -float(meta["val_macro_f1"]),
            float(meta["val_loss"]),
            int(meta["epoch"]),
        )

    def _checkpoint_filename(self, epoch: int, val_loss: float) -> str:
        return f"{self.file_prefix}_epoch{epoch:03d}_valloss{val_loss:.6f}.pt"

    def _build_meta(
        self,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "val_bal_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            "path": path,
        }

    def _build_payload(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "epoch": int(epoch),
            "model_state_dict": _move_to_cpu(copy.deepcopy(model.state_dict())),
            "optimizer_state_dict": _move_to_cpu(copy.deepcopy(optimizer.state_dict()))
            if optimizer is not None
            else None,
            # canonical fields
            "val_loss": float(val_loss),
            "val_bal_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            # legacy-friendly aliases
            "best_val_loss": float(val_loss),
            "best_val_bal_acc": float(val_bal_acc),
            "best_val_macro_f1": float(val_macro_f1),
        }

        if extra_state is not None:
            payload.update(_move_to_cpu(copy.deepcopy(extra_state)))

        return payload

    def _maybe_remove_from_disk(self, meta: Dict[str, Any]) -> None:
        path = meta.get("path", None)
        if path is None:
            return
        if os.path.exists(path):
            try:
                os.remove(path)
                if self.verbose:
                    print(f"Removed checkpoint that fell out of top-{self.top_k}: {path}")
            except OSError:
                if self.verbose:
                    print(f"Warning: could not remove old checkpoint: {path}")

    def _qualifies_for_topk(self, candidate_meta: Dict[str, Any]) -> bool:
        if len(self.checkpoints) < self.top_k:
            return True

        worst_meta = sorted(self.checkpoints, key=self._storage_sort_key)[-1]
        return self._storage_sort_key(candidate_meta) < self._storage_sort_key(worst_meta)

    def _insert_topk(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        candidate_meta = self._build_meta(
            epoch=epoch,
            val_loss=val_loss,
            val_bal_acc=val_bal_acc,
            val_macro_f1=val_macro_f1,
            path=None,
        )

        if not self._qualifies_for_topk(candidate_meta):
            return None

        if self.save_dir is None:
            # metadata only, no file path
            saved_meta = candidate_meta
        else:
            ckpt_path = os.path.join(
                self.save_dir,
                self._checkpoint_filename(epoch=epoch, val_loss=val_loss),
            )
            payload = self._build_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_bal_acc=val_bal_acc,
                val_macro_f1=val_macro_f1,
                extra_state=extra_state,
            )
            torch.save(payload, ckpt_path)
            saved_meta = self._build_meta(
                epoch=epoch,
                val_loss=val_loss,
                val_bal_acc=val_bal_acc,
                val_macro_f1=val_macro_f1,
                path=ckpt_path,
            )

            if self.verbose:
                print(
                    f"Saved top-k checkpoint: epoch={epoch}, "
                    f"val_loss={val_loss:.6f}, "
                    f"val_bal_acc={val_bal_acc:.4f}, "
                    f"val_macro_f1={val_macro_f1:.4f}"
                )

        self.checkpoints.append(saved_meta)
        self.checkpoints = sorted(self.checkpoints, key=self._storage_sort_key)

        while len(self.checkpoints) > self.top_k:
            removed = self.checkpoints.pop(-1)
            self._maybe_remove_from_disk(removed)

        return saved_meta

    def __call__(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns:
            True if training should stop, else False
        """
        val_loss = float(val_loss)
        val_bal_acc = float(val_bal_acc)
        val_macro_f1 = float(val_macro_f1)
        epoch = int(epoch)

        # Always maintain top-k checkpoints
        self._insert_topk(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            val_bal_acc=val_bal_acc,
            val_macro_f1=val_macro_f1,
            extra_state=extra_state,
        )

        monitor_active = epoch >= self.start_epoch

        if self._improved(val_loss):
            self.best_val_loss = val_loss
            self.best_epoch_by_loss = epoch
            if monitor_active:
                self.counter = 0

            if self.verbose:
                print(
                    f"Validation loss improved at epoch {epoch}: "
                    f"{val_loss:.6f}"
                )
        else:
            if monitor_active:
                self.counter += 1
                if self.verbose:
                    print(
                        f"No val-loss improvement at epoch {epoch}. "
                        f"patience {self.counter}/{self.patience}"
                    )

        if monitor_active and self.counter >= self.patience:
            self.should_stop = True
            self.stop_epoch = epoch
            if self.verbose:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val_loss={self.best_val_loss:.6f} "
                    f"(epoch {self.best_epoch_by_loss})."
                )

        return self.should_stop

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Select final checkpoint from saved top-k by:
          1) highest val_bal_acc
          2) highest val_macro_f1
          3) lower val_loss
        """
        if len(self.checkpoints) == 0:
            return None
        best_meta = sorted(self.checkpoints, key=self._selection_sort_key)[0]
        return copy.deepcopy(best_meta)

    def get_topk_checkpoints(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.checkpoints)

def fit_segment_baseline_subject_es(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
    epochs: int,
    patience: int,
    start_epoch: int,
    min_delta: float,
    top_k: int,
    save_path: Optional[str],
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        file_prefix = os.path.splitext(os.path.basename(save_path))[0] + "_topk"
    else:
        save_dir = None
        file_prefix = "segment_topk"

    if EarlyStopping is not None:
        early_stopper = EarlyStopping(
            patience=patience,
            start_epoch=start_epoch,
            min_delta=min_delta,
            top_k=top_k,
            save_dir=save_dir,
            verbose=verbose,
            file_prefix=file_prefix,
        )
    else:
        early_stopper = None
        best_loss = float("inf")
        bad_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        train_seg_metrics = train_segment_one_epoch(model, train_loader, optimizer, criterion, device, num_classes)
                
        train_subject_metrics, _, _, _ = evaluate_segment_subject_level(model, train_loader, criterion, device, num_classes, split="train")
        val_subject_metrics, _, _, val_seg_metrics = evaluate_segment_subject_level(model, val_loader, criterion, device, num_classes, split="val")

        row = {
            "epoch": int(epoch),
            "train_seg_loss": float(train_seg_metrics["loss"]),
            "train_seg_acc": float(train_seg_metrics["accuracy"]),
            "train_seg_bal_acc": float(train_seg_metrics["balanced_accuracy"]),
            "train_seg_macro_f1": float(train_seg_metrics["macro_f1"]),
            "train_loss": float(train_subject_metrics["loss"]),
            "train_acc": float(train_subject_metrics["accuracy"]),
            "train_bal_acc": float(train_subject_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_subject_metrics["macro_f1"]),
            "val_seg_loss": float(val_seg_metrics["loss"]),
            "val_seg_acc": float(val_seg_metrics["accuracy"]),
            "val_seg_bal_acc": float(val_seg_metrics["balanced_accuracy"]),
            "val_seg_macro_f1": float(val_seg_metrics["macro_f1"]),
            "val_loss": float(val_subject_metrics["loss"]),
            "val_acc": float(val_subject_metrics["accuracy"]),
            "val_bal_acc": float(val_subject_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_subject_metrics["macro_f1"]),
        }

        # if residual_stats is not None:
        #     row.update(residual_stats)
        history.append(row)

        if verbose:
            if epoch%10==0:
                print(
                    f"Epoch [{epoch:03d}/{epochs}] | "
                    f"TrainSeg loss={row['train_seg_loss']:.4f}, acc={row['train_seg_acc']:.4f} | "
                    f"TrainSubj loss={row['train_loss']:.4f}, bal={row['train_bal_acc']:.4f}, f1={row['train_macro_f1']:.4f} || "
                    f"ValSubj loss={row['val_loss']:.4f}, acc={row['val_acc']:.4f}, "
                    f"bal={row['val_bal_acc']:.4f}, f1={row['val_macro_f1']:.4f}"
                )

        if early_stopper is not None:
            should_stop = early_stopper(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=float(val_subject_metrics["loss"]),
                val_bal_acc=float(val_subject_metrics["balanced_accuracy"]),
                val_macro_f1=float(val_subject_metrics["macro_f1"]),
                extra_state={"history": history},
            )
        else:
            improved = float(val_subject_metrics["loss"]) < best_loss - float(min_delta)
            if improved:
                best_loss = float(val_subject_metrics["loss"])
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
                if save_path is not None:
                    torch.save({"epoch": epoch, "model_state_dict": best_state, "optimizer_state_dict": optimizer.state_dict()}, save_path)
            elif epoch >= start_epoch:
                bad_epochs += 1
            should_stop = bad_epochs >= patience

        if should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    if early_stopper is not None:
        best_meta = early_stopper.get_best_checkpoint()
        if best_meta is not None and best_meta.get("path") is not None and os.path.exists(best_meta["path"]):
            checkpoint = torch.load(best_meta["path"], map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_state = checkpoint["model_state_dict"]
            if save_path is not None and best_meta["path"] != save_path:
                torch.save(checkpoint, save_path)
        elif best_state is not None:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    best_val_metrics, _, _, _ = evaluate_segment_subject_level(model, val_loader, criterion, device, num_classes, split="val")
    return model, best_val_metrics, history, best_state



def evaluate_segment_subject_level(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
    split: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    segment_metrics = evaluate_segment_loader(model, loader, criterion, device, num_classes=num_classes)
    segment_df = segment_metrics_to_df(segment_metrics, num_classes=num_classes, split=split)
    subject_df = aggregate_segment_df_to_subject_df(segment_df, num_classes=num_classes, split=split)
    subject_metrics = subject_df_to_metrics(subject_df, num_classes=num_classes)
    return subject_metrics, subject_df, segment_df, segment_metrics


@torch.no_grad()
def evaluate_segment_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    subject_ids: List[str] = []
    segment_ids: List[int] = []
    start_samples: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        labels = batch["labels"]
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
        subject_ids.extend([str(x) for x in batch["subject_ids"]])
        segment_ids.extend([int(x) for x in batch["segment_ids"]])
        start_samples.extend([int(x) for x in batch["start_samples"]])

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["y_prob"] = y_prob
    metrics["subject_ids"] = subject_ids
    metrics["segment_ids"] = segment_ids
    metrics["start_samples"] = start_samples
    return metrics


def segment_metrics_to_df(metrics: Dict[str, Any], num_classes: int, split: str) -> pd.DataFrame:
    rows = []
    for sid, seg_id, st, yt, yp, prob in zip(
        metrics["subject_ids"], metrics["segment_ids"], metrics["start_samples"],
        metrics["y_true"], metrics["y_pred"], metrics["y_prob"]
    ):
        row = {
            "split": split,
            "subject_id": str(sid),
            "segment_id": int(seg_id),
            "start_sample": int(st),
            "true_label": int(yt),
            "segment_pred_label": int(yp),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(prob[c])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_segment_df_to_subject_df(segment_df: pd.DataFrame, num_classes: int, split: str) -> pd.DataFrame:
    rows = []
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    for sid, sdf in segment_df.groupby("subject_id"):
        labels = sdf["true_label"].astype(int).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Subject {sid} has mixed labels in segment_df: {labels}")
        mean_prob = sdf[prob_cols].to_numpy(dtype=np.float64).mean(axis=0)
        pred = int(np.argmax(mean_prob))
        row = {
            "split": split,
            "subject_id": str(sid),
            "true_label": int(labels[0]),
            "pred_label": pred,
            "num_segments": int(len(sdf)),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(mean_prob[c])
        rows.append(row)
    return pd.DataFrame(rows)


def subject_df_to_metrics(subject_df: pd.DataFrame, num_classes: int) -> Dict[str, Any]:
    y_true = subject_df["true_label"].astype(int).to_numpy()
    y_pred = subject_df["pred_label"].astype(int).to_numpy()
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    prob = subject_df[[f"prob_{c}" for c in range(num_classes)]].to_numpy(dtype=np.float64)
    p_true = np.clip(prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    metrics["loss"] = float((-np.log(p_true)).mean())
    metrics["y_prob"] = prob.tolist()
    return metrics


def _extract_bank_attention_from_output(out):
    """
    Return attention tensor from model output.

    Expected shapes:
      [B, K]    for scalar candidate attention
      [B, K, D] for dimension-wise attention/gates
    """
    candidate_keys = [
        "view_attention",
        "candidate_fusion_weights",
        "fusion_weights",
        "graph_attention_weights",
    ]

    for k in candidate_keys:
        if k in out and out[k] is not None:
            return out[k], k

    if "graph_attn" in out and out["graph_attn"] is not None:
        graph_attn = out["graph_attn"]
        if isinstance(graph_attn, dict):
            for k in candidate_keys:
                if k in graph_attn and graph_attn[k] is not None:
                    return graph_attn[k], f"graph_attn.{k}"
        elif torch.is_tensor(graph_attn):
            return graph_attn, "graph_attn"

    return None, None


def move_batch_to_device_simple(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
    
@torch.no_grad()
def collect_bank_attention_segment_level(
    model,
    loader,
    device,
    split_name,
    candidate_names=None,
    num_classes=None,
):
    """
    Collect candidate/topology attention per segment.

    Output:
      long_df:
        one row per segment-candidate pair

      summary_df:
        one row per segment, with entropy/effective-k/max-attention
    """
    model.eval()

    long_rows = []
    summary_rows = []

    for batch in loader:
        batch = move_batch_to_device_simple(batch, device)
        out = model(batch)

        attn, attn_key = _extract_bank_attention_from_output(out)
        if attn is None:
            raise KeyError(
                "No candidate attention found in model output. "
                "For concat/mean fusion, attention is usually None. "
                "Use gated/context_gated/dim_gated/attn_residual to collect attention."
            )

        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        y_true = batch["labels"].detach().cpu().numpy().astype(int)
        subject_ids = list(batch["subject_ids"])
        segment_ids = list(batch.get("segment_ids", range(len(subject_ids))))
        start_samples = list(batch.get("start_samples", [-1] * len(subject_ids)))

        attn = attn.detach().cpu()

        # If dimension-wise gates [B, K, D], average over D for plotting.
        if attn.ndim == 3:
            attn_plot = attn.mean(dim=-1)
        elif attn.ndim == 2:
            attn_plot = attn
        else:
            raise ValueError(f"Expected attention [B,K] or [B,K,D], got {tuple(attn.shape)}")

        attn_np = attn_plot.numpy().astype(np.float64)

        # Normalize across candidates for safety.
        row_sum = attn_np.sum(axis=1, keepdims=True)
        attn_np = attn_np / np.clip(row_sum, 1e-12, None)

        B, K = attn_np.shape

        if candidate_names is None:
            if "topology_names" in batch and batch["topology_names"] is not None:
                names = list(batch["topology_names"])
            else:
                names = [f"cand_{k}" for k in range(K)]
        else:
            names = list(candidate_names)

        if len(names) != K:
            names = [f"cand_{k}" for k in range(K)]

        for i in range(B):
            a = attn_np[i]
            max_attention = float(a.max())
            dominant_idx = int(a.argmax())
            entropy = float(-(a * np.log(a + 1e-12)).sum())
            norm_entropy = float(entropy / np.log(K)) if K > 1 else 0.0
            effective_k = float(1.0 / np.sum(a ** 2))

            true_label = int(y_true[i])
            pred_label = int(preds[i])
            correct = int(true_label == pred_label)

            summary_row = {
                "split": split_name,
                "subject_id": str(subject_ids[i]),
                "segment_id": int(segment_ids[i]),
                "start_sample": int(start_samples[i]),
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": correct,
                "attn_key": attn_key,
                "num_candidates": int(K),
                "dominant_candidate_idx": dominant_idx,
                "dominant_candidate_name": names[dominant_idx],
                "max_attention": max_attention,
                "normalized_entropy": norm_entropy,
                "effective_num_candidates": effective_k,
            }

            for c in range(probs.shape[1]):
                summary_row[f"prob_{c}"] = float(probs[i, c])

            summary_rows.append(summary_row)

            for k in range(K):
                long_rows.append({
                    "split": split_name,
                    "subject_id": str(subject_ids[i]),
                    "segment_id": int(segment_ids[i]),
                    "start_sample": int(start_samples[i]),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": correct,
                    "candidate_idx": int(k),
                    "candidate_name": names[k],
                    "attention": float(a[k]),
                    "dominant_candidate_idx": dominant_idx,
                    "dominant_candidate_name": names[dominant_idx],
                    "max_attention": max_attention,
                    "normalized_entropy": norm_entropy,
                    "effective_num_candidates": effective_k,
                })

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)

    return long_df, summary_df


# def get_bank_residual_stats(model):
#     enc = getattr(model, "graph_encoder", None)
#     if enc is None:
#         return None

#     if not hasattr(enc, "bank_residual_logit"):
#         return None

#     if enc.bank_residual_logit is None:
#         return None

#     with torch.no_grad():
#         logit = enc.bank_residual_logit
#         gamma_max = getattr(enc, "bank_residual_max", 1.0)
#         gamma = gamma_max * torch.sigmoid(logit)

#         grad = enc.bank_residual_logit.grad
#         grad_val = None if grad is None else float(grad.detach().cpu().item())

#         return {
#             "bank_residual_logit": float(logit.detach().cpu().item()),
#             "bank_residual_gamma": float(gamma.detach().cpu().item()),
#             "bank_residual_logit_grad": grad_val,
#         }

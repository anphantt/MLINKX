from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.config as config
from src.graphs import (
    GraphSegmentDataset,
    build_graph_bank_from_specs,
    build_graphs_from_payload,
    collate_graph_segments,
)
from src.models import SegmentGraphClassifierFromMIL
from src.train_evaluate import (
    build_model_kwargs,
    evaluate_segment_subject_level,
    fit_segment_baseline_subject_es,
)
from src.utils import (
    collect_required_connectivity_metrics,
    load_bank_specs,
    load_h5_payload_for_subjects,
    save_seed_aggregation,
    save_summary_metrics_csv,
    set_global_seed,
    normalize_fixed_edges,
)


BANK_ENCODERS = {"mlinkx", "cnn_bank", "gnn_bank"}

DEFAULT_BANK_SPECS = [
    {"name": "wpli_theta_full", "connectivity_metric": "wpli", "connectivity_band": 1, "filter_method": "full"},
    {"name": "wpli_alpha_fixed", "connectivity_metric": "wpli", "connectivity_band": 2, "filter_method": "fixed"},
    {"name": "coherence_alpha_combined", "connectivity_metric": "coherence", "connectivity_band": 2, "filter_method": "combined"},
    {"name": "coherence_theta_topk4", "connectivity_metric": "coherence", "connectivity_band": 1, "filter_method": "topk"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train graph models on the official CAUEEG split.")
    p.add_argument("--out_h5", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--split_meta", default=None)
    p.add_argument("--feature_families_str", default="relative_band_power,statistical")
    p.add_argument("--connectivity_metric", default="wpli")
    p.add_argument("--connectivity_band", type=int, default=2)
    p.add_argument("--topology", default="fixed")
    p.add_argument("--encoder_type", default="linkx")
    p.add_argument("--candidate_fusion_mode", default="concat")
    p.add_argument("--candidate_fusion_hidden_dim", type=int, default=None)
    p.add_argument("--candidate_fusion_dropout", type=float, default=0.0)
    p.add_argument("--bank_specs_json", default=None)
    p.add_argument("--graph_backbone", default="gatv2", choices=["gatv2", "gcn", "sage"])
    p.add_argument("--seeds", default="15,42,100")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--start_epoch", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--graph_emb_dim", type=int, default=64)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--gnn_hidden_dim", type=int, default=64)
    p.add_argument("--node_hidden_dims", default="256,128")
    p.add_argument("--edge_hidden_dims", default="128,64")
    p.add_argument("--branch_emb_dim", type=int, default=64)
    p.add_argument("--edge_mode", default="topology_weighted")
    p.add_argument("--device", default="cuda")
    p.add_argument("--class_names", nargs="+", default=["HC", "Dementia", "MCI"])
    return p.parse_args()


def _int_tuple(value: str):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def load_split_ids(h5_path: str, meta_path: str | None):
    if meta_path is None:
        candidate = h5_path.replace(".h5", "_meta.json")
        meta_path = candidate if os.path.isfile(candidate) else None

    if meta_path is not None:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return list(meta["train_ids"]), list(meta["val_ids"]), list(meta["test_ids"])

    with h5py.File(h5_path, "r") as h5f:
        ids = list(h5f["subjects"].keys())
    train_ids = [sid for sid in ids if sid.startswith("train_")]
    val_ids = [sid for sid in ids if sid.startswith("val_")]
    test_ids = [sid for sid in ids if sid.startswith("test_")]
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError("Could not recover CAUEEG train/val/test IDs from H5 prefixes.")
    return train_ids, val_ids, test_ids


def build_graphs(
    payload,
    subject_ids,
    *,
    args,
    feature_families,
    fixed_edges,
    channel_names,
    bank_specs,
):
    if args.encoder_type.lower() in BANK_ENCODERS:
        return build_graph_bank_from_specs(
            payload,
            subject_ids,
            feature_families=feature_families,
            default_connectivity_metric=args.connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=args.topology,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )

    graphs = build_graphs_from_payload(
        payload,
        subject_ids,
        feature_families=feature_families,
        connectivity_metric=args.connectivity_metric,
        connectivity_band=args.connectivity_band,
        filter_method=args.topology,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        undirected=True,
        standardize_features=True,
    )
    return graphs, None


def run_one_seed(
    *,
    args,
    seed,
    train_graphs,
    val_graphs,
    test_graphs,
    num_classes,
    run_dir,
):
    set_global_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")
    test_batch_size = args.test_batch_size or max(16, args.batch_size)

    train_loader = DataLoader(
        GraphSegmentDataset(train_graphs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        GraphSegmentDataset(train_graphs),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        GraphSegmentDataset(val_graphs),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        GraphSegmentDataset(test_graphs),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )

    first_graph = train_graphs[0]
    bank_encoder = args.encoder_type.lower() in BANK_ENCODERS
    num_candidates = int(first_graph.adj_bank.shape[0]) if bank_encoder else None

    model_kwargs = build_model_kwargs(
        num_node_features=int(first_graph.x.shape[-1]),
        num_classes=num_classes,
        num_nodes=int(first_graph.x.shape[0]),
        encoder_type=args.encoder_type,
        edge_mode=args.edge_mode,
        graph_emb_dim=args.graph_emb_dim,
        dropout=args.dropout,
        temp=args.temp,
        attn_dim=args.attn_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        node_hidden_dims=args.node_hidden_dims,
        edge_hidden_dims=args.edge_hidden_dims,
        branch_emb_dim=args.branch_emb_dim,
        cnn_num_bands=num_candidates,
        num_candidates=num_candidates,
        candidate_fusion_mode=args.candidate_fusion_mode,
        candidate_fusion_hidden_dim=args.candidate_fusion_hidden_dim,
        candidate_fusion_dropout=args.candidate_fusion_dropout,
        graph_backbone=args.graph_backbone,
    )
    model = SegmentGraphClassifierFromMIL(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    model, _, history, _ = fit_segment_baseline_subject_es(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        epochs=args.epochs,
        patience=args.patience,
        start_epoch=args.start_epoch,
        min_delta=args.min_delta,
        top_k=args.top_k,
        save_path=ckpt_path,
        verbose=True,
    )

    train_metrics, train_subject_df, train_segment_df, train_seg_metrics = evaluate_segment_subject_level(
        model, train_eval_loader, criterion, device, num_classes, "train"
    )
    val_metrics, val_subject_df, val_segment_df, val_seg_metrics = evaluate_segment_subject_level(
        model, val_loader, criterion, device, num_classes, "val"
    )
    test_metrics, test_subject_df, test_segment_df, test_seg_metrics = evaluate_segment_subject_level(
        model, test_loader, criterion, device, num_classes, "test"
    )

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    train_subject_df.to_csv(os.path.join(run_dir, "train_predictions_subject_agg.csv"), index=False)
    val_subject_df.to_csv(os.path.join(run_dir, "val_predictions_subject_agg.csv"), index=False)
    test_subject_df.to_csv(os.path.join(run_dir, "test_predictions_subject_agg.csv"), index=False)
    train_segment_df.to_csv(os.path.join(run_dir, "train_predictions_segment.csv"), index=False)
    val_segment_df.to_csv(os.path.join(run_dir, "val_predictions_segment.csv"), index=False)
    test_segment_df.to_csv(os.path.join(run_dir, "test_predictions_segment.csv"), index=False)

    summary_rows = []
    for split, subj_m, seg_m in [
        ("train", train_metrics, train_seg_metrics),
        ("val", val_metrics, val_seg_metrics),
        ("test", test_metrics, test_seg_metrics),
    ]:
        summary_rows.append({
            "split": split,
            "loss": float(subj_m["loss"]),
            "accuracy": float(subj_m["accuracy"]),
            "balanced_accuracy": float(subj_m["balanced_accuracy"]),
            "macro_f1": float(subj_m["macro_f1"]),
            "segment_loss": float(seg_m["loss"]),
            "segment_accuracy": float(seg_m["accuracy"]),
            "segment_balanced_accuracy": float(seg_m["balanced_accuracy"]),
            "segment_macro_f1": float(seg_m["macro_f1"]),
            "confusion_matrix": subj_m["conf_matrix"],
        })
    save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))

    summary_test = {
        "dataset": "caueeg",
        "evaluation_protocol": "official split",
        "seed": int(seed),
        "encoder_type": args.encoder_type,
        "training_approach": "segment_all",
        "feature_families": list(args.feature_families),
        "topology": args.topology,
        "connectivity_metric": args.connectivity_metric,
        "connectivity_band": args.connectivity_band,
        "candidate_fusion_mode": args.candidate_fusion_mode if bank_encoder else None,
        "graph_backbone": args.graph_backbone,
        "accuracy": float(test_metrics["accuracy"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "confusion_matrix": test_metrics["conf_matrix"],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "start_epoch": args.start_epoch,
        "lr": args.lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "graph_emb_dim": args.graph_emb_dim,
        "attn_dim": args.attn_dim,
    }
    save_summary_metrics_csv([summary_test], os.path.join(run_dir, "summary_test.csv"))
    return summary_test


def main() -> None:
    args = parse_args()
    args.feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    args.node_hidden_dims = _int_tuple(args.node_hidden_dims)
    args.edge_hidden_dims = _int_tuple(args.edge_hidden_dims)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    train_ids, val_ids, test_ids = load_split_ids(args.out_h5, args.split_meta)
    all_ids = train_ids + val_ids + test_ids

    bank_specs = load_bank_specs(args.bank_specs_json) or DEFAULT_BANK_SPECS
    required_metrics = (
        collect_required_connectivity_metrics(bank_specs, args.connectivity_metric)
        if args.encoder_type.lower() in BANK_ENCODERS
        else [args.connectivity_metric]
    )

    payload = load_h5_payload_for_subjects(
        h5_path=args.out_h5,
        subject_ids=all_ids,
        feature_families=args.feature_families,
        connectivity_metrics=required_metrics,
        connectivity_band=None,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    labels = [int(payload[sid]["label"]) for sid in all_ids]
    num_classes = len(sorted(set(labels)))
    channel_names = list(payload[train_ids[0]]["channel_names"])
    fixed_edges = normalize_fixed_edges(config.MONOFIXEDGES, len(channel_names), channel_names)

    train_graphs, topology_names = build_graphs(
        payload,
        train_ids,
        args=args,
        feature_families=args.feature_families,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        bank_specs=bank_specs,
    )
    val_graphs, _ = build_graphs(
        payload,
        val_ids,
        args=args,
        feature_families=args.feature_families,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        bank_specs=bank_specs,
    )
    test_graphs, _ = build_graphs(
        payload,
        test_ids,
        args=args,
        feature_families=args.feature_families,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        bank_specs=bank_specs,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        args.output_root,
        f"{stamp}_{args.encoder_type}_{args.candidate_fusion_mode}_{args.connectivity_metric}_{args.topology}",
    )
    os.makedirs(experiment_root, exist_ok=True)

    summary_rows = []
    for seed in seeds:
        run_dir = os.path.join(experiment_root, f"seed{seed}")
        summary_rows.append(
            run_one_seed(
                args=args,
                seed=seed,
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                test_graphs=test_graphs,
                num_classes=num_classes,
                run_dir=run_dir,
            )
        )

    save_seed_aggregation(summary_rows, os.path.join(experiment_root, "agg_seed_results"))
    print(f"Done. Experiment root: {experiment_root}")


if __name__ == "__main__":
    main()

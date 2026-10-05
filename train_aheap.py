from __future__ import annotations

import argparse
import os
from datetime import datetime

import h5py
import numpy as np

import src.config as config
from src.graphs import build_graph_bank_from_specs, build_graphs_from_payload
from src.train_evaluate import run_one_graph_split
from src.utils import (
    aggregate_fold_summaries_to_seed,
    balanced_kfold_split,
    collect_required_connectivity_metrics,
    load_bank_specs,
    load_h5_payload_for_subjects,
    save_seed_aggregation,
    stratified_split_subjects,
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
    p = argparse.ArgumentParser(description="Train graph models on AHEAP using 5-fold subject CV.")
    p.add_argument("--out_h5", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--feature_families_str", default="relative_band_power,statistical")
    p.add_argument("--connectivity_metric", default="wpli")
    p.add_argument("--connectivity_band", type=int, default=2)
    p.add_argument("--topology", default="fixed")
    p.add_argument("--encoder_type", default="linkx")
    p.add_argument("--candidate_fusion_mode", default="static_learned", choices=["concat", "static_learned"])
    p.add_argument("--bank_specs_json", default=None)
    p.add_argument("--backbone", default="gatv2", choices=["gatv2", "gcn", "sage"])
    p.add_argument("--seeds", default="15,42,100")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--start_epoch", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--temp", type=float, default=2.5)
    p.add_argument("--class_names", nargs="+", default=["HC", "AD", "FTD"])
    return p.parse_args()


def load_subjects_and_labels(h5_path: str):
    with h5py.File(h5_path, "r") as h5f:
        subject_ids = list(h5f["subjects"].keys())
        labels = [int(h5f[f"subjects/{sid}/metadata"].attrs["label"]) for sid in subject_ids]
    return subject_ids, labels


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


def main() -> None:
    args = parse_args()
    args.feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    subject_ids, labels = load_subjects_and_labels(args.out_h5)
    num_classes = len(sorted(set(labels)))

    bank_specs = load_bank_specs(args.bank_specs_json) or DEFAULT_BANK_SPECS
    required_metrics = (
        collect_required_connectivity_metrics(bank_specs, args.connectivity_metric)
        if args.encoder_type.lower() in BANK_ENCODERS
        else [args.connectivity_metric]
    )

    payload = load_h5_payload_for_subjects(
        h5_path=args.out_h5,
        subject_ids=subject_ids,
        feature_families=args.feature_families,
        connectivity_metrics=required_metrics,
        connectivity_band=None,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    channel_names = list(payload[subject_ids[0]]["channel_names"])
    fixed_edges = normalize_fixed_edges(config.MONOFIXEDGES, len(channel_names), channel_names)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        args.output_root,
        f"{stamp}_{args.encoder_type}_{args.candidate_fusion_mode}_{args.connectivity_metric}_{args.topology}",
    )
    os.makedirs(experiment_root, exist_ok=True)

    seed_rows = []

    for seed in seeds:
        fold_rows = []
        folds = balanced_kfold_split(subject_ids, labels, seed, args.k_folds)

        for fold_idx, test_subjects in enumerate(folds):
            train_subjects = [sid for sid in subject_ids if sid not in set(test_subjects)]
            label_map = dict(zip(subject_ids, labels))
            train_subjects, val_subjects = stratified_split_subjects(
                train_subjects,
                label_map,
                args.val_ratio,
                seed,
            )

            train_graphs, topology_names = build_graphs(
                payload,
                train_subjects,
                args=args,
                feature_families=args.feature_families,
                fixed_edges=fixed_edges,
                channel_names=channel_names,
                bank_specs=bank_specs,
            )
            val_graphs, _ = build_graphs(
                payload,
                val_subjects,
                args=args,
                feature_families=args.feature_families,
                fixed_edges=fixed_edges,
                channel_names=channel_names,
                bank_specs=bank_specs,
            )
            test_graphs, _ = build_graphs(
                payload,
                test_subjects,
                args=args,
                feature_families=args.feature_families,
                fixed_edges=fixed_edges,
                channel_names=channel_names,
                bank_specs=bank_specs,
            )

            run_dir = os.path.join(experiment_root, f"seed{seed}", f"fold{fold_idx}")
            summary = run_one_graph_split(
                args=args,
                dataset_name="aheap",
                evaluation_protocol="5-fold subject CV",
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                test_graphs=test_graphs,
                num_classes=num_classes,
                class_names=args.class_names,
                seed=seed,
                run_dir=run_dir,
                fold_index=fold_idx,
                topology_names=topology_names,
            )
            fold_rows.append(summary)

        seed_rows.append(aggregate_fold_summaries_to_seed(fold_rows, seed=seed))

    save_seed_aggregation(seed_rows, os.path.join(experiment_root, "agg_seed_results"))
    print(f"Done. Experiment root: {experiment_root}")


if __name__ == "__main__":
    main()

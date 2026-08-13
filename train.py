import os
import argparse
import pandas as pd
import numpy as np

from src.utils import * #balanced_kfold_split , make_torch_generator, stratified_split_subjects
from src.graphs import * #build_graph_bank_from_specs, build_graphs_from_payload, summarize_graph_pool
import src.config as config
from src.train_evaluate import *
from src.models import *
from src.visualize import *
from datetime import datetime
import time



BANK_ENCODERS = {
    "linkx_bank",
    "cnn_bank",
    "gnn_bank"
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--training_approach", type=str, default="segment_all", choices=["segment_k", "segment_all"])
    parser.add_argument("--out_h5", type=str, default=None, required=True, help="out_h5")
    parser.add_argument("--output_root", type=str, default=None, required=True, help="output_root")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--feature_families_str", type=str,  default="relative_band_power,statistical")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument("--test_code", action="store_true")
    parser.add_argument("--base_k", type=nullable_int, default=10, required=False, help="base_k")
    parser.add_argument("--segment_selection_strategy", type=str, default="all_raw", 
        choices=["original_random_k", "global_cluster_random_k", "global_cluster_proportional_random_k", 
                "all_raw", "clean_random_k", "clean_kmeans_k", "all_clean", "clean_weighted_k"])
    parser.add_argument("--candidate_fusion_mode", type=str, default="concat", 
        choices=[
            "concat",
            "gated",
            "mean",
            "context_gated",
            "dim_gated",
            "attn_residual",
            "node_context_gated",
            "node_context_dim_gated",
            "static_learned"
        ]
        )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="linkx",
        choices=["gnn", "cnn_lstm", "gnn_bank","cnn_bank", 'linkx_bank', 'gat', "linkx", "mlp_node"]
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="gatv2",
        choices=["gatv2", "gcn", "sage"]
    )

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--start_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--bank_specs_json",
        type=str,
        default=None,
        help="Path to JSON list of bank candidate specs, or inline JSON string."
    )
    parser.add_argument("--temp", type=float, default=2.0)

    return parser.parse_args()

def run_one_graph_split(
    *,
    args,
    dataset_name,
    evaluation_protocol,
    train_graphs,
    val_graphs,
    test_graphs,
    num_classes,
    class_names,
    seed,
    run_dir,
    fold_index=None,
    topology_names=None,
):
    os.makedirs(run_dir, exist_ok=True)

    summarize_graph_pool(train_graphs, "train_graphs")
    summarize_graph_pool(val_graphs, "val_graphs")
    summarize_graph_pool(test_graphs, "test_graphs")

    encoder_type = args.encoder_type
    encoder_l = encoder_type.lower()

    bank_encoder = is_bank_encoder(encoder_l)
    multiband_encoder = is_multiband_encoder(encoder_l)

    if args.test_batch_size is None:
        test_batch_size = max(16, args.batch_size)
    else:
        test_batch_size = args.test_batch_size

    fold_seed = (
        seed
        if fold_index is None
        else seed * 1000 + fold_index
    )
    set_reproducible(fold_seed)

    train_gen = make_torch_generator(fold_seed + 123)

    if args.training_approach == "segment_k":
        train_dataset = SubjectBalancedSegmentKDataset(
            train_graphs,
            k=args.base_k,
            seed=fold_seed,
            fill_with_replacement=True,
        )
    else:
        train_dataset = GraphSegmentDataset(train_graphs)

    train_eval_dataset = GraphSegmentDataset(train_graphs)
    val_dataset = GraphSegmentDataset(val_graphs)
    test_dataset = GraphSegmentDataset(test_graphs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
        generator=train_gen,
        worker_init_fn=seed_worker,
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )

    # Giữ phần build model hiện có ở đây.
    # Giữ fit_segment_baseline_subject_es() ở đây.
    # Giữ evaluation ở đây.

    return summary_test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    class_set ="all3" 

    channel_names = config.MONO_CHANNELS
    fixed_pairs = config.MONOFIXEDGES
    channel_name = "mono"
    n_channels = 19
    fixed_edges = normalize_fixed_edges(fixed_pairs, n_channels, channel_names)

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
   
    args = parse_args()
    save_path =args.output_root
    os.makedirs(save_path,exist_ok = True)


    if args.test_code:
        data_paths, labels, sub_id_list = data_paths[:15]+data_paths[40:55]+data_paths[75:], labels[:15]+labels[40:55]+labels[75:], sub_id_list[:15]+sub_id_list[40:55]+sub_id_list[75:]
        save_path = os.path.join(save_path,'result-testonly')
        os.makedirs(save_path,exist_ok = True)

    out_h5 = args.out_h5
    topology = args.topology
    test_code = args.test_code
    feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    encoder_type = args.encoder_type
    connectivity_metric=args.connectivity_metric
    connectivity_band = args.connectivity_band
    filter_method=args.topology

    k = 5
    val_ratio = 0.15

    edge_mode="topology_weighted"
    max_k_per_subject = 1000
    seeds_list = [15, 42, 100]
    base_k=args.base_k
    standardize_features=True
    lr = args.lr
    weight_decay = args.weight_decay
    dropout=args.dropout
    dim = args.dim
    start_epoch=args.start_epoch
    patience = args.patience
    batch_size = args.batch_size

    min_delta=1e-3
    top_k=3
    epochs = 100

    attn_dim = dim * 2
    gnn_hidden_dim=dim
    graph_emb_dim=dim*2
    attn_dim=dim*2
    node_hidden_dims=(dim*2, dim)
    edge_hidden_dims=(dim*2, dim)
    branch_emb_dim=dim
    num_candidates = None    
    bank_fusion_mode = "static"
    bank_topology_rule = "union"
    bank_vote_threshold = 0.5
    bank_fusion_temperature = 1.0
    bank_hidden_dim = dim*2
    candidate_fusion_hidden_dim = dim*2
    candidate_fusion_dropout = 0.0
    share_linkx_weights = False
    backbone = args.backbone
    use_gcn_norm = False
    bank_specs = load_bank_specs(args.bank_specs_json)

    if encoder_type in BANK_ENCODERS:

        if bank_specs is not None:
            print("\n[BANK SPECS] Using user-provided bank:")
            for i, spec in enumerate(bank_specs):
                print(f"  cand {i}: {spec}")
        else:
            print("\n[BANK SPECS] No --bank_specs_json provided; using default bank specs.")
            bank_specs = [
                {"name": "wpli_theta_full", 
                "connectivity_metric": "wpli", 
                "connectivity_band": 1, 
                "filter_method": "full"},
                {"name": "wpli_alpha_fixed", 
                "connectivity_metric": "wpli", 
                "connectivity_band": 2, 
                "filter_method": "fixed"},
                {"name": "coherence_alpha_combined", 
                "connectivity_metric": "coherence", 
                "connectivity_band": 2, 
                "filter_method": "combined"},
                {"name": "coherence_theta_topk4", 
                "connectivity_metric": "coherence", 
                "connectivity_band": 1, 
                "filter_method": "topk"},
            ]
    candidate_fusion_mode = args.candidate_fusion_mode
    segment_selection_strategy = args.segment_selection_strategy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.training_approach}_"
        f"{args.encoder_type}_temp{args.temp}_{args.connectivity_metric}_{args.topology}_{args.candidate_fusion_mode}"
    )
    

    output_dir = os.path.join(save_path,f"{timestamp}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"log.txt")


    with open(log_path, "w") as f:
        f.write(f"data source {out_h5}\n")
        f.write(f"seeds_list {seeds_list}\n")
        f.write(f"bank_specs {bank_specs}\n")
        f.write(f"candidate_fusion_mode: {candidate_fusion_mode}\n")
        f.write(f"topology: {filter_method}, fixed_edges: {fixed_edges}, channel_names: {channel_names}\n")
        f.write(f"feature_families: {feature_families}\nconnectivity_metric: {connectivity_metric}, connectivity_band: {connectivity_band}\n")
        f.write(f"model_name: {encoder_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta={min_delta}, top_k={top_k} \n")
        f.write(f"batch_size {batch_size}, lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        f.write(f"graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")

    if encoder_type in BANK_ENCODERS:
        required_connectivity_metrics = collect_required_connectivity_metrics(
            bank_specs=bank_specs,
            default_connectivity_metric=connectivity_metric,
        )
    else:
        required_connectivity_metrics = [connectivity_metric]


    payload_connectivity_band = None if encoder_type in BANK_ENCODERS else connectivity_band

    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=sub_id_list,
        feature_families=feature_families,
        connectivity_metrics=required_connectivity_metrics,
        connectivity_band=payload_connectivity_band,
        load_raw_for_alignment=True,
        load_bad_segment_flag=False,
    )


    all_result_rows = []
    agg_seed_results = []


    if args.test_code:
        epochs=1
        seeds_list = [15]


    for seed in seeds_list:

        set_global_seed(seed)

        print(f"\n========== Split seed: {seed} ==========")
        seed_dir = os.path.join(output_dir, f"seed{seed}")
        os.makedirs(seed_dir,exist_ok = True)
        all_folds = balanced_kfold_split(sub_id_list, labels, seed, k)

        check_dir = os.path.join(f"{seed_dir}/checkpoints")
        os.makedirs(check_dir,exist_ok=True)
        all_fold_data = []


        for i, test_subjects in enumerate(all_folds):
            fold_seed = int(seed) * 1000 + int(i)
            set_reproducible(fold_seed)

            train_gen = make_torch_generator(fold_seed + 123)
            print(f"\n========== Fold: {i} ==========")
            run_dir = os.path.join(seed_dir, f"fold{i}")
            os.makedirs(run_dir, exist_ok=True)

            attention_dir = os.path.join(run_dir, "bank_attention")
            os.makedirs(attention_dir, exist_ok=True)

            print(test_subjects)
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]
            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            subject_label_map = dict(zip(train_subjects, train_labels))
       
            new_train_subjects, val_subjects = stratified_split_subjects(
                train_subjects, subject_label_map, val_ratio, seed
            )
   
            print(f"# Train_subjects = {len(new_train_subjects)} | # Validation subjects = {len(val_subjects)}")

            if encoder_type in BANK_ENCODERS:
                train_graphs, topology_names = build_graph_bank_from_specs(
                    payload,
                    new_train_subjects,
                    feature_families=feature_families,
                    default_connectivity_metric=connectivity_metric,
                    default_connectivity_band=None,
                    default_filter_method=filter_method,
                    default_fixed_edges=fixed_edges,
                    channel_names=channel_names,
                    bank_specs=bank_specs,
                    standardize_features=True,
                )
                val_graphs, _ = build_graph_bank_from_specs(
                    payload,
                    val_subjects,
                    feature_families=feature_families,
                    default_connectivity_metric=connectivity_metric,
                    default_connectivity_band=None,
                    default_filter_method=filter_method,
                    default_fixed_edges=fixed_edges,
                    channel_names=channel_names,
                    bank_specs=bank_specs,
                    standardize_features=True,
                )
                test_graphs, _ = build_graph_bank_from_specs(
                    payload,
                    test_subjects,
                    feature_families=feature_families,
                    default_connectivity_metric=connectivity_metric,
                    default_connectivity_band=None,
                    default_filter_method=filter_method,
                    default_fixed_edges=fixed_edges,
                    channel_names=channel_names,
                    bank_specs=bank_specs,
                    standardize_features=True,
                )
                num_candidates = len(topology_names)
            # elif encoder_type not in ["linkx_cnn5", "cnn5"]:

            else:
                train_graphs = build_graphs_from_payload(
                    payload, new_train_subjects,
                    feature_families=feature_families,
                    connectivity_metric=connectivity_metric,
                    connectivity_band=connectivity_band,
                    filter_method=filter_method,
                    fixed_edges=fixed_edges,          # from config
                    channel_names=channel_names,      # whatever list you use for this payload
                    undirected=True,
                    standardize_features=True,       # or True if desired
                )
                val_graphs = build_graphs_from_payload(
                    payload, val_subjects,
                    feature_families=feature_families,
                    connectivity_metric=connectivity_metric,
                    connectivity_band=connectivity_band,
                    filter_method=filter_method,
                    fixed_edges=fixed_edges,          # from config
                    channel_names=channel_names,      # whatever list you use for this payload
                    undirected=True,
                    standardize_features=True,       # or True if desired
                )
                test_graphs = build_graphs_from_payload(
                    payload, test_subjects,
                    feature_families=feature_families,
                    connectivity_metric=connectivity_metric,
                    connectivity_band=connectivity_band,
                    filter_method=filter_method,
                    fixed_edges=fixed_edges,          # from config
                    channel_names=channel_names,      # whatever list you use for this payload
                    undirected=True,
                    standardize_features=True,       # or True if desired
                )

                # train_graphs = build_graphs_from_payload_multiband(
                #     payload, new_train_subjects,
                #     feature_families=feature_families,
                #     connectivity_metric=connectivity_metric,
                # )
                # val_graphs = build_graphs_from_payload_multiband(
                #     payload, val_subjects,
                #     feature_families=feature_families,
                #     connectivity_metric=connectivity_metric,
                # )
                # test_graphs = build_graphs_from_payload_multiband(
                #     payload, test_subjects,
                #     feature_families=feature_families,
                #     connectivity_metric=connectivity_metric,
                # )

            summarize_graph_pool(train_graphs, "train_graphs_original")
            summarize_graph_pool(val_graphs, "val_graphs_original")
            summarize_graph_pool(test_graphs, "test_graphs_original")


            selection_strategy = str(segment_selection_strategy).lower()
            train_dataset_mode = "fixed_all_selected"

            training_approach = "segment_all"
            # if training_approach in ["segment_all", "segment_k"]:

            encoder_l = str(encoder_type).lower()
            bank_encoder = is_bank_encoder(encoder_l)
            multiband_encoder = is_multiband_encoder(encoder_l)
            test_batch_size = batch_size * 4

                # if training_approach == "segment_k":
                #     train_dataset = SubjectBalancedSegmentKDataset(
                #         train_graphs,
                #         k=base_k,
                #         seed=seed,
                #         fill_with_replacement=True,
                #     )
                # else:
            train_dataset = GraphSegmentDataset(train_graphs)
            val_dataset = GraphSegmentDataset(val_graphs)
            test_dataset = GraphSegmentDataset(test_graphs)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_graph_segments,
                num_workers=0,
                pin_memory=True,
                generator=train_gen,
                worker_init_fn=seed_worker,
            )
            train_eval_loader = DataLoader(
                GraphSegmentDataset(train_graphs),
                batch_size=test_batch_size,
                shuffle=False,
                collate_fn=collate_graph_segments,
                num_workers=0,
                pin_memory=True,
                generator=train_gen,
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                collate_fn=collate_graph_segments,
                num_workers=0,
                pin_memory=True,
                generator=train_gen,
                worker_init_fn=seed_worker,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                collate_fn=collate_graph_segments,
                num_workers=0,
                pin_memory=True,
                generator=train_gen,
                worker_init_fn=seed_worker,
            )

            first_graph = train_graphs[0]

            model = SegmentGraphClassifierFromMIL(
                num_node_features=int(first_graph.x.shape[-1]),
                num_classes=num_classes,
                num_nodes=int(first_graph.x.shape[0]),
                encoder_type=encoder_type,
                edge_mode=edge_mode,
                graph_emb_dim=graph_emb_dim,
                dropout=dropout,
                attn_dim=attn_dim,
                temp=args.temp,
                gnn_hidden_dim=gnn_hidden_dim,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                cnn_num_bands=getattr(first_graph, "conn_stack", torch.empty(0)).shape[0]
                    if (multiband_encoder or bank_encoder) else None,
                num_candidates=getattr(
                    first_graph,
                    "adj_bank",
                    getattr(first_graph, "conn_stack", torch.empty(0)),
                ).shape[0] if bank_encoder else None,
                candidate_fusion_mode=candidate_fusion_mode,
                candidate_fusion_hidden_dim=candidate_fusion_hidden_dim,
                candidate_fusion_dropout=candidate_fusion_dropout,
                # bank_fusion_mode=bank_fusion_mode,
                graph_backbone=backbone,
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            ckpt_path = os.path.join(run_dir, "best_model.pt")

            model, val_metrics, history, best_state = fit_segment_baseline_subject_es(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                epochs=epochs,
                patience=patience,
                start_epoch=start_epoch,
                min_delta=min_delta,
                top_k=top_k,
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
            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

            train_subject_df.to_csv(os.path.join(run_dir, "train_predictions_subject_agg.csv"), index=False)
            val_subject_df.to_csv(os.path.join(run_dir, "val_predictions_subject_agg.csv"), index=False)
            test_subject_df.to_csv(os.path.join(run_dir, "test_predictions_subject_agg.csv"), index=False)

            train_segment_df.to_csv(os.path.join(run_dir, "train_predictions_segment.csv"), index=False)
            val_segment_df.to_csv(os.path.join(run_dir, "val_predictions_segment.csv"), index=False)
            test_segment_df.to_csv(os.path.join(run_dir, "test_predictions_segment.csv"), index=False)
            
            if encoder_type in {"linkx_bank", "gnn_bank"} and candidate_fusion_mode not in {"concat", "mean"}:

                all_long = []
                all_summary = []

                for split_name, attn_loader in [
                        ("train", train_eval_loader),
                        ("val", val_loader),
                        ("test", test_loader),
                    ]:
                    long_df, summary_df = collect_bank_attention_segment_level(
                        model=model,
                        loader=attn_loader,
                        device=device,
                        split_name=split_name,
                        candidate_names=topology_names if "topology_names" in locals() else None,
                        num_classes=num_classes,
                    )

                    long_df.to_csv(os.path.join(attention_dir, f"{split_name}_bank_attention_long.csv"), index=False)
                    summary_df.to_csv(os.path.join(attention_dir, f"{split_name}_bank_attention_summary.csv"), index=False)

                    all_long.append(long_df)
                    all_summary.append(summary_df)

                all_long_df = pd.concat(all_long, ignore_index=True)
                all_summary_df = pd.concat(all_summary, ignore_index=True)

                all_long_df.to_csv(os.path.join(attention_dir, "all_splits_bank_attention_long.csv"), index=False)
                all_summary_df.to_csv(os.path.join(attention_dir, "all_splits_bank_attention_summary.csv"), index=False)

                save_bank_attention_plots(
                    long_df=all_long_df,
                    summary_df=all_summary_df,
                    out_dir=attention_dir,
                    class_names=class_names if "class_names" in locals() else None,
                )
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
            summary_df= pd.DataFrame(summary_rows)
            summary_df.to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False)

            summary_test = [{
                "encoder_type": encoder_type,
                "training_approach": training_approach,
                "segment_selection_strategy": segment_selection_strategy,
                "feature_families": feature_families,
                "accuracy": float(test_metrics["accuracy"]),
                "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "confusion_matrix": test_metrics["conf_matrix"],
                "topology": filter_method,
                "connectivity_metric": connectivity_metric,
                "connectivity_band": connectivity_band,
                "edge_mode": edge_mode,
                "base_k": base_k,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "start_epoch": start_epoch,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "graph_emb_dim": graph_emb_dim,
                "attn_dim": attn_dim,
                "seed": seed,
            }]
            summary_test_df = pd.DataFrame(summary_test)
            summary_test_df.to_csv(os.path.join(run_dir, "summary_test.csv"), index=False)
            all_fold_data.extend(summary_test)


        seed_summary = aggregate_fold_summaries_to_seed(
            all_fold_data,
            seed=seed,
        )

        agg_seed_results.append(seed_summary)
    agg_dir = os.path.join(output_dir, "agg_seed_results.csv")
    seed_df, agg_df = save_seed_aggregation(
        agg_seed_results,
        output_dir=agg_dir,
    )

    print("\nAggregate across seeds:")
    print(agg_df[[
        "accuracy_mean_std",
        "balanced_accuracy_mean_std",
        "macro_f1_mean_std",
    ]])


## caueeg
    args = parse_args()
    feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    feature_families_str = args.feature_families_str.replace(",","_")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    bad_ids = [x.strip() for x in args.bad_ids_str.split(",") if x.strip()] if args.bad_ids_str else None
    step = int(args.crop_len * (1.0 - float(args.overlap)))
    out_h5 = args.out_h5
    if out_h5 is None:
        out_h5 = os.path.join(args.output_root, f"caueeg_{args.task}_{args.file_format}_features.h5")

    channel_names = CAUEEG_EEG19
    fixed_edges = load_fixed_edges_from_config(channel_names)
    
    bank_specs = load_bank_specs(args.bank_specs_json)

    if bank_specs is not None:
        print("\n[BANK SPECS] Using user-provided bank:")
        for i, spec in enumerate(bank_specs):
            print(f"  cand {i}: {spec}")
    else:
        print("\n[BANK SPECS] No --bank_specs_json provided; using default bank specs.")

        bank_specs = [
                {"name": "wpli_theta_full", 
                "connectivity_metric": "wpli", 
                "connectivity_band": 1, 
                "filter_method": "full"},
                {"name": "wpli_alpha_fixed", 
                "connectivity_metric": "wpli", 
                "connectivity_band": 2, 
                "filter_method": "fixed"},
                {"name": "coherence_alpha_combined", 
                "connectivity_metric": "coherence", 
                "connectivity_band": 2, 
                "filter_method": "combined"},
                {"name": "coherence_theta_topk4", 
                "connectivity_metric": "coherence", 
                "connectivity_band": 1, 
                "filter_method": "topk"},
            ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        args.output_root,
        f"{timestamp}_{args.training_approach}_{args.base_k}_{args.encoder_type}_{args.candidate_fusion_mode}_temp{args.temp}",
    )

    os.makedirs(experiment_root, exist_ok=True)

    summary_rows = []
    for seed in seeds:
        out = run_caueeg_linkx_training(
            training_approach=args.training_approach,
            dataset_path=args.dataset_path,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            task=args.task,
            file_format=args.file_format,
            out_h5=out_h5,
            feature_families=feature_families,
            connectivity_metric=args.connectivity_metric,
            connectivity_band=args.connectivity_band,
            encoder_type=args.encoder_type,
            # mil_pool_type=args.mil_pool_type,
            # filter_method=args.topology,
            # segment_selection_strategy=args.segment_selection_strategy,
            # cleancluster_manifest_path=args.cleancluster_manifest_path,
            level=args.level,
            # macro_duration_sec=args.macro_duration_sec,
            # level_reduce=args.level_reduce,
            base_k=args.base_k,
            seed=seed,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            epochs=args.epochs,
            patience=args.patience,
            start_epoch=args.start_epoch,
            min_delta=args.min_delta,
            top_k=args.top_k,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            temp=args.temp,
            graph_emb_dim=args.graph_emb_dim,
            attn_dim=args.attn_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            branch_emb_dim=args.branch_emb_dim,
            edge_mode=args.edge_mode,
            device=args.device,
            rebuild_h5=args.rebuild_h5,
            output_root=experiment_root,
            use_split_prefix=args.use_split_prefix,
            bad_ids=bad_ids,
            crop_len=args.crop_len,
            step=step,
            latency=args.latency,
            test_code=args.test_code,
            test_n_subjects=args.test_n_subjects,
            bank_specs=bank_specs,
            candidate_fusion_mode=args.candidate_fusion_mode,
            candidate_fusion_hidden_dim=args.candidate_fusion_hidden_dim,
            candidate_fusion_dropout=args.candidate_fusion_dropout,
            # bank_fusion_mode=args.bank_fusion_mode,
            graph_backbone=args.graph_backbone,
            # use_gcn_norm=args.use_gcn_norm,
            # late_fusion_mode=args.late_fusion_mode,
            # late_vote_weight=args.late_vote_weight,
            predict_only_ckpt=args.predict_only_ckpt,
            # predict_only_dirname=args.predict_only_dirname
        )
        if isinstance(out["summary_test"], list):
            summary_rows.extend(out["summary_test"])
        else:
            summary_rows.append(out["summary_test"])

    save_seed_aggregation(summary_rows, os.path.join(experiment_root, "agg_seed_results"))
    print(f"Done. Experiment root: {experiment_root}")


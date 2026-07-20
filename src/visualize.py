import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
def save_bank_attention_plots(long_df, summary_df, out_dir, class_names=None):
    os.makedirs(out_dir, exist_ok=True)

    long_df = long_df.copy()
    summary_df = summary_df.copy()

    if class_names is not None:
        label_map = {i: name for i, name in enumerate(class_names)}
        long_df["class_name"] = long_df["true_label"].map(label_map).fillna(long_df["true_label"].astype(str))
        summary_df["class_name"] = summary_df["true_label"].map(label_map).fillna(summary_df["true_label"].astype(str))
    else:
        long_df["class_name"] = long_df["true_label"].astype(str)
        summary_df["class_name"] = summary_df["true_label"].astype(str)

    # --------------------------------------------------
    # 1. Boxplot: attention by candidate, per split
    # --------------------------------------------------
    for split, sdf in long_df.groupby("split"):
        candidates = sdf["candidate_name"].unique().tolist()
        data = [
            sdf.loc[sdf["candidate_name"] == c, "attention"].to_numpy()
            for c in candidates
        ]

        plt.figure(figsize=(max(8, 0.45 * len(candidates)), 5))
        plt.boxplot(data, labels=candidates, showfliers=False)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Attention weight")
        plt.title(f"Candidate attention distribution | {split}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{split}_boxplot_attention_by_candidate.png"), dpi=300)
        plt.close()

    # --------------------------------------------------
    # 2. Boxplot: attention by candidate and class
    # --------------------------------------------------
    for split, sdf_split in long_df.groupby("split"):
        for cls, sdf in sdf_split.groupby("class_name"):
            candidates = sdf["candidate_name"].unique().tolist()
            data = [
                sdf.loc[sdf["candidate_name"] == c, "attention"].to_numpy()
                for c in candidates
            ]

            safe_cls = str(cls).replace("/", "_")
            plt.figure(figsize=(max(8, 0.45 * len(candidates)), 5))
            plt.boxplot(data, labels=candidates, showfliers=False)
            plt.xticks(rotation=60, ha="right")
            plt.ylabel("Attention weight")
            plt.title(f"Candidate attention | split={split} | class={cls}")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"{split}_class_{safe_cls}_boxplot_attention_by_candidate.png"),
                dpi=300,
            )
            plt.close()

    # --------------------------------------------------
    # 3. Histogram: max attention
    # --------------------------------------------------
    for split, sdf in summary_df.groupby("split"):
        plt.figure(figsize=(7, 5))
        plt.hist(sdf["max_attention"].to_numpy(), bins=25)
        plt.xlabel("Max candidate attention per segment")
        plt.ylabel("Number of segments")
        plt.title(f"Max attention distribution | {split}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{split}_hist_max_attention.png"), dpi=300)
        plt.close()

    # --------------------------------------------------
    # 4. Histogram: normalized entropy
    # --------------------------------------------------
    for split, sdf in summary_df.groupby("split"):
        plt.figure(figsize=(7, 5))
        plt.hist(sdf["normalized_entropy"].to_numpy(), bins=25)
        plt.xlabel("Normalized attention entropy")
        plt.ylabel("Number of segments")
        plt.title(f"Attention entropy distribution | {split}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{split}_hist_normalized_entropy.png"), dpi=300)
        plt.close()

    # --------------------------------------------------
    # 5. Histogram: effective number of candidates
    # --------------------------------------------------
    for split, sdf in summary_df.groupby("split"):
        plt.figure(figsize=(7, 5))
        plt.hist(sdf["effective_num_candidates"].to_numpy(), bins=25)
        plt.xlabel("Effective number of candidates")
        plt.ylabel("Number of segments")
        plt.title(f"Effective candidate count | {split}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{split}_hist_effective_num_candidates.png"), dpi=300)
        plt.close()

    # --------------------------------------------------
    # 6. Dominant candidate frequency by split
    # --------------------------------------------------
    dom = pd.crosstab(
        summary_df["split"],
        summary_df["dominant_candidate_name"],
        normalize="index",
    )

    ax = dom.plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_ylabel("Fraction of segments")
    ax.set_title("Dominant candidate frequency by split")
    ax.legend(title="Dominant candidate", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dominant_candidate_frequency_by_split.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 7. Dominant candidate frequency by class
    # --------------------------------------------------
    dom_class = pd.crosstab(
        summary_df["class_name"],
        summary_df["dominant_candidate_name"],
        normalize="index",
    )

    ax = dom_class.plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_ylabel("Fraction of segments")
    ax.set_title("Dominant candidate frequency by class")
    ax.legend(title="Dominant candidate", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dominant_candidate_frequency_by_class.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # 8. Subject-level mean attention heatmap
    # --------------------------------------------------
    subj_mean = (
        long_df
        .groupby(["split", "subject_id", "candidate_name"])["attention"]
        .mean()
        .reset_index()
    )

    for split, sdf in subj_mean.groupby("split"):
        mat = sdf.pivot(index="subject_id", columns="candidate_name", values="attention").fillna(0.0)

        # Sort by dominant candidate to make collapse easier to see.
        dom_idx = mat.to_numpy().argmax(axis=1)
        order = np.argsort(dom_idx)
        mat = mat.iloc[order]

        plt.figure(figsize=(max(8, 0.45 * mat.shape[1]), max(5, 0.12 * mat.shape[0])))
        plt.imshow(mat.to_numpy(), aspect="auto")
        plt.colorbar(label="Mean attention")
        plt.xticks(np.arange(mat.shape[1]), mat.columns.tolist(), rotation=60, ha="right")
        plt.yticks([])
        plt.xlabel("Candidate")
        plt.ylabel("Subjects")
        plt.title(f"Subject-level mean candidate attention | {split}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{split}_subject_candidate_attention_heatmap.png"), dpi=300)
        plt.close()
def summarize_attention_collapse(summary_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = summary_df.copy()

    # Adjustable thresholds.
    df["collapsed_max080"] = df["max_attention"] >= 0.80
    df["collapsed_entropy035"] = df["normalized_entropy"] <= 0.35
    df["collapsed_eff150"] = df["effective_num_candidates"] <= 1.50

    rows = []

    for keys, sdf in df.groupby(["split"]):
        rows.append({
            "split": keys,
            "n_segments": int(len(sdf)),
            "mean_max_attention": float(sdf["max_attention"].mean()),
            "median_max_attention": float(sdf["max_attention"].median()),
            "mean_normalized_entropy": float(sdf["normalized_entropy"].mean()),
            "median_normalized_entropy": float(sdf["normalized_entropy"].median()),
            "mean_effective_num_candidates": float(sdf["effective_num_candidates"].mean()),
            "frac_max_attention_ge_0.80": float(sdf["collapsed_max080"].mean()),
            "frac_entropy_le_0.35": float(sdf["collapsed_entropy035"].mean()),
            "frac_effective_k_le_1.50": float(sdf["collapsed_eff150"].mean()),
        })

    split_summary = pd.DataFrame(rows)
    split_summary.to_csv(os.path.join(out_dir, "attention_collapse_summary_by_split.csv"), index=False)

    rows = []
    for (split, label), sdf in df.groupby(["split", "true_label"]):
        rows.append({
            "split": split,
            "true_label": int(label),
            "n_segments": int(len(sdf)),
            "mean_max_attention": float(sdf["max_attention"].mean()),
            "mean_normalized_entropy": float(sdf["normalized_entropy"].mean()),
            "mean_effective_num_candidates": float(sdf["effective_num_candidates"].mean()),
            "frac_max_attention_ge_0.80": float(sdf["collapsed_max080"].mean()),
            "frac_entropy_le_0.35": float(sdf["collapsed_entropy035"].mean()),
            "frac_effective_k_le_1.50": float(sdf["collapsed_eff150"].mean()),
        })

    class_summary = pd.DataFrame(rows)
    class_summary.to_csv(os.path.join(out_dir, "attention_collapse_summary_by_split_class.csv"), index=False)

    return split_summary, class_summary

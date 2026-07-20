from __future__ import annotations

import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse, to_dense_batch
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GraphNorm, global_mean_pool
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def collate_graph_segments(batch: Sequence[Data]) -> Dict[str, Any]:
    graphs = list(batch)
    pyg_batch = Batch.from_data_list(graphs)
    labels = torch.tensor([int(g.y.view(-1)[0].item()) for g in graphs], dtype=torch.long)
    subject_ids = [str(getattr(g, "subject_id", "")) for g in graphs]
    segment_ids = [int(getattr(g, "segment_id", -1)) for g in graphs]
    start_samples = [int(getattr(g, "start_sample", -1)) for g in graphs]

    full_adj = []
    conn_stack = []
    adj_bank = []
    topology_bank = []
    topology_names = None

    for g in graphs:
        if hasattr(g, "adj") and g.adj is not None:
            a = g.adj.detach().cpu().float() if torch.is_tensor(g.adj) else torch.tensor(g.adj, dtype=torch.float32)
            full_adj.append(a)
        if hasattr(g, "conn_stack") and g.conn_stack is not None:
            cs = g.conn_stack.detach().cpu().float() if torch.is_tensor(g.conn_stack) else torch.tensor(g.conn_stack, dtype=torch.float32)
            conn_stack.append(cs)
        if hasattr(g, "adj_bank") and g.adj_bank is not None:
            ab = g.adj_bank.detach().cpu().float() if torch.is_tensor(g.adj_bank) else torch.tensor(g.adj_bank, dtype=torch.float32)
            adj_bank.append(ab)
        if hasattr(g, "topology_bank") and g.topology_bank is not None:
            tb = g.topology_bank.detach().cpu().float() if torch.is_tensor(g.topology_bank) else torch.tensor(g.topology_bank, dtype=torch.float32)
            topology_bank.append(tb)
        if topology_names is None and hasattr(g, "topology_names"):
            topology_names = list(g.topology_names)

    out: Dict[str, Any] = {
        "pyg_batch": pyg_batch,
        "labels": labels,
        "subject_ids": subject_ids,
        "segment_ids": segment_ids,
        "start_samples": start_samples,
    }
    if len(full_adj) == len(graphs):
        out["full_adj"] = torch.stack(full_adj, dim=0)
    if len(conn_stack) == len(graphs):
        out["conn_stack"] = torch.stack(conn_stack, dim=0)
    if len(adj_bank) == len(graphs):
        out["adj_bank"] = torch.stack(adj_bank, dim=0)
    if len(topology_bank) == len(graphs):
        out["topology_bank"] = torch.stack(topology_bank, dim=0)
    if topology_names is not None:
        out["topology_names"] = topology_names
    return out



def summarize_graph_pool(graphs, name: str):

    subject_to_count = defaultdict(int)
    label_to_subjects = defaultdict(set)

    for g in graphs:
        sid = str(g.subject_id)
        y = int(g.y.view(-1)[0].item())
        subject_to_count[sid] += 1
        label_to_subjects[y].add(sid)

    counts = np.array(list(subject_to_count.values()), dtype=np.int64)

    print(f"\n[{name}]")
    print(f"num graphs: {len(graphs)}")
    print(f"num subjects: {len(subject_to_count)}")
    print(f"segments per subject: min={counts.min()}, mean={counts.mean():.2f}, max={counts.max()}")
    print("subjects per label:", {k: len(v) for k, v in label_to_subjects.items()})


def _to_2d_features(x) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError(f"Expected node features [N, F], got shape {tuple(x.shape)}")
    return x


def _zscore_per_feature(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-graph standardization across nodes, feature by feature.
    x: [num_nodes, num_node_features]
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _maximum_spanning_tree_edges(
    edge_list: Sequence[Tuple[int, int]],
    edge_weights: Sequence[float],
    n_channels: int,
) -> set[tuple[int, int]]:

    parent = list(range(n_channels))
    rank = [0] * n_channels

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    edges = [
        (int(i), int(j), float(w))
        for (i, j), w in zip(edge_list, edge_weights)
    ]
    edges.sort(key=lambda x: x[2], reverse=True)

    chosen = set()
    for i, j, _ in edges:
        if union(i, j):
            if i > j:
                i, j = j, i
            chosen.add((i, j))

    return chosen

def _topk_per_node_edges(
    edge_list: Sequence[Tuple[int, int]],
    edge_weights: Sequence[float],
    n_channels: int,
    topk: int = 4,
    score_mode: str = "raw",
) -> set[tuple[int, int]]:

    if topk is None:
        raise ValueError("topk must be provided for per-node top-k.")

    k = int(topk)
    if k < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")

    neighbors = {i: [] for i in range(int(n_channels))}

    for (i, j), w in zip(edge_list, edge_weights):
        i, j = int(i), int(j)
        w = float(w)

        if score_mode == "abs":
            score = abs(w)
        elif score_mode == "raw":
            score = w
        else:
            raise ValueError("score_mode must be 'raw' or 'abs'.")

        neighbors[i].append((j, score))
        neighbors[j].append((i, score))

    selected = set()

    for i in range(int(n_channels)):
        cand = sorted(neighbors[i], key=lambda x: x[1], reverse=True)
        for j, _ in cand[:k]:
            a, b = sorted((i, int(j)))
            selected.add((a, b))

    return selected
    
def _topk_edges(
    edge_list: Sequence[Tuple[int, int]],
    edge_weights: Sequence[float],
    topk: Optional[int] = None,
    top_percent: Optional[float] = None,
) -> set[tuple[int, int]]:
    if len(edge_list) == 0:
        return set()

    if topk is None and top_percent is None:
        raise ValueError("Provide topk or top_percent for filter_method='topk'")

    pairs = [(tuple(map(int, e)), float(w)) for e, w in zip(edge_list, edge_weights)]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if top_percent is not None:
        if not (0 < top_percent <= 1):
            raise ValueError(f"top_percent must be in (0, 1], got {top_percent}")
        k = max(1, int(np.ceil(len(pairs) * top_percent)))
    else:
        k = int(topk)
        if k < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

    k = min(k, len(pairs))
    return {tuple(sorted(p[0])) for p in pairs[:k]}

def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    channel_names: Optional[Sequence[str]],
    n_channels: int,
) -> set[tuple[int, int]]:
    if fixed_edges is None:
        return set()

    out = set()

    if channel_names is not None:
        name_to_idx = {str(ch): i for i, ch in enumerate(channel_names)}
    else:
        name_to_idx = None

    for a, b in fixed_edges:
        if isinstance(a, str) or isinstance(b, str):
            if name_to_idx is None:
                raise ValueError("fixed_edges contains channel names but channel_names was not provided.")
            if a not in name_to_idx or b not in name_to_idx:
                raise ValueError(f"Unknown channel in fixed_edges: {(a, b)}")
            i, j = name_to_idx[a], name_to_idx[b]
        else:
            i, j = int(a), int(b)

        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(a, b)} resolved to invalid node indices {(i, j)}")

        if i == j:
            continue

        if i > j:
            i, j = j, i

        out.add((i, j))

    return out

def dense_adj_to_candidate_edges(adj: torch.Tensor, undirected: bool = True):
    """
    Convert dense adjacency to candidate undirected edges + weights.

    Returns
    -------
    edge_index : torch.LongTensor [2, E_sparse]
        Symmetric sparse edge_index for the current dense adjacency.
    edge_attr : torch.FloatTensor [E_sparse]
        Symmetric edge weights matching edge_index.
    edge_list : list[tuple[int, int]]
        Undirected edge list with i < j.
    edge_weights : list[float]
        One weight per undirected edge in edge_list.
    """
    adj = torch.as_tensor(adj, dtype=torch.float32)
    n = int(adj.shape[0])

    if undirected:
        iu = torch.triu_indices(n, n, offset=1)
        w = adj[iu[0], iu[1]]
        mask = torch.abs(w) > 1e-12

        row = iu[0][mask]
        col = iu[1][mask]
        w = w[mask]

        edge_list = [(int(i), int(j)) for i, j in zip(row.tolist(), col.tolist())]
        edge_weights = [float(x) for x in w.tolist()]

        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.cat(
                [
                    torch.stack([row, col], dim=0),
                    torch.stack([col, row], dim=0),
                ],
                dim=1,
            ).long()
            edge_attr = torch.cat([w, w], dim=0).float()

    else:
        row, col = torch.nonzero(torch.abs(adj) > 1e-12, as_tuple=True)
        w = adj[row, col]
        edge_index = torch.stack([row, col], dim=0).long()
        edge_attr = w.float()
        edge_list = [(int(i), int(j)) for i, j in zip(row.tolist(), col.tolist())]
        edge_weights = [float(x) for x in w.tolist()]

    return edge_index, edge_attr, edge_list, edge_weights


def _build_dense_adj_from_selected_edges(
    selected_edges: set[tuple[int, int]],
    edge_to_weight: dict[tuple[int, int], float],
    n_channels: int,
    undirected: bool = True,
) -> np.ndarray:
    adj = np.zeros((n_channels, n_channels), dtype=np.float32)

    for i, j in selected_edges:
        w = float(edge_to_weight[(i, j)])
        adj[i, j] = w
        if undirected:
            adj[j, i] = w

    np.fill_diagonal(adj, 0.0)
    return adj



def apply_edge_filter(
    edge_index,
    edge_attr,
    edge_list,
    edge_weights,
    n_channels: int,
    filter_method: str = "mst",
    topk: Optional[int] = 4,
    top_percent: Optional[float] = None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    undirected: bool = True,
):

    method = str(filter_method).lower()
    edge_to_weight = {
        tuple(sorted((int(i), int(j)))): float(w)
        for (i, j), w in zip(edge_list, edge_weights)
    }

    full_edges = set(edge_to_weight.keys())
    fixed_set = _normalize_fixed_edges(
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        n_channels=n_channels,
    )

    if method in {"full", "none", "dense", "all"}:
        selected = full_edges

    elif method in {"mst", "maxst", "maximum_spanning_tree"}:
        selected = _maximum_spanning_tree_edges(edge_list, edge_weights, n_channels=n_channels)

    elif method == "fixed":
        selected = fixed_set

    elif method == "topk":
        selected = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)

    elif method == "reconnect":
        mst_set = _maximum_spanning_tree_edges(edge_list, edge_weights, n_channels=n_channels)
        selected = fixed_set | mst_set

    elif method == "combined":
        topk_set = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)
        selected = fixed_set | topk_set

    elif method == "overlap":
        topk_set = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)
        selected = fixed_set & topk_set
    elif method == "topk_node":
        selected = _topk_per_node_edges(
            edge_list,
            edge_weights,
            n_channels=n_channels,
            topk=topk,
            score_mode="raw",
        )

    elif method in {"combined_node", "fixed_topk_node"}:
        topk_set = _topk_per_node_edges(
            edge_list,
            edge_weights,
            n_channels=n_channels,
            topk=topk,
            score_mode="raw",
        )
        selected = fixed_set | topk_set
    else:
        raise ValueError(f"Unknown filter_method={filter_method!r}")

    selected = {e for e in selected if e in edge_to_weight}

    final_adj = _build_dense_adj_from_selected_edges(
        selected_edges=selected,
        edge_to_weight=edge_to_weight,
        n_channels=n_channels,
        undirected=undirected,
    )

    final_edge_index, final_edge_weight = dense_to_sparse(torch.tensor(final_adj, dtype=torch.float32))
    return final_edge_index.long(), final_edge_weight.float(), final_adj



def build_graphs_from_payload(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    # edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    undirected=True,
    filter_method="mst",             # "mst", "fixed", "topk", "reconnect", "combined", "overlap", "full"
    topk=4,
    top_percent=None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    # corruption_mode=None,            # None, "identity", "random", "permute_consistent", "permute_adj_only"
    standardize_features=True,
    # region_to_channels=None,
    # hyperedge_weight_mode="mean_abs_adj",
    # clique_combine_mode="sum",
    # keep_empty_hyperedges=False,
    # add_graph_theory_to_node_features=False,
):
    """
    Build one PyG graph per window from payload, using the selected topology
    instead of blindly converting the full adjacency to sparse.

    payload[sid] must contain:
      - "label"
      - "features"[family] -> [W, N, F_family]
      - "segment_id" -> [W]
      - "start_sample" -> [W]
      - optionally:
          "connectivity"[metric] -> [W, N, N]
          "channel_names" -> list[str]
    """
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        # -------------------------------------------------
        # node features
        # -------------------------------------------------
        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")

            xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
            if xfam.ndim != 3:
                raise ValueError(
                    f"Feature family {fam!r} for subject {sid!r} must have shape [W, N, F], got {xfam.shape}"
                )

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                    raise ValueError(
                        f"Feature family {fam!r} for subject {sid!r} has incompatible shape {xfam.shape}; "
                        f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                    )

            feat_list.append(xfam)

        if len(feat_list) == 0:
            raise ValueError("feature_families is empty")

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
        num_windows = node_x_all.shape[0]
        num_nodes = node_x_all.shape[1]

        # -------------------------------------------------
        # metadata
        # -------------------------------------------------
        seg_ids = np.asarray(subj.get("segment_id", np.arange(num_windows)), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(num_windows, -1)), dtype=np.int64)

        if len(seg_ids) != num_windows:
            raise ValueError(
                f"segment_id length mismatch for subject {sid!r}: got {len(seg_ids)}, expected {num_windows}"
            )
        if len(start_samples) != num_windows:
            raise ValueError(
                f"start_sample length mismatch for subject {sid!r}: got {len(start_samples)}, expected {num_windows}"
            )

        # prefer subject-level channel names if available
        entry_channel_names = subj.get("channel_names", channel_names)

        # -------------------------------------------------
        # adjacency source
        # -------------------------------------------------
        # if edge_source == "connectivity":
        if connectivity_metric is None:
            adj_all = None
        else:
            if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
                raise KeyError(
                    f"payload[{sid!r}]['connectivity'] missing metric {connectivity_metric!r}"
                )
            adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

            if adj_all.ndim == 4:
                if connectivity_band is None:
                    raise ValueError(
                        f"Connectivity for {sid!r}/{connectivity_metric!r} is banded [W,B,N,N], "
                        "but connectivity_band is None."
                    )
                band_idx = int(connectivity_band) if not isinstance(connectivity_band, str) else connectivity_band
                if isinstance(band_idx, str):
                    raise ValueError(
                        "String band selection is not supported here unless you also pass band-name metadata. "
                        "Prefer slicing earlier in load_h5_payload_for_subjects(...)."
                    )
                adj_all = adj_all[:, band_idx]

        # else:
        #     raise ValueError(f"Unsupported edge_source={edge_source!r}")

        if adj_all is not None:
            if adj_all.ndim != 3:
                raise ValueError(
                    f"Adjacency tensor for subject {sid!r} must have shape [W, N, N], got {adj_all.shape}"
                )
            if adj_all.shape[0] != num_windows:
                raise ValueError(
                    f"Adjacency window count mismatch for subject {sid!r}: {adj_all.shape[0]} vs {num_windows}"
                )
            if adj_all.shape[1] != num_nodes or adj_all.shape[2] != num_nodes:
                raise ValueError(
                    f"Adjacency node count mismatch for subject {sid!r}: {adj_all.shape} vs num_nodes={num_nodes}"
                )

        # -------------------------------------------------
        # build one graph per window
        # -------------------------------------------------
        for w in range(num_windows):
            x = _to_2d_features(node_x_all[w])   # [N, F]

            if standardize_features:
                x = torch.from_numpy(_zscore_per_feature(x.numpy()))

            if adj_all is None:
                adj_full = np.eye(num_nodes, dtype=np.float32)
            else:
                adj_full = np.asarray(adj_all[w], dtype=np.float32).copy()

            if symmetrize_adj:
                adj_full = 0.5 * (adj_full + adj_full.T)

            if zero_diagonal:
                np.fill_diagonal(adj_full, 0.0)

            adj_full = np.nan_to_num(adj_full, nan=0.0, posinf=0.0, neginf=0.0)
            adj_full_t = torch.tensor(adj_full, dtype=torch.float32)

            # # --------------------------------
            # # optional corruption on full adj
            # # --------------------------------
            # if corruption_mode == "identity":
            #     adj_used = make_identity_adj(num_nodes)

            # elif corruption_mode == "random":
            #     adj_used = make_random_adj_like_with_weights(adj_full_t, undirected=undirected)

            # elif corruption_mode == "permute_consistent":
            #     x, adj_used, _ = permute_graph_consistently(x, adj_full_t)

            # elif corruption_mode == "permute_adj_only":
            #     adj_used, _ = permute_adj_only(adj_full_t)

            # elif corruption_mode is None:
            adj_used = adj_full_t.clone()

            # else:
            #     raise ValueError(f"Unknown corruption_mode={corruption_mode}")

            adj_used = adj_used.clone()
            if zero_diagonal:
                adj_used.fill_diagonal_(0.0)

            # --------------------------------
            # candidate edges from current full matrix
            # --------------------------------
            edge_index, edge_attr, edge_list, edge_weights = dense_adj_to_candidate_edges(
                adj_used,
                undirected=undirected,
            )

            # --------------------------------
            # apply topology filter
            # --------------------------------
            final_edge_index, final_edge_weight, final_adj = apply_edge_filter(
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_list=edge_list,
                edge_weights=edge_weights,
                n_channels=num_nodes,
                filter_method=filter_method,
                topk=topk,
                top_percent=top_percent,
                fixed_edges=fixed_edges,
                channel_names=entry_channel_names,
                undirected=undirected,
            )

            # --------------------------------
            # node augmentation from filtered topology
            # --------------------------------
            x_np = x.detach().cpu().numpy().astype(np.float32)

            # if add_graph_theory_to_node_features:
            #     x_aug, _ = append_weighted_graph_theory_to_node_features(
            #         node_features=x_np,
            #         adj=final_adj,          # <-- filtered topology, not full dense adj
            #         signed_input=False,
            #     )
            # else:
            #     x_aug = x_np

            g = Data(
                x=torch.tensor(x_np, dtype=torch.float32),
                edge_index=final_edge_index,
                y=torch.tensor([label], dtype=torch.long),
            )

            # pipeline compatibility
            g.edge_weight = final_edge_weight
            g.edge_attr = final_edge_weight.view(-1, 1)

            if attach_dense_adj:
                g.adj = torch.tensor(final_adj, dtype=torch.float32)

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            graphs.append(g)

    return graphs

def build_graph_bank_from_specs(
    payload,
    subject_ids,
    *,
    feature_families,
    default_connectivity_metric,
    default_connectivity_band,
    default_filter_method,
    default_fixed_edges,
    channel_names,
    bank_specs,
    standardize_features=True,
):
    """
    Reuse existing build_graphs_from_payload(...) repeatedly and attach
    a bank [K, N, N] to each graph.

    Each spec can override:
      - name
      - connectivity_metric
      - connectivity_band
      - filter_method
      - fixed_edges
    """
    if bank_specs is None or len(bank_specs) == 0:
        raise ValueError("bank_specs must contain at least one candidate.")

    candidate_names = []
    candidate_graph_lists = []

    for spec_idx, spec in enumerate(bank_specs):
        name = str(spec.get("name", f"cand_{spec_idx}"))
        cand_metric = spec.get("connectivity_metric", default_connectivity_metric)

        if "connectivity_band" in spec:
            cand_band = spec["connectivity_band"]
        else:
            cand_band = default_connectivity_band

        cand_filter_method = spec.get("filter_method", default_filter_method)
        cand_fixed_edges = spec.get("fixed_edges", default_fixed_edges)

        gs = build_graphs_from_payload(
            payload,
            subject_ids,
            feature_families=feature_families,
            connectivity_metric=cand_metric,
            connectivity_band=cand_band,
            filter_method=cand_filter_method,
            fixed_edges=cand_fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=standardize_features,
        )

        candidate_names.append(name)
        candidate_graph_lists.append(gs)

    base_graphs = candidate_graph_lists[0]

    def _graph_key(g):
        sid = str(getattr(g, "subject_id", ""))
        seg = int(getattr(g, "segment_id", -1))
        start = int(getattr(g, "start_sample", -1))
        return (sid, seg, start)

    # precompute maps once
    candidate_maps = []
    for cand_name, gs in zip(candidate_names, candidate_graph_lists):
        gmap = {}
        for g in gs:
            gmap[_graph_key(g)] = g
        candidate_maps.append(gmap)


    # attach [K, N, N] bank to each base graph
    for g in base_graphs:
        key = _graph_key(g)

        bank_adj = []
        bank_topo = []

        for cand_name, gmap in zip(candidate_names, candidate_maps):
            if key not in gmap:
                raise KeyError(f"Graph key {key} missing in candidate {cand_name!r}.")
            gg = gmap[key]

            if not hasattr(gg, "adj") or gg.adj is None:
                raise ValueError(
                    f"Candidate {cand_name!r} graph for key {key} is missing dense adj. "
                    "Make sure build_graphs_from_payload(..., attach_dense_adj=True)."
                )

            adj = gg.adj
            if torch.is_tensor(adj):
                adj = adj.detach().cpu().float()
            else:
                adj = torch.tensor(adj, dtype=torch.float32)

            topo = (adj != 0).float()

            bank_adj.append(adj)
            bank_topo.append(topo)

        g.adj_bank = torch.stack(bank_adj, dim=0)          # [K, N, N]
        g.topology_bank = torch.stack(bank_topo, dim=0)    # [K, N, N]
        g.topology_names = list(candidate_names)
        g.conn_stack = g.adj_bank
        g.conn_stack_names = list(candidate_names)
    return base_graphs, candidate_names


class SubjectBalancedSegmentKDataset(Dataset):
    """
    Flat segment dataset that resamples k segments per subject per epoch.

    This is the budget-matched segment baseline for MIL:
        same candidate graph pool, same base_k, but segment-level CE loss.
    """
    def __init__(
        self,
        graphs: Sequence[Data],
        k: int,
        seed: int = 42,
        fill_with_replacement: bool = True,
        sort_graphs_by: str = "segment_id",
    ):
        if k is None or int(k) <= 0:
            raise ValueError(f"k must be positive for SubjectBalancedSegmentKDataset, got {k}")
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        self.fill_with_replacement = bool(fill_with_replacement)
        self.subject_to_graphs: Dict[str, List[Data]] = defaultdict(list)
        self.subject_to_label: Dict[str, int] = {}

        for g in graphs:
            sid = str(g.subject_id)
            y = int(g.y.view(-1)[0].item())
            self.subject_to_graphs[sid].append(g)
            if sid in self.subject_to_label and self.subject_to_label[sid] != y:
                raise ValueError(f"Subject {sid} has inconsistent labels.")
            self.subject_to_label[sid] = y

        self.subject_ids = sorted(self.subject_to_graphs.keys())
        self.subject_labels = [self.subject_to_label[sid] for sid in self.subject_ids]
        if len(self.subject_ids) == 0:
            raise ValueError("No subjects in SubjectBalancedSegmentKDataset.")

        for sid in self.subject_ids:
            if sort_graphs_by == "segment_id":
                self.subject_to_graphs[sid] = sorted(
                    self.subject_to_graphs[sid],
                    key=lambda g: (int(getattr(g, "segment_id", 0)), int(getattr(g, "start_sample", 0))),
                )
            elif sort_graphs_by == "start_sample":
                self.subject_to_graphs[sid] = sorted(
                    self.subject_to_graphs[sid],
                    key=lambda g: (int(getattr(g, "start_sample", 0)), int(getattr(g, "segment_id", 0))),
                )
            else:
                raise ValueError(f"Unsupported sort_graphs_by={sort_graphs_by!r}")

        first_graph = self.subject_to_graphs[self.subject_ids[0]][0]
        self.num_node_features = int(first_graph.x.shape[-1])
        self.num_nodes = int(first_graph.x.shape[0])
        self._indices: List[Tuple[str, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        indices: List[Tuple[str, int]] = []
        for sid in self.subject_ids:
            graphs = self.subject_to_graphs[sid]
            n = len(graphs)
            rng = random.Random(self.seed + 1000003 * self.epoch + _stable_int_from_string(sid))
            if n >= self.k:
                chosen = rng.sample(range(n), self.k)
            else:
                chosen = list(range(n))
                if self.fill_with_replacement:
                    chosen += [rng.randrange(n) for _ in range(self.k - n)]
            for j in chosen:
                indices.append((sid, int(j)))
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        sid, j = self._indices[idx]
        return self.subject_to_graphs[sid][j]


class GraphSegmentDataset(Dataset):
    def __init__(self, graphs: Sequence[Data]):
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("GraphSegmentDataset received an empty graph list.")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


class SubjectMILClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",
        num_nodes: Optional[int] = None,

        # shared graph encoder settings
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",

        # existing GNN settings
        gnn_hidden_dim: int = 64,

        # GraphSAGE settings
        sage_layers: int = 2,

        # GCNII settings
        gcn2_layers: int = 8,
        gcn2_alpha: float = 0.1,
        gcn2_theta: float = 0.5,
        gcn2_shared_weights: bool = True,
        gcn2_use_edge_weight: bool = True,

        # H2GCN settings
        h2gcn_layers: int = 2,

        # raw-MLP params
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        cnn_channels: Sequence[int] = (16, 32),
        # MIL settings
        mil_pool_type: str = "gated",   # "mean" or "gated" or "constrained_weighted_mean"
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
        # cnn_channels: Sequence[int] = (16, 32, 64),
        cnn_num_bands: int = 5,

        num_gnn_layers: int = 2,
        readout_type: str = "mean",
        node_pooling_type: str = "none",
        node_pool_ratio: float = 0.8,
        use_edge_weight: bool = True,
        gat_heads: int = 4,
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,

        graph_backbone: str = "gcn",          # "gcn" | "sage" | "gatv2"
        use_batchnorm: bool = True,
        return_graph_attention_weights: bool = False,
        pool_every_layer: bool = True,
        stage_readout_fusion: str = "concat",

        num_candidates: Optional[int] = None,
        bank_fusion_mode: str = "static",
        bank_topology_rule: str = "union",
        bank_vote_threshold: float = 0.5,
        bank_fusion_temperature: float = 1.0,
        bank_hidden_dim: int = 64,

        candidate_fusion_mode: str = "concat",
        candidate_fusion_hidden_dim: int = 64,
        candidate_fusion_dropout: float = 0.0,
        share_linkx_weights: bool = False,
        use_gcn_norm: bool = False,
        gcn_norm_add_self_loops: bool = True,

        # Prototype-aware MIL settings
        use_prototypes: bool = False,
        num_prototypes: int = 0,
        prototype_emb_dim: int = 16,
        prototype_hidden_dim: int = 64,
        prototype_use_soft: bool = True,
        prototype_use_dist: bool = True,

        gcn_normalize_input: bool = False,
        gcn_norm_abs_weights: bool = False,
        gcn_norm_abs_degree: bool = False,
        use_prototype_classifier: bool = False,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()
        self.use_prototypes = bool(use_prototypes)
        self.num_prototypes = int(num_prototypes)
        self.prototype_use_soft = bool(prototype_use_soft)
        self.prototype_use_dist = bool(prototype_use_dist)
        self.gcn_normalize_input = bool(gcn_normalize_input)
        self.gcn_norm_add_self_loops = bool(gcn_norm_add_self_loops)
        self.gcn_norm_abs_weights = bool(gcn_norm_abs_weights)
        self.gcn_norm_abs_degree = bool(gcn_norm_abs_degree)
        self.use_prototype_classifier = bool(use_prototype_classifier)
        self.prototype_classifier = PrototypeClassifier(graph_emb_dim, num_classes)
        
        if self.use_prototypes:
            if self.num_prototypes <= 0:
                raise ValueError("num_prototypes must be > 0 when use_prototypes=True")

            self.prototype_embedding = nn.Embedding(
                self.num_prototypes,
                prototype_emb_dim,
            )

            proto_input_dim = prototype_emb_dim

            if self.prototype_use_soft:
                proto_input_dim += self.num_prototypes

            if self.prototype_use_dist:
                proto_input_dim += self.num_prototypes

            self.prototype_mlp = nn.Sequential(
                nn.Linear(proto_input_dim, prototype_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(prototype_hidden_dim, prototype_emb_dim),
                nn.ReLU(),
            )

            # Keep graph_emb_dim unchanged, so the existing MIL pool and classifier still work.
            self.prototype_fusion = nn.Sequential(
                nn.Linear(graph_emb_dim + prototype_emb_dim, graph_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.prototype_embedding = None
            self.prototype_mlp = None
            self.prototype_fusion = None

        self.num_gnn_layers = int(num_gnn_layers)
        self.readout_type = str(readout_type)
        self.node_pooling_type = str(node_pooling_type)
        self.node_pool_ratio = float(node_pool_ratio)
        self.use_edge_weight = bool(use_edge_weight)
        self.gat_heads = int(gat_heads)
        self.readout_hidden_dim = int(readout_hidden_dim)
        self.readout_dropout = float(readout_dropout)

        self.graph_backbone = str(graph_backbone).lower()
        self.use_batchnorm = bool(use_batchnorm)
        self.return_graph_attention_weights = bool(return_graph_attention_weights)


        self.pool_every_layer = bool(pool_every_layer)
        self.stage_readout_fusion = str(stage_readout_fusion).lower()
        if self.encoder_type == "sage":
            self.graph_encoder = GraphSAGEEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=sage_layers,
                dropout=dropout,
                pool=graph_pool,
                jk_mode="last",
            )

        elif self.encoder_type == "linkx_bank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_bank'")
            if num_candidates is None or int(num_candidates) < 1:
                raise ValueError("num_candidates must be provided and >= 1 for encoder_type='linkx_bank'")

            self.graph_encoder = MultiBranchLinkXEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=int(num_candidates),
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                candidate_fusion_mode=candidate_fusion_mode,
                fusion_hidden_dim=candidate_fusion_hidden_dim,
                fusion_dropout=candidate_fusion_dropout,
                share_linkx_weights=share_linkx_weights,
            )
        elif self.encoder_type == "linkx_fused_bank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_fused_bank'")
            if num_candidates is None or int(num_candidates) < 1:
                raise ValueError("num_candidates must be provided and >= 1 for encoder_type='linkx_fused_bank'")

            self.graph_encoder = FusedBankLinkXEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=int(num_candidates),
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                bank_fusion_mode=bank_fusion_mode,
                topology_rule=bank_topology_rule,
                vote_threshold=bank_vote_threshold,
                fusion_temperature=bank_fusion_temperature,
                fusion_hidden_dim=bank_hidden_dim,
            )
        elif self.encoder_type == "gnn_block":

            from pipeline.gnn import GraphEncoderBlock
            self.graph_encoder = GraphEncoderBlock(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=num_gnn_layers,
                backbone=self.graph_backbone,
                dropout=dropout,
                gat_heads=gat_heads,
                use_edge_weight=use_edge_weight,
                use_batchnorm=use_batchnorm,
                node_pooling_type=node_pooling_type,
                node_pool_ratio=node_pool_ratio,
                readout_type=readout_type,
                readout_hidden_dim=readout_hidden_dim,
                readout_dropout=readout_dropout,
                return_attention_weights=return_graph_attention_weights,
            )

        elif self.encoder_type == "hier_gnn_block":
            from pipeline.gnn import ProgressiveGraphEncoderBlock

            self.graph_encoder = ProgressiveGraphEncoderBlock(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=num_gnn_layers,
                backbone=self.graph_backbone,
                dropout=dropout,
                gat_heads=gat_heads,
                use_edge_weight=use_edge_weight,
                use_batchnorm=use_batchnorm,
                node_pooling_type=node_pooling_type,
                node_pool_ratio=node_pool_ratio,
                readout_type=readout_type,
                readout_hidden_dim=readout_hidden_dim,
                readout_dropout=readout_dropout,
                return_attention_weights=return_graph_attention_weights,
                pool_every_layer=pool_every_layer,
                stage_readout_fusion=stage_readout_fusion,
            )
        elif self.encoder_type == "linkx_cnn":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_cnn'")

            self.graph_encoder = RawNodeAdjCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                cnn_channels=cnn_channels,
                dropout=dropout,
                symmetrize_adj=True,
                zero_diagonal=False,
            )
        elif self.encoder_type in {"linkx_cnn5", "linkx_cnn_bank"}:
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_cnn5'")

            self.graph_encoder = RawNodeMultiBandCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                num_bands=cnn_num_bands,
                symmetrize_adj=True,
                zero_diagonal=False,
            )

        elif self.encoder_type in {"cnn5", "cnn_bank"}:
            self.graph_encoder = MultiBandCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                num_bands=cnn_num_bands,
                symmetrize_adj=True,
                zero_diagonal=False,
                )
        elif self.encoder_type == "gcn2":
            self.graph_encoder = GCNIIEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=gcn2_layers,
                dropout=dropout,
                alpha=gcn2_alpha,
                theta=gcn2_theta,
                shared_weights=gcn2_shared_weights,
                pool=graph_pool,
                use_edge_weight=gcn2_use_edge_weight,
            )

        elif self.encoder_type == "h2gcn":
            self.graph_encoder = H2GCNLikeEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=h2gcn_layers,
                dropout=dropout,
                pool=graph_pool,
            )

        elif self.encoder_type == "gnn":
            self.graph_encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif self.encoder_type == "gat":
            self.graph_encoder = GNNEncoder_GAT(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                num_layers=3,
                dropout=dropout,
                heads=gat_heads,
                edge_dim=1,
                pooling=graph_pool
            )

        elif self.encoder_type == "hybrid":
            self.graph_encoder = HybridGNNEncoder(
                 in_channels=num_node_features, 
                 hidden_channels=gnn_hidden_dim, 
                 emb_dim=graph_emb_dim,
                 gat_layers=num_gnn_layers//2, 
                 cheb_layers=num_gnn_layers//2,
                 dropout=dropout, 
                 heads=gat_heads, 
                 edge_dim=1,
                 pooling=graph_pool)


        elif self.encoder_type == "linkx":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx'")

            self.graph_encoder = RawNodeEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                use_upper_triangle=True,
                symmetrize_adj=True,
            )
        elif self.encoder_type == "mlp_node":
            self.graph_encoder = RawNodeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                proj_dim = branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout)

        elif self.encoder_type in ["gnn_bank"]:
            self.graph_encoder = RawNodeGraphBankGNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                graph_emb_dim=graph_emb_dim,
                use_gcn_norm=False,
                gcn_norm_add_self_loops=False,
                branch_emb_dim=branch_emb_dim,
                node_hidden_dims=node_hidden_dims,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_out_dim=gnn_hidden_dim,
                gnn_layers=num_gnn_layers,
                backbone=graph_backbone,        # "gcn", "sage", or "gatv2"
                readout=graph_pool,
                fusion=candidate_fusion_mode,
                dropout=dropout,
                gat_heads=gat_heads,
                adj_value_mode="abs",    # good default for correlation-like adjacency
                symmetrize_adj=True,
                zero_diagonal=True,
                use_edge_weight=True,

            )
        elif self.encoder_type == "edge_token":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='edge_token'")

            self.graph_encoder = RawNodeEdgeTokenTransformerEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_bands=cnn_num_bands,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                edge_transformer_dim=64,
                edge_heads=4,
                edge_layers=2,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )
        elif self.encoder_type == "node_token":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided for encoder_type='node_token'")

            self.graph_encoder = NodeTokenEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                token_dim=branch_emb_dim,
                num_layers=1,
                num_heads=4,
                ff_dim=max(128, 2 * branch_emb_dim),
                emb_dim=graph_emb_dim,
                dropout=dropout,
                use_cls_token=False,
            )

        elif self.encoder_type == "node_token_bank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided for encoder_type='node_token_bank'")
            if num_candidates is None:
                raise ValueError("num_candidates must be provided for encoder_type='node_token_bank'")

            self.graph_encoder = NodeTokenBankSummaryEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=num_candidates,
                token_dim=branch_emb_dim,
                num_layers=1,
                num_heads=4,
                ff_dim=max(128, 2 * branch_emb_dim),
                emb_dim=graph_emb_dim,
                dropout=dropout,
                conn_summary_topk=4,
                conn_value_mode="raw",
            )

        elif self.encoder_type == "node_token_linkx_bank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided for encoder_type='node_token_linkx_bank'")
            if num_candidates is None:
                raise ValueError("num_candidates must be provided for encoder_type='node_token_linkx_bank'")

            self.graph_encoder = NodeTokenLinkXBankResidualEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=num_candidates,
                token_dim=branch_emb_dim,
                token_layers=1,
                token_heads=4,
                token_ff_dim=max(128, 2 * branch_emb_dim),
                graph_emb_dim=graph_emb_dim,
                branch_emb_dim=branch_emb_dim,
                edge_hidden_dims=tuple(edge_hidden_dims),
                dropout=dropout,
                conn_summary_topk=4,
                conn_value_mode="raw",
                edge_fusion_mode="mean",
                edge_value_mode="raw",
                init_edge_scale=-3.0,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type='{encoder_type}'. "
                f"Choose from ['gnn','hybrid', 'gat','linkx', 'cnn5', 'linkx_cnn5', 'mlp_node', 'sage', 'gcn2', 'h2gcn']"
            )

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )

        elif self.mil_pool_type in ["constrained_weighted_mean", "cwmean"]:
            self.mil_pool = ConstrainedWeightedMeanMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
                dropout=dropout,
                temperature=2.0,
                gamma_max=0.6,
                min_effective_frac=0.35,
                min_entropy=0.75,
                lambda_entropy=0.01,
                lambda_effective=0.01,
                lambda_max_weight=0.01,
                segment_dropout=0.2,
            )

        else:
            raise ValueError(f"Unknown mil_pool_type='{mil_pool_type}'")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )
    def _get_prototype_tensors(self, batch_dict: Dict):
        """
        Recover prototype tensors from PyG Batch.

        Expected attached graph fields:
            g.proto_id       -> batched as [G]
            g.proto_soft     -> batched as [G, K]
            g.proto_dist_log -> batched as [G, K]
        """
        pyg_batch = batch_dict["pyg_batch"]

        if not hasattr(pyg_batch, "proto_id"):
            raise KeyError(
                "pyg_batch is missing proto_id. "
                "Run attach_segment_prototypes(...) before creating the dataset/dataloader."
            )

        proto_id = pyg_batch.proto_id.view(-1).long()

        proto_soft = getattr(pyg_batch, "proto_soft", None)
        proto_dist_log = getattr(pyg_batch, "proto_dist_log", None)

        if proto_soft is not None:
            proto_soft = proto_soft.float()
            if proto_soft.dim() == 1:
                proto_soft = proto_soft.view(-1, self.num_prototypes)

        if proto_dist_log is not None:
            proto_dist_log = proto_dist_log.float()
            if proto_dist_log.dim() == 1:
                proto_dist_log = proto_dist_log.view(-1, self.num_prototypes)

        return proto_id, proto_soft, proto_dist_log


    def _fuse_prototypes(self, graph_emb: torch.Tensor, batch_dict: Dict) -> torch.Tensor:
        """
        Fuse graph embedding with prototype context.

        graph_emb: [G, graph_emb_dim]
        """
        if not self.use_prototypes:
            return graph_emb

        proto_id, proto_soft, proto_dist_log = self._get_prototype_tensors(batch_dict)

        proto_id = proto_id.to(graph_emb.device)

        parts = [
            self.prototype_embedding(proto_id)
        ]

        if self.prototype_use_soft:
            if proto_soft is None:
                raise KeyError("prototype_use_soft=True, but proto_soft is missing.")
            parts.append(proto_soft.to(graph_emb.device))

        if self.prototype_use_dist:
            if proto_dist_log is None:
                raise KeyError("prototype_use_dist=True, but proto_dist_log is missing.")
            parts.append(proto_dist_log.to(graph_emb.device))

        proto_input = torch.cat(parts, dim=-1)
        proto_context = self.prototype_mlp(proto_input)

        graph_emb = self.prototype_fusion(
            torch.cat([graph_emb, proto_context], dim=-1)
        )

        return graph_emb
    # def _encode_graphs(self, batch_dict):
    #     pyg_batch = batch_dict["pyg_batch"]

    #     if self.encoder_type == "linkx_cnn5":
    #         return self.graph_encoder(pyg_batch, batch_dict["conn_stack"])
    #     elif self.encoder_type == "linkx_cnn":
    #         return self.graph_encoder(pyg_batch, batch_dict["full_adj"])
    #     else:
    #         return self.graph_encoder(pyg_batch)
    def _run_graph_encoder(self, batch_dict):
        pyg_batch = batch_dict["pyg_batch"]

        if self.encoder_type == "linkx_cnn":
            out = self.graph_encoder(pyg_batch, batch_dict["full_adj"])
            return out, None

        elif self.encoder_type in ["linkx_cnn5"]:
            out = self.graph_encoder(pyg_batch, batch_dict["conn_stack"])
            return out, None

        elif self.encoder_type == "linkx_fused_bank":
            out = self.graph_encoder(
                pyg_batch,
                adj_bank=batch_dict.get("adj_bank", None),
                topology_bank=batch_dict.get("topology_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["linkx_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                adj_bank=batch_dict.get("adj_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["cnn5", "cnn_bank", "linkx_cnn_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                conn_stack=batch_dict.get("conn_stack", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["gnn_bank", "gatv2_bank", "gcn_bank", "sage_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                adj_bank=batch_dict.get("adj_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None
        elif self.encoder_type == "edge_token":
            if "conn_stack" in batch_dict:
                out = self.graph_encoder(
                    pyg_batch,
                    batch_dict["conn_stack"],
                )
                return out, None
            elif "full_adj" in batch_dict:
                out = self.graph_encoder(
                    pyg_batch,
                    batch_dict["full_adj"],
                )
                return out, None
            else:
                raise KeyError(
                    "edge_token encoder needs batch_dict['conn_stack'] or batch_dict['full_adj']"
                )
        elif self.encoder_type in {"node_token_bank", "node_token_linkx_bank"}:
            adj_bank = batch_dict.get("adj_bank", None)
            if adj_bank is None:
                adj_bank = batch_dict.get("conn_stack", None)
            if adj_bank is None:
                adj_bank = batch_dict.get("full_adj", None)
            if adj_bank is None:
                raise KeyError(
                    f"{self.encoder_type} needs batch_dict['adj_bank'], "
                    "batch_dict['conn_stack'], or batch_dict['full_adj']."
                )

            out = self.graph_encoder(pyg_batch, adj_bank)

            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type == "node_token":
            out = self.graph_encoder(pyg_batch, None)

            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None


        out = self.graph_encoder(pyg_batch)

        if isinstance(out, tuple) and len(out) == 2:
            graph_emb, graph_attn = out
            return graph_emb, graph_attn

        return out, None


    def forward(self, batch_dict: Dict):
        # graph_emb = self._encode_graphs(batch_dict)
        if self.gcn_normalize_input:
            batch_dict = apply_gcn_norm_to_batch_dict(
                batch_dict,
                add_self_loops=self.gcn_norm_add_self_loops,
                use_abs_edge_weight=self.gcn_norm_abs_weights,
                use_abs_degree=self.gcn_norm_abs_degree,
            )
        graph_emb, graph_attn = self._run_graph_encoder(batch_dict)
        
        # NEW: prototype-aware segment embedding
        graph_emb = self._fuse_prototypes(graph_emb, batch_dict)

        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        if self.use_prototype_classifier:
            logits = self.prototype_classifier(bag_emb)
        else:
            logits = self.classifier(bag_emb)

        # return {
        #     "logits": logits,
        #     "bag_emb": bag_emb,
        #     "graph_emb": graph_emb,
        #     "attn_list": attn_list,
        #     "graph_attention_weights": graph_attn,
        # }


        out = {
            "graph_emb": graph_emb,
            "bag_emb": bag_emb,
            "logits": logits,
            "attn_list": attn_list,
        }

        if self.encoder_type == "edge_token":
            out["edge_attn"] = edge_attn

        if graph_attn is not None:
            out["graph_attn"] = graph_attn

            # convenience aliases for common encoder outputs
            if isinstance(graph_attn, dict):
                if "stage_readout_attention" in graph_attn:
                    out["stage_readout_attention"] = graph_attn["stage_readout_attention"]
                if "graph_attention_weights" in graph_attn:
                    out["graph_attention_weights"] = graph_attn["graph_attention_weights"]
                if "fusion_weights" in graph_attn:
                    out["fusion_weights"] = graph_attn["fusion_weights"]
                if "fused_adjacency" in graph_attn:
                    out["fused_adjacency"] = graph_attn["fused_adjacency"]
                if "fused_topology" in graph_attn:
                    out["fused_topology"] = graph_attn["fused_topology"]
                if "candidate_fusion_weights" in graph_attn:
                    out["candidate_fusion_weights"] = graph_attn["candidate_fusion_weights"]
                if "candidate_embeddings" in graph_attn:
                    out["candidate_embeddings"] = graph_attn["candidate_embeddings"]
        
        if self.use_prototypes:
            pyg_batch = batch_dict["pyg_batch"]
            out["proto_id"] = pyg_batch.proto_id.detach().cpu()
            if hasattr(pyg_batch, "proto_soft"):
                out["proto_soft"] = pyg_batch.proto_soft.detach().cpu()

        if hasattr(self.mil_pool, "last_reg_loss") and self.mil_pool.last_reg_loss is not None:
            out["reg_loss"] = self.mil_pool.last_reg_loss

        if hasattr(self.mil_pool, "last_diagnostics"):
            out["pool_diagnostics"] = self.mil_pool.last_diagnostics



        return out

    def forward_with_embeddings(self, batch_dict: Dict):
        """
        Same as forward(...), but explicit name for analysis code.
        """
        return self.forward(batch_dict)

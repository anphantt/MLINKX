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

    if method == "full":
        selected = full_edges

    elif method == "fixed":
        selected = fixed_set

    elif method == "topk":
        selected = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)

    elif method == "combined":
        topk_set = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)
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
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    undirected=True,
    filter_method="fixed",
    topk=4,
    top_percent=None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    standardize_features=True,
):

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

            adj_used = adj_full_t.clone()
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

            x_np = x.detach().cpu().numpy().astype(np.float32)

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


class GraphSegmentDataset(Dataset):
    def __init__(self, graphs: Sequence[Data]):
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("GraphSegmentDataset received an empty graph list.")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

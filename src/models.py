from __future__ import annotations
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse, to_dense_batch, to_dense_adj
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GraphNorm, global_mean_pool, global_add_pool, global_max_pool, BatchNorm
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


BANK_ENCODERS = {
    "mlinkx",
    "cnn_bank",
    "gnn_bank",
}

MULTIBAND_ENCODERS = {}

def make_mlp_local(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_dims:
        h = int(h)
        layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(float(dropout))])
        prev = h
    return nn.Sequential(*layers), prev



class Gat_block(nn.Module):
    def __init__(self, 
                 hidden_channels=64,
                 concat=True,
                 edge_dim=1,
                 heads=4, 
                 ):
        super(Gat_block, self).__init__()
        self.conv = GATv2Conv(hidden_channels, int(hidden_channels/heads), heads=heads, concat=concat, edge_dim=edge_dim)
        self.bn = BatchNorm(hidden_channels)
    def forward(self,x,edge_index,batch,edge_attr=None):
        #print(x.shape)
        xs = F.relu(self.bn(self.conv(x, edge_index, edge_attr=edge_attr)))
        return xs

class GNNEncoder_GAT(nn.Module):
    """
    Segment graph -> graph embedding
    Converted from EEGGNN_GAT while keeping the architecture as similar as possible.

    Original classifier:
        pooled graph -> fc1 -> fc2 -> logits

    Encoder version:
        pooled graph -> fc1 -> fc2 -> graph embedding
    """
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 emb_dim=128,
                 num_layers=3,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(GNNEncoder_GAT, self).__init__()
        
        self.dropout = dropout
        self.pooling = pooling.lower()
        self.num_layers = num_layers

        # ----- Input projection -----
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ----- Stack of GAT blocks -----
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(Gat_block(
                hidden_channels=hidden_channels,
                concat=True,
                edge_dim=edge_dim,
                heads=heads
            ))

        # ----- Projection head -----
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, emb_dim)

    def forward(self, data_batch: Batch):
        x = data_batch.x
        edge_index = data_batch.edge_index
        batch = data_batch.batch

        # support either edge_attr or edge_weight stored in the batch
        edge_attr = getattr(data_batch, "edge_attr", None)
        if edge_attr is None:
            edge_attr = getattr(data_batch, "edge_weight", None)

        # Ensure edge_attr shape for GATv2Conv(edge_dim=1)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # ----- Input projection -----
        x = self.input_mlp(x)

        # ----- GAT layers -----
        for conv in self.gat_layers:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- Global pooling -----
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # ----- Projection head -----
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self.fc2(x)

        return graph_emb


class RawNodeMLPEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        proj_dim: int = 128,   # use 64 for strict ablation, 128 for capacity-matched
        emb_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        node_input_dim = num_nodes * num_node_features
        self.num_nodes = num_nodes

        self.node_mlp, node_last_dim = make_mlp_local(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, proj_dim)

        self.fusion = nn.Sequential(
            nn.Linear(proj_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, pyg_batch):
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")

        node_x = dense_x.reshape(dense_x.size(0), -1)
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)
        graph_emb = self.fusion(node_emb)
        return graph_emb

class RawNodeEdgeMLPEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_upper_triangle: bool = True,
        symmetrize_adj: bool = True,
        edge_mode: str = "topology_weighted",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.use_upper_triangle = use_upper_triangle
        self.symmetrize_adj = symmetrize_adj
        self.edge_mode = edge_mode.lower()


        if self.edge_mode not in ["topology_weighted", "topology_binary", "full_adj"]:
            raise ValueError(f"Unsupported edge_mode={edge_mode}")

        node_input_dim = num_nodes * num_node_features
        if use_upper_triangle:
            edge_input_dim = num_nodes * (num_nodes - 1) // 2
        else:
            edge_input_dim = num_nodes * num_nodes

        # Node branch
        self.node_mlp, node_last_dim = make_mlp_local(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # Edge branch
        self.edge_mlp, edge_last_dim = make_mlp_local(edge_input_dim, edge_hidden_dims, dropout)
        self.edge_proj = nn.Linear(edge_last_dim, branch_emb_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _get_topology_weighted_adj(self, pyg_batch):
        edge_attr = getattr(pyg_batch, "edge_attr", None)
        if edge_attr is None:
            edge_attr = getattr(pyg_batch, "edge_weight", None)

        if edge_attr is not None:
            if edge_attr.dim() > 1:
                if edge_attr.size(-1) == 1:
                    edge_attr = edge_attr.squeeze(-1)
                else:
                    edge_attr = edge_attr[:, 0]

        adj = to_dense_adj(
            pyg_batch.edge_index,
            batch=pyg_batch.batch,
            edge_attr=edge_attr,
            max_num_nodes=self.num_nodes,
        )
        return adj

    def _get_topology_binary_adj(self, pyg_batch):
        num_edges = pyg_batch.edge_index.size(1)
        binary_edge_attr = torch.ones(
            num_edges,
            device=pyg_batch.edge_index.device,
            dtype=pyg_batch.x.dtype,
        )

        adj = to_dense_adj(
            pyg_batch.edge_index,
            batch=pyg_batch.batch,
            edge_attr=binary_edge_attr,
            max_num_nodes=self.num_nodes,
        )
        return adj

    def forward(self, pyg_batch):
        """
        pyg_batch.x         : [total_nodes, F]
        pyg_batch.batch     : [total_nodes]
        pyg_batch.edge_index: [2, total_edges]
        pyg_batch.edge_attr : [total_edges, 1] or [total_edges] or absent
        """
        # ----- node branch -----
        dense_x, mask = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [num_graphs, N, F]

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}"
            )

        node_x = dense_x.reshape(dense_x.size(0), -1)  # [num_graphs, N*F]
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)              # [num_graphs, branch_emb_dim]

        # ----- edge branch -----
        if self.edge_mode == "topology_weighted":
            adj = self._get_topology_weighted_adj(pyg_batch)

        elif self.edge_mode == "topology_binary":
            adj = self._get_topology_binary_adj(pyg_batch)

        else:
            raise ValueError(f"Unsupported edge_mode={self.edge_mode}")

        if self.symmetrize_adj:
            adj = 0.5 * (adj + adj.transpose(1, 2))

        if self.use_upper_triangle:
            iu = torch.triu_indices(
                self.num_nodes, self.num_nodes, offset=1, device=adj.device
            )
            edge_x = adj[:, iu[0], iu[1]]             # [num_graphs, N*(N-1)/2]
        else:
            edge_x = adj.reshape(adj.size(0), -1)     # [num_graphs, N*N]

        edge_h = self.edge_mlp(edge_x)
        edge_emb = self.edge_proj(edge_h)             # [num_graphs, branch_emb_dim]

        # ----- fuse -----
        fused = torch.cat([node_emb, edge_emb], dim=1)
        graph_emb = self.fusion(fused)                # [num_graphs, emb_dim]

        return graph_emb

class SubjectMILClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",
        num_nodes: Optional[int] = None,
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",
        gnn_hidden_dim: int = 64,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        cnn_channels: Sequence[int] = (16, 32),
        mil_pool_type: str = "gated",   # "mean" or "gated" or "constrained_weighted_mean"
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
        num_gnn_layers: int = 3,
        gat_heads: int = 4,
        
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()
        self.num_gnn_layers = int(num_gnn_layers)
        self.gat_heads = int(gat_heads)

        if self.encoder_type == "gat":
            self.graph_encoder = GNNEncoder_GAT(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                num_layers=num_gnn_layers,
                # num_layers=3,
                dropout=dropout,
                heads=gat_heads,
                edge_dim=1,
                pooling=graph_pool
            )

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

        else:
            raise ValueError(
                f"Unknown encoder_type='{encoder_type}'. "
                f"Choose from ['gat','linkx', 'mlp_node']"
            )

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )

        else:
            raise ValueError(f"Unknown mil_pool_type='{mil_pool_type}'")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def _run_graph_encoder(self, batch_dict):
        pyg_batch = batch_dict["pyg_batch"]

        if self.encoder_type in ["mlinkx"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                adj_bank=batch_dict.get("adj_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["cnn_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                conn_stack=batch_dict.get("conn_stack", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["gnn_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                adj_bank=batch_dict.get("adj_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        out = self.graph_encoder(pyg_batch)

        if isinstance(out, tuple) and len(out) == 2:
            graph_emb, graph_attn = out
            return graph_emb, graph_attn

        return out, None


    def forward(self, batch_dict: Dict):
        graph_emb, graph_attn = self._run_graph_encoder(batch_dict)
        

        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        logits = self.classifier(bag_emb)

        out = {
            "graph_emb": graph_emb,
            "bag_emb": bag_emb,
            "logits": logits,
            "attn_list": attn_list,
        }

        return out

class SegmentGraphClassifierFromMIL(nn.Module):
    def __init__(self, **mil_kwargs):
        super().__init__()
        mil_kwargs = dict(mil_kwargs)
        mil_kwargs.setdefault("mil_pool_type", "mean")
        base = make_subject_model(**mil_kwargs)
        self.encoder_type = str(getattr(base, "encoder_type", mil_kwargs.get("encoder_type", "gnn"))).lower()
        self.graph_encoder = base.graph_encoder
        self.classifier = base.classifier

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.encoder_type in BANK_ENCODERS:
            adj_bank = batch_dict.get("adj_bank", None)
            conn_stack = batch_dict.get("conn_stack", None)
            if adj_bank is None and conn_stack is not None:
                adj_bank = conn_stack
            if conn_stack is None and adj_bank is not None:
                conn_stack = adj_bank
            if adj_bank is None:
                raise KeyError("Bank encoder needs batch_dict['adj_bank'] or batch_dict['conn_stack'].")
            if self.encoder_type in {"cnn_bank"}:
                graph_emb = self.graph_encoder(batch_dict["pyg_batch"], conn_stack)
            else:
                graph_emb = self.graph_encoder(batch_dict["pyg_batch"], adj_bank)
        else:
            graph_emb = self.graph_encoder(batch_dict["pyg_batch"])

        # Some newer encoders may return (embedding, aux).
        if isinstance(graph_emb, tuple):
            graph_emb, aux = graph_emb
        else:
            aux = {}
        logits = self.classifier(graph_emb)
        return {"logits": logits, "graph_emb": graph_emb, **{k: v for k, v in aux.items() if v is not None}}


def _filter_kwargs_for_class(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_subject_model(**model_kwargs) -> nn.Module:
    encoder_l = str(model_kwargs.get("encoder_type", "")).lower()
    if encoder_l in BANK_ENCODERS:
        return BankAwareSubjectMILClassifier(**model_kwargs)
    return SubjectMILClassifier(**_filter_kwargs_for_class(SubjectMILClassifier, model_kwargs))


def _prepare_dense_stack(
    stack: torch.Tensor,
    *,
    num_nodes: int,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    value_mode: str = "raw",
) -> torch.Tensor:
    """Prepare [B,K,N,N] or [B,N,N] dense adjacency tensors."""
    if not torch.is_tensor(stack):
        stack = torch.as_tensor(stack, dtype=torch.float32)
    x = stack.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected dense stack [B,K,N,N] or [B,N,N], got {tuple(x.shape)}")
    if x.shape[-2:] != (int(num_nodes), int(num_nodes)):
        raise ValueError(f"Expected N={num_nodes}, got dense stack shape {tuple(x.shape)}")
    if symmetrize:
        x = 0.5 * (x + x.transpose(-1, -2))
    if zero_diagonal:
        eye = torch.eye(int(num_nodes), device=x.device, dtype=x.dtype).view(1, 1, int(num_nodes), int(num_nodes))
        x = x * (1.0 - eye)
    value_mode = str(value_mode).lower()
    if value_mode == "abs":
        x = x.abs()
    elif value_mode == "positive":
        x = torch.clamp(x, min=0.0)
    elif value_mode == "binary":
        x = (x.abs() > 1e-8).float()
    elif value_mode in {"raw", "none"}:
        pass
    else:
        raise ValueError(f"Unknown value_mode={value_mode!r}")
    return x


class RawNodeBankEdgeMLPEncoder(nn.Module):
    """Node MLP + shared edge MLP over an adjacency bank [B,K,N,N]."""
    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_candidates: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        temp: float = 2.0,
        candidate_fusion_mode: str = "concat",
        candidate_fusion_hidden_dim: Optional[int] = None,
        candidate_fusion_dropout: float = 0.0,
        edge_value_mode: str = "raw",
        layernorm: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.layernorm = bool(layernorm)
        self.num_node_features = int(num_node_features)
        self.num_candidates = int(num_candidates)
        self.edge_value_mode = str(edge_value_mode).lower()
        node_input_dim = self.num_nodes * self.num_node_features
        edge_input_dim = self.num_nodes * (self.num_nodes - 1) // 2
        self.node_mlp, node_last = make_mlp_local(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last, int(branch_emb_dim))
        self.edge_mlp, edge_last = make_mlp_local(edge_input_dim, edge_hidden_dims, dropout)
        self.edge_proj = nn.Linear(edge_last, int(branch_emb_dim))
        self.view_norm = nn.LayerNorm(branch_emb_dim)
        self.candidate_fusion_mode = str(candidate_fusion_mode).lower()

        self.candidate_fusion = BankCandidateFusion(
            num_candidates=self.num_candidates,
            emb_dim=int(branch_emb_dim),
            mode=candidate_fusion_mode if self.candidate_fusion_mode not in {"direct_concat", "node_residual_from_concat", "gamma_dimvec"} else "mean",
            hidden_dim=candidate_fusion_hidden_dim,
            dropout=candidate_fusion_dropout,
            temperature=temp,

        )

        if self.candidate_fusion_mode == "direct_concat":
            fusion_in_dim = (1 + self.num_candidates) * int(branch_emb_dim)
        else:
            fusion_in_dim = 2 * int(branch_emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, int(graph_emb_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        if self.candidate_fusion_mode == "node_residual_from_concat":
            self.node_to_out = nn.Linear(int(branch_emb_dim), int(graph_emb_dim))

            self.bank_to_delta = nn.Sequential(
                nn.Linear(self.num_candidates * int(branch_emb_dim), int(graph_emb_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(graph_emb_dim), int(graph_emb_dim)),
            )

            # Starts small: sigmoid(-4) ≈ 0.018
            self.bank_residual_logit = nn.Parameter(torch.tensor(-4.0))
            self.bank_residual_max = 0.5
        elif self.candidate_fusion_mode == "gamma_dimvec":
            self.node_to_out = nn.Linear(int(branch_emb_dim), int(graph_emb_dim))

            self.bank_to_delta = nn.Sequential(
                nn.Linear(self.num_candidates * int(branch_emb_dim), int(graph_emb_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(graph_emb_dim), int(graph_emb_dim)),
            )
            self.bank_residual_logit = nn.Parameter(
                torch.full((int(graph_emb_dim),), -4.0)
            )
            self.bank_residual_max = 0.5
        else:
            self.node_to_out = None
            self.bank_to_delta = None
            self.bank_residual_logit = None
            self.bank_residual_max = None

    def forward(self, pyg_batch: Batch, adj_bank: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        dense_x, _ = to_dense_batch(pyg_batch.x, pyg_batch.batch, max_num_nodes=self.num_nodes)
        B = dense_x.shape[0]
        node_emb = self.node_proj(self.node_mlp(dense_x.reshape(B, -1)))
        adj_bank = _prepare_dense_stack(adj_bank, num_nodes=self.num_nodes, value_mode=self.edge_value_mode)
        K = adj_bank.shape[1]
        if K != self.num_candidates:
            raise ValueError(f"Expected num_candidates={self.num_candidates}, got K={K}")
        iu = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=adj_bank.device)
        edge_x = adj_bank[:, :, iu[0], iu[1]]  # [B,K,E]
        edge_h = self.edge_mlp(edge_x.reshape(B * K, -1))
        edge_emb = self.edge_proj(edge_h).reshape(B, K, -1)

        if self.candidate_fusion_mode in {"node_residual_from_concat", "gamma_dimvec"}:
            edge_flat = edge_emb.reshape(B, K * edge_emb.shape[-1])  # [B, K*D]

            node_out = self.node_to_out(node_emb)                    # [B, graph_emb_dim]
            bank_delta = self.bank_to_delta(edge_flat)               # [B, graph_emb_dim]

            gamma = self.bank_residual_max * torch.sigmoid(self.bank_residual_logit)
            if self.candidate_fusion_mode == "gamma_dimvec":
                graph_emb = node_out + gamma.unsqueeze(0) * bank_delta
            else:
                graph_emb = node_out + gamma * bank_delta

            aux = {
                "view_attention": None,
                "node_embedding": node_emb.detach(),
                "candidate_embeddings": edge_emb.detach(),
                "bank_delta": bank_delta.detach(),
                "bank_residual_gamma": gamma.detach(),
            }

            return graph_emb, aux
        # if self.layernorm:
        #     edge_emb = self.view_norm(edge_emb)
        if self.candidate_fusion_mode == "direct_concat":
            edge_flat = edge_emb.reshape(B, K * edge_emb.shape[-1])
            graph_emb = self.fusion(torch.cat([node_emb, edge_flat], dim=-1))
            alpha = None
            bank_emb = None
        else:
            bank_emb, alpha = self.candidate_fusion(edge_emb, node_emb=node_emb)
            graph_emb = self.fusion(torch.cat([node_emb, bank_emb], dim=-1))
        # aux = {"view_attention": alpha.detach() if alpha is not None else None}

        aux = {
            "view_attention": alpha.detach() if alpha is not None else None,
            "node_embedding": node_emb.detach(),
            "candidate_embeddings": edge_emb.detach(),
        }

        if bank_emb is not None:
            aux["bank_embedding"] = bank_emb.detach()

        return graph_emb, aux


class BankCNNEncoder(nn.Module):
    """CNN over a candidate adjacency stack [B,K,N,N], optionally fused with node MLP."""
    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_candidates: int,
        branch_emb_dim: int = 64,
        graph_emb_dim: int = 128,
        node_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        include_node_branch: bool = False,
        adj_value_mode: str = "raw",
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_candidates = int(num_candidates)
        self.include_node_branch = bool(include_node_branch)
        self.adj_value_mode = str(adj_value_mode).lower()
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_candidates, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cnn_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(branch_emb_dim)),
            nn.ReLU(),
        )
        if self.include_node_branch:
            self.node_mlp, node_last = make_mlp_local(self.num_nodes * self.num_node_features, node_hidden_dims, dropout)
            self.node_proj = nn.Linear(node_last, int(branch_emb_dim))
            fusion_in = 2 * int(branch_emb_dim)
        else:
            fusion_in = int(branch_emb_dim)
        self.fusion = nn.Sequential(nn.Linear(fusion_in, int(graph_emb_dim)), nn.ReLU(), nn.Dropout(float(dropout)))

    def forward(self, pyg_batch: Batch, conn_stack: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = _prepare_dense_stack(conn_stack, num_nodes=self.num_nodes, value_mode=self.adj_value_mode)
        if x.shape[1] != self.num_candidates:
            raise ValueError(f"Expected num_candidates={self.num_candidates}, got K={x.shape[1]}")
        cnn_emb = self.cnn_proj(self.conv(x))
        if self.include_node_branch:
            dense_x, _ = to_dense_batch(pyg_batch.x, pyg_batch.batch, max_num_nodes=self.num_nodes)
            node_emb = self.node_proj(self.node_mlp(dense_x.reshape(dense_x.shape[0], -1)))
            graph_emb = self.fusion(torch.cat([node_emb, cnn_emb], dim=-1))
        else:
            graph_emb = self.fusion(cnn_emb)
        return graph_emb, {}


class SharedBankGNN(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        backbone: str = "gatv2",
        dropout: float = 0.3,
        use_edge_weight: bool = True,
    ):
        super().__init__()
        self.backbone = str(backbone).lower()
        self.dropout = float(dropout)
        self.use_edge_weight = bool(use_edge_weight)
        dims = [int(in_dim)] + [int(hidden_dim)] * (int(num_layers) - 1) + [int(out_dim)]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(int(num_layers)):
            if self.backbone == "gcn":
                conv = GCNConv(dims[i], dims[i + 1])
            elif self.backbone == "sage":
                conv = SAGEConv(dims[i], dims[i + 1])
            elif self.backbone == "gatv2":
                conv = GATv2Conv(dims[i], dims[i + 1], heads=4, concat=False, edge_dim=1, dropout=dropout)
            else:
                raise ValueError("graph_backbone must be one of: gatv2, gcn, sage")
            self.convs.append(conv)
            self.norms.append(GraphNorm(dims[i + 1]))
        self.output_dim = dims[-1]

    def forward(self, batch: Batch) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = getattr(batch, "edge_weight", None)
        edge_attr = getattr(batch, "edge_attr", None)
        for conv, norm in zip(self.convs, self.norms):
            if self.backbone == "gcn":
                x = conv(x, edge_index, edge_weight=edge_weight if self.use_edge_weight else None)
            elif self.backbone == "gatv2":
                x = conv(x, edge_index, edge_attr=edge_attr if self.use_edge_weight else None)
            else:
                x = conv(x, edge_index)
            x = norm(x, batch.batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch.batch)


class RawNodeGraphBankGNNEncoderLocal(nn.Module):
    """Node MLP + shared GNN over each adjacency candidate, then candidate fusion."""
    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_candidates: int,
        graph_emb_dim: int = 128,
        branch_emb_dim: int = 64,
        node_hidden_dims: Sequence[int] = (256, 128),
        gnn_hidden_dim: int = 64,
        gnn_out_dim: Optional[int] = None,
        gnn_layers: int = 2,
        graph_backbone: str = "gatv2",
        candidate_fusion_mode: str = "gated",
        candidate_fusion_hidden_dim: Optional[int] = None,
        candidate_fusion_dropout: float = 0.0,
        dropout: float = 0.3,
        adj_value_mode: str = "abs",
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_candidates = int(num_candidates)
        self.adj_value_mode = str(adj_value_mode).lower()
        self.node_mlp, node_last = make_mlp_local(self.num_nodes * self.num_node_features, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last, int(branch_emb_dim))
        gnn_out_dim = int(gnn_out_dim or branch_emb_dim)
        self.shared_gnn = SharedBankGNN(
            in_dim=self.num_node_features,
            hidden_dim=int(gnn_hidden_dim),
            out_dim=gnn_out_dim,
            num_layers=int(gnn_layers),
            backbone=graph_backbone,
            dropout=float(dropout),
            use_edge_weight=True,
        )
        self.graph_proj = nn.Linear(self.shared_gnn.output_dim, int(branch_emb_dim))
        self.candidate_fusion = BankCandidateFusion(
            num_candidates=self.num_candidates,
            emb_dim=int(branch_emb_dim),
            mode=candidate_fusion_mode,
            hidden_dim=candidate_fusion_hidden_dim,
            dropout=candidate_fusion_dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * int(branch_emb_dim), int(graph_emb_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def _view_to_batch(self, dense_x: torch.Tensor, dense_adj: torch.Tensor, node_mask: torch.Tensor) -> Batch:
        data_list: List[Data] = []
        for b in range(dense_x.shape[0]):
            valid = node_mask[b].bool()
            xb = dense_x[b, valid]
            ab = dense_adj[b][valid][:, valid]
            ab = torch.nan_to_num(ab, nan=0.0, posinf=0.0, neginf=0.0)
            ab = 0.5 * (ab + ab.T)
            ab.fill_diagonal_(0.0)
            edge_index, edge_weight = dense_to_sparse(ab)
            g = Data(x=xb, edge_index=edge_index.long(), edge_weight=edge_weight.float(), num_nodes=xb.shape[0])
            g.edge_attr = edge_weight.view(-1, 1).float()
            data_list.append(g)
        return Batch.from_data_list(data_list)

    def forward(self, pyg_batch: Batch, adj_bank: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        dense_x, node_mask = to_dense_batch(pyg_batch.x, pyg_batch.batch, max_num_nodes=self.num_nodes)
        B = dense_x.shape[0]
        node_emb = self.node_proj(self.node_mlp(dense_x.reshape(B, -1)))
        adj_bank = _prepare_dense_stack(adj_bank, num_nodes=self.num_nodes, value_mode=self.adj_value_mode)
        if adj_bank.shape[1] != self.num_candidates:
            raise ValueError(f"Expected num_candidates={self.num_candidates}, got K={adj_bank.shape[1]}")
        view_embs = []
        for k in range(self.num_candidates):
            view_batch = self._view_to_batch(dense_x, adj_bank[:, k], node_mask).to(dense_x.device)
            z = self.graph_proj(self.shared_gnn(view_batch))
            view_embs.append(z)
        view_embs_t = torch.stack(view_embs, dim=1)
        bank_emb, alpha = self.candidate_fusion(view_embs_t)
        graph_emb = self.fusion(torch.cat([node_emb, bank_emb], dim=-1))
        return graph_emb, {"view_attention": alpha.detach() if alpha is not None else None}



class MeanMILPool(nn.Module):
    def forward(self, graph_emb: torch.Tensor, bag_sizes: torch.Tensor):
        bag_embs = []
        start = 0
        dummy_attn = []
        for size in bag_sizes.tolist():
            end = start + size
            h = graph_emb[start:end]
            z = h.mean(dim=0)
            bag_embs.append(z)
            dummy_attn.append(torch.ones(size, device=h.device) / size)
            start = end
        bag_embs = torch.stack(bag_embs, dim=0)
        return bag_embs, dummy_attn        


class MeanMILPoolLocal(nn.Module):
    def forward(self, graph_emb: torch.Tensor, bag_sizes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        chunks = torch.split(graph_emb, bag_sizes.detach().cpu().tolist(), dim=0)
        bag_embs = []
        attn_list = []
        for c in chunks:
            bag_embs.append(c.mean(dim=0))
            attn_list.append(torch.ones(c.shape[0], device=c.device, dtype=c.dtype) / max(c.shape[0], 1))
        return torch.stack(bag_embs, dim=0), attn_list


class GatedAttentionMILLocal(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.v = nn.Linear(int(in_dim), int(attn_dim))
        self.u = nn.Linear(int(in_dim), int(attn_dim))
        self.w = nn.Linear(int(attn_dim), 1)

    def forward(self, graph_emb: torch.Tensor, bag_sizes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        chunks = torch.split(graph_emb, bag_sizes.detach().cpu().tolist(), dim=0)
        bag_embs = []
        attn_list = []
        for c in chunks:
            a = self.w(torch.tanh(self.v(c)) * torch.sigmoid(self.u(c))).squeeze(-1)
            a = torch.softmax(a, dim=0)
            bag_embs.append(torch.sum(a.unsqueeze(-1) * c, dim=0))
            attn_list.append(a)
        return torch.stack(bag_embs, dim=0), attn_list

class BankCandidateFusion(nn.Module):
    def __init__(
        self,
        num_candidates: int,
        emb_dim: int,
        mode: str = "concat",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        temperature: float = 2.0,
        residual_max: float = 0.5,
    ):
        super().__init__()

        self.num_candidates = int(num_candidates)
        self.emb_dim = int(emb_dim)
        self.mode = str(mode).lower()
        self.temperature = float(temperature)
        self.residual_max = float(residual_max)

        hidden_dim = int(hidden_dim or emb_dim)

        if self.mode == "concat":
            self.proj = nn.Sequential(
                nn.Linear(self.num_candidates * self.emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, self.emb_dim),
            )

        elif self.mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.emb_dim, hidden_dim),
                nn.Tanh(),
                #nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, 1),
            )

        elif self.mode == "context_gated":
            self.score_mlp = nn.Sequential(
                nn.Linear(self.num_candidates * self.emb_dim, hidden_dim),
                nn.ReLU(),
                #nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, self.num_candidates),
            )

        elif self.mode == "dim_gated":
            self.gate = nn.Sequential(
                nn.Linear(self.emb_dim, hidden_dim),
                nn.Tanh(),
                #nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, self.emb_dim),
            )

        elif self.mode == "attn_residual":
            self.gate = nn.Sequential(
                nn.Linear(self.emb_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, 1),
            )
            self.mix_logit = nn.Parameter(torch.tensor(-3.0))

        elif self.mode == "node_context_dim_gated":
            self.node_dim_gate = nn.Sequential(
                nn.Linear(3 * self.emb_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.emb_dim),
            )
        elif self.mode == "static_learned":
            self.alpha_logits = nn.Parameter(torch.zeros(self.num_candidates))
        elif self.mode == "static_learned_dim":
            self.alpha_logits = nn.Parameter(
                torch.zeros(self.num_candidates, self.emb_dim)
            )
        elif self.mode == "mean":
            pass

        else:
            raise ValueError(
                "candidate_fusion_mode must be one of: "
                "concat, mean, gated, context_gated, dim_gated, attn_residual"
            )

    def forward(
        self,
        view_embs: torch.Tensor,
        node_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        view_embs: [B, K, D]
        """
        if view_embs.ndim != 3:
            raise ValueError(f"Expected view_embs [B,K,D], got {tuple(view_embs.shape)}")

        B, K, D = view_embs.shape

        if self.mode == "mean":
            return view_embs.mean(dim=1), None

        if self.mode == "concat":
            fused = self.proj(view_embs.reshape(B, K * D))
            return fused, None

        if self.mode == "gated":
            scores = self.gate(view_embs).squeeze(-1)        # [B, K]
            alpha = torch.softmax(scores / self.temperature, dim=1)
            fused = torch.sum(alpha.unsqueeze(-1) * view_embs, dim=1)
            return fused, alpha
        if self.mode == "static_learned":
            alpha = torch.softmax(self.alpha_logits / self.temperature, dim=0)  # [K]
            fused = torch.sum(alpha.view(1, K, 1) * view_embs, dim=1)            # [B, D]
            alpha_log = alpha.view(1, K).expand(B, K)                           # [B, K]
            return fused, alpha_log
        if self.mode == "context_gated":
            flat = view_embs.reshape(B, K * D)
            scores = self.score_mlp(flat)                    # [B, K]
            alpha = torch.softmax(scores / self.temperature, dim=1)
            fused = torch.sum(alpha.unsqueeze(-1) * view_embs, dim=1)
            # print(f"context_gated alpha value: {alpha:.6f}")
            return fused, alpha

        if self.mode == "dim_gated":
            gates = torch.sigmoid(self.gate(view_embs))       # [B, K, D]

            # Normalize across candidates for each dimension.
            weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

            fused = torch.sum(weights * view_embs, dim=1)     # [B, D]

            # For logging, also save candidate-level average gate.
            alpha_log = weights.mean(dim=-1)                  # [B, K]
            return fused, alpha_log

        if self.mode == "attn_residual":
            scores = self.gate(view_embs).squeeze(-1)         # [B, K]
            alpha = torch.softmax(scores / self.temperature, dim=1)

            attn_fused = torch.sum(alpha.unsqueeze(-1) * view_embs, dim=1)
            mean_fused = view_embs.mean(dim=1)

            gamma = self.residual_max * torch.sigmoid(self.mix_logit)
            fused = mean_fused + gamma * (attn_fused - mean_fused)

            return fused, alpha

        if self.mode == "node_context_dim_gated":
            if node_emb is None:
                raise ValueError("node_context_dim_gated requires node_emb [B,D].")

            if node_emb.shape != (B, D):
                raise ValueError(
                    f"Expected node_emb shape {(B, D)}, got {tuple(node_emb.shape)}"
                )

            node_expand = node_emb.unsqueeze(1).expand(B, K, D)

            gate_input = torch.cat(
                [
                    node_expand,
                    view_embs,
                    node_expand * view_embs,
                ],
                dim=-1,
            )  # [B, K, 3D]

            gates = torch.sigmoid(self.node_dim_gate(gate_input))  # [B, K, D]

            # Normalize across candidates for each embedding dimension.
            weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

            fused = torch.sum(weights * view_embs, dim=1)  # [B, D]

            # For visualization only: average dimension-wise weights into candidate-level summary.
            alpha_log = weights.mean(dim=-1)  # [B, K]
            alpha_log = alpha_log / (alpha_log.sum(dim=1, keepdim=True) + 1e-8)

            return fused, alpha_log
        if self.mode == "static_learned":
            alpha = torch.softmax(self.alpha_logits / self.temperature, dim=0)  # [K]
            fused = torch.sum(alpha.view(1, K, 1) * view_embs, dim=1)            # [B, D]

            alpha_log = alpha.view(1, K).expand(B, K)                           # [B, K]
            return fused, alpha_log
        if self.mode == "static_learned_dim":
            weights = torch.softmax(self.alpha_logits / self.temperature, dim=0)  # [K, D]

            fused = torch.sum(
                weights.unsqueeze(0) * view_embs,   # [B, K, D]
                dim=1,
            )  # [B, D]

            alpha_log = weights.mean(dim=-1)        # [K]
            alpha_log = alpha_log / (alpha_log.sum() + 1e-8)
            alpha_log = alpha_log.view(1, K).expand(B, K)

            return fused, alpha_log
        raise RuntimeError(f"Unhandled candidate_fusion_mode={self.mode}")

class BankAwareSubjectMILClassifier(nn.Module):
    """MIL classifier for bank encoders; non-bank encoders still use SubjectMILClassifier."""
    def __init__(
        self,
        *,
        num_node_features: int,
        num_classes: int,
        encoder_type: str,
        num_nodes: int,
        num_candidates: int,
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        temp: float = 2.0,
        gnn_hidden_dim: int = 64,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        mil_pool_type: str = "gated",
        attn_dim: int = 128,
        candidate_fusion_mode: str = "concat",
        candidate_fusion_hidden_dim: Optional[int] = None,
        candidate_fusion_dropout: float = 0.0,
        bank_fusion_mode: str = "static",
        graph_backbone: str = "gatv2",
        **_: Any,
    ):
        super().__init__()
        self.encoder_type = str(encoder_type).lower()
        self.mil_pool_type = str(mil_pool_type).lower()
        if self.encoder_type == "mlinkx":
            self.graph_encoder = RawNodeBankEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=num_candidates,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                graph_emb_dim=graph_emb_dim,
                dropout=dropout,
                candidate_fusion_mode=candidate_fusion_mode,
                candidate_fusion_hidden_dim=candidate_fusion_hidden_dim,
                candidate_fusion_dropout=candidate_fusion_dropout,
                temp=temp,
            )
        elif self.encoder_type == "cnn_bank":
            self.graph_encoder = BankCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=num_candidates,
                branch_emb_dim=branch_emb_dim,
                graph_emb_dim=graph_emb_dim,
                dropout=dropout,
                include_node_branch=False,
            )
        elif self.encoder_type == "gnn_bank":
            self.graph_encoder = RawNodeGraphBankGNNEncoderLocal(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=num_candidates,
                graph_emb_dim=graph_emb_dim,
                branch_emb_dim=branch_emb_dim,
                node_hidden_dims=node_hidden_dims,
                gnn_hidden_dim=gnn_hidden_dim,
                graph_backbone=graph_backbone,
                candidate_fusion_mode=candidate_fusion_mode if candidate_fusion_mode != "concat" else "gated",
                candidate_fusion_hidden_dim=candidate_fusion_hidden_dim,
                candidate_fusion_dropout=candidate_fusion_dropout,
                dropout=dropout,
            )
        else:
            raise ValueError(f"BankAwareSubjectMILClassifier got non-bank encoder_type={encoder_type!r}")

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPoolLocal()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMILLocal(graph_emb_dim, attn_dim=attn_dim)
        else:
            raise ValueError("mil_pool_type must be one of: mean, gated")
        self.classifier = nn.Sequential(
            nn.Linear(int(graph_emb_dim), int(graph_emb_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(graph_emb_dim), int(num_classes)),
        )

    def encode_graphs(self, batch_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pyg_batch = batch_dict["pyg_batch"]
        adj_bank = batch_dict.get("adj_bank", None)
        conn_stack = batch_dict.get("conn_stack", None)
        if adj_bank is None and conn_stack is not None:
            adj_bank = conn_stack
        if conn_stack is None and adj_bank is not None:
            conn_stack = adj_bank
        if adj_bank is None:
            raise KeyError("Bank encoder requires batch_dict['adj_bank'] or batch_dict['conn_stack'].")
        if self.encoder_type in {"cnn_bank", "linkx_cnn_bank"}:
            return self.graph_encoder(pyg_batch, conn_stack)
        return self.graph_encoder(pyg_batch, adj_bank)

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        graph_emb, aux = self.encode_graphs(batch_dict)
        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        logits = self.classifier(bag_emb)
        out = {"logits": logits, "bag_emb": bag_emb, "graph_emb": graph_emb, "attn_list": attn_list}
        out.update({k: v for k, v in aux.items() if v is not None})
        return out

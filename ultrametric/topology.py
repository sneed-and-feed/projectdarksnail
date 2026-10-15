"""
Ultrametric AI — Dynamic Topology Router (PyTorch)

Maps continuous token embeddings into discrete Bruhat-Tits tree branches
via per-head factorized Gumbel-Softmax routing. Each attention head routes
independently, enabling different heads to attend to different hierarchical
sub-structures of the fractal tree.

Includes auxiliary load-balancing loss to prevent routing collapse
(Switch Transformer, Fedus et al. 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DynamicTopologyRouter(nn.Module):
    """
    Multi-Head Dynamic Topology Router.

    Projects token embeddings into per-head recursive p-adic tree paths.
    Each head gets its own routing decision at every level of the Bruhat-Tits
    tree, producing a genuinely nested hierarchical mask — not a flat partition.

    Args:
        embed_dim: token embedding dimension
        seq_len: maximum sequence length (determines tree depth)
        num_heads: number of independent routing heads
        p: tree arity (2 = binary Bruhat-Tits tree)
        tau: Gumbel-Softmax temperature (higher = softer routing)
        hard: if True, use straight-through estimator for discrete routing
    """

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int = 1,
        p: int = 2,
        tau: float = 1.0,
        hard: bool = True,
        init_mode: str = "collapse",
        levels: Optional[int] = None,
        tree_mode: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.p = p
        self.tau = tau
        self.hard = hard
        self.levels = levels if levels is not None else int(math.ceil(math.log(max(seq_len, 2), p)))
        self.tree_mode = tree_mode

        # Per-head routing: shared backbone, per-head projection heads
        self.backbone = nn.Linear(embed_dim, embed_dim)
        if self.tree_mode:
            self.num_internal = (self.p ** self.levels - 1) // (self.p - 1)
            self.route_heads = nn.Linear(embed_dim, num_heads * self.num_internal * self.p)
        else:
            self.route_heads = nn.Linear(embed_dim, num_heads * self.levels * self.p)

        with torch.no_grad():
            if init_mode == "random":
                # Pure random projection: tests if pre-trained representation geometry alone partitions tokens
                nn.init.normal_(self.backbone.weight, std=0.02)
                nn.init.zeros_(self.backbone.bias)
                nn.init.normal_(self.route_heads.weight, std=0.02)
                nn.init.zeros_(self.route_heads.bias)
            else:
                # Deterministic Collapse Initialization for Continuous Logit Homotopy
                nn.init.zeros_(self.route_heads.weight)
                if self.tree_mode:
                    b = torch.full((self.num_heads * self.num_internal * self.p,), -5.0)
                    for h in range(self.num_heads):
                        for node in range(self.num_internal):
                            idx = (h * self.num_internal * self.p) + (node * self.p) + 0
                            b[idx] = 5.0
                    self.route_heads.bias.copy_(b)
                else:
                    b = torch.full((self.num_heads * self.levels * self.p,), -5.0)
                    for h in range(self.num_heads):
                        for l in range(self.levels):
                            idx = (h * self.levels * self.p) + (l * self.p) + 0
                            b[idx] = 5.0
                    self.route_heads.bias.copy_(b)


    def forward(
        self, x: torch.Tensor, tau_override: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        tau = tau_override if tau_override is not None else self.tau

        h = F.gelu(self.backbone(x))  # (batch, seq_len, embed_dim)
        logits = self.route_heads(h).clamp(min=-50000.0, max=50000.0)
        
        if self.tree_mode:
            logits = logits.view(batch_size, seq_len, self.num_heads, self.num_internal, self.p)
            log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
            
            path_log_probs = torch.zeros((batch_size, seq_len, self.num_heads, 1), device=logits.device, dtype=torch.float32)
            current_node_start = 0
            
            for l in range(self.levels):
                num_nodes_level = self.p ** l
                level_log_probs = log_probs[:, :, :, current_node_start : current_node_start + num_nodes_level, :]
                new_path_log_probs = path_log_probs.unsqueeze(-1) + level_log_probs
                path_log_probs = new_path_log_probs.view(batch_size, seq_len, self.num_heads, num_nodes_level * self.p)
                current_node_start += num_nodes_level
                
            if self.training:
                sampled_leaves = F.gumbel_softmax(path_log_probs, tau=tau, hard=self.hard, dim=-1)
            else:
                indices = path_log_probs.argmax(dim=-1)
                sampled_leaves = F.one_hot(indices, num_classes=self.p**self.levels).float()
                
            if getattr(self, "leaf_to_assignments", None) is None:
                mapping = torch.zeros((self.p**self.levels, self.levels, self.p), device=logits.device, dtype=torch.float32)
                for leaf_idx in range(self.p**self.levels):
                    val = leaf_idx
                    for l in range(self.levels - 1, -1, -1):
                        choice = val % self.p
                        mapping[leaf_idx, l, choice] = 1.0
                        val = val // self.p
                self.register_buffer("leaf_to_assignments", mapping, persistent=False)
                
            assignments = torch.matmul(sampled_leaves, self.leaf_to_assignments.view(self.p**self.levels, -1))
            assignments = assignments.view(batch_size, seq_len, self.num_heads, self.levels, self.p).to(logits.dtype)
            assignments = assignments.permute(0, 2, 1, 3, 4)
            
            num_paths = self.p ** self.levels
            f = (sampled_leaves.detach() > 0.5).float().mean(dim=1)
            P = path_log_probs.exp().mean(dim=1)
            load_balance_loss = (f * P).sum(dim=-1).mean() * num_paths
        else:
            logits = logits.view(batch_size, seq_len, self.num_heads, self.levels, self.p)
            logits = logits.permute(0, 2, 1, 3, 4)  # (batch, heads, seq_len, levels, p)

            if self.training:
                flat = logits.reshape(-1, self.p)
                sampled = F.gumbel_softmax(flat.to(torch.float32), tau=tau, hard=self.hard, dim=-1).to(logits.dtype)
                assignments = sampled.view_as(logits)
            else:
                indices = logits.argmax(dim=-1)
                assignments = F.one_hot(indices, num_classes=self.p).float()

            load_balance_loss = self.compute_load_balance_loss(assignments)

        return assignments, load_balance_loss

    @staticmethod
    def compute_load_balance_loss(assignments: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer-style load balancing loss.

        Computes the joint probability of the full root-to-leaf path to prevent
        the model from collapsing into highly correlated parallel choices.
        """
        B, H, S, L, p = assignments.shape
        if L <= 8:
            joint_P = assignments[..., 0, :]
            for l in range(1, L):
                joint_P = joint_P.unsqueeze(-1) * assignments[..., l, :].unsqueeze(-2)
                joint_P = joint_P.view(B, H, S, -1)
            num_paths = p ** L
            f = (joint_P.detach() > 0.5).float().mean(dim=2)
            P = joint_P.mean(dim=2)
            loss = (f * P).sum(dim=-1).mean() * num_paths
        else:
            f = (assignments.detach() > 0.5).float().mean(dim=2)
            P = assignments.mean(dim=2)
            loss = (f * P).sum(dim=-1).mean() * p
        return loss

    @staticmethod
    def get_tau_schedule(
        step: int,
        warmup_steps: int = 2000,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
    ) -> float:
        """
        Cosine temperature annealing: soft routing → hard routing over training.

        Args:
            step: current training step
            warmup_steps: total annealing steps
            tau_start: initial temperature (soft)
            tau_end: final temperature (hard)
        Returns:
            tau: current temperature value
        """
        if step >= warmup_steps:
            return tau_end
        progress = step / warmup_steps
        return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Ultrametric Mask Utilities
# ============================================================================


def get_dynamic_ultrametric_mask(
    assignments: torch.Tensor, p: int = 2, max_dist: Optional[int] = None, local_window: int = 0
) -> torch.Tensor:
    """
    Generates a dynamic ultrametric mask from learned routing assignments.

    Two tokens attend to each other if their expected p-adic distance
    (height of lowest common ancestor in the Bruhat-Tits tree) ≤ max_dist.

    Supports both shared routing (4D input) and per-head routing (5D input).

    Args:
        assignments: (batch, seq_len, levels, p) shared routing, OR
                     (batch, heads, seq_len, levels, p) per-head routing
        p: tree arity
        max_dist: maximum p-adic distance for attention. Default: levels // 2

    Returns:
        mask: (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
    """
    if assignments.dim() == 5:
        B, H, S, L, P = assignments.shape
        a_flat = assignments.reshape(B * H, S, L, P)
        mask_flat = _compute_distance_mask(a_flat, L, max_dist, local_window)
        return mask_flat.view(B, H, S, S)
    else:
        B, S, L, P = assignments.shape
        return _compute_distance_mask(assignments, L, max_dist, local_window)


def _compute_distance_mask(
    assignments: torch.Tensor, levels: int, max_dist: Optional[int], local_window: int = 0
) -> torch.Tensor:
    """
    Core p-adic distance mask computation via reversed cumulative product.

    The expected p-adic distance between tokens i and j is:
        d(i,j) = levels - sum_{l=0}^{levels-1} prod_{m=l}^{levels-1} M[i,j,m]
    where M[i,j,l] is the probability that tokens i,j share branch l.
    """
    # M[b, i, j, l] = prob tokens i, j agree at level l
    M = torch.einsum("bilp,bjlp->bijl", assignments, assignments)

    # CRITICAL FIX: The previous version flipped M and computed suffix distance instead of prefix distance.
    # We must compute cummin directly on M (starting from the root at l=0) to ensure a true tree topology.
    # cummin is perfectly stable compared to cumprod.
    P_flipped = M.cummin(dim=-1)[0]
    sum_P = P_flipped.sum(dim=-1)
    expected_dist = levels - sum_P

    if max_dist is None:
        max_dist = max(levels // 2, 1)

    # Straight-Through Estimator: hard mask forward, soft gradient backward
    temperature = 0.5
    soft_mask = torch.sigmoid((max_dist - expected_dist) / temperature)
    hard_mask = (expected_dist <= max_dist).float()
    
    if local_window > 0:
        S = expected_dist.shape[-1]
        idx = torch.arange(S, device=expected_dist.device)
        band = (torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= local_window).float()
        hard_mask = torch.clamp(hard_mask + band, max=1.0)
        
    # CRITICAL: Attention Sink! Always keep the first token visible so Softmax doesn't collapse
    hard_mask[..., :, 0] = 1.0
        
    mask = hard_mask.detach() - soft_mask.detach() + soft_mask
    return mask


def get_ultrametric_mask(seq_len: int, p: int = 2) -> torch.Tensor:
    """
    Generates a static boolean mask for ultrametric (p-adic) attention.

    In a standard dense transformer, every token attends to every other token.
    In an ultrametric Bruhat-Tits topology, tokens are leaves on a tree.
    Tokens only strongly attend to tokens that share a deep common ancestor.

    This function generates a block-sparse mask where the density decreases
    as the p-adic distance increases, dropping connections that are
    topologically "far".

    Args:
        seq_len: sequence length
        p: tree arity (default: 2 for binary tree)
    Returns:
        mask: (seq_len, seq_len) boolean tensor
    """
    levels = int(math.ceil(math.log(max(seq_len, 2), p)))
    pad_len = p**levels

    mask = torch.zeros((pad_len, pad_len), dtype=torch.bool)
    for level in range(levels):
        block_size = p**level
        for i in range(0, pad_len, block_size):
            mask[i : i + block_size, i : i + block_size] = True

    return mask[:seq_len, :seq_len]


def compute_p_adic_distance(i: int, j: int, p: int = 2) -> int:
    """
    Computes the p-adic distance between two token indices.
    Corresponds to the height of their lowest common ancestor in a p-ary tree.
    """
    if i == j:
        return 0
    diff = i ^ j
    return int(math.floor(math.log(diff, p))) + 1

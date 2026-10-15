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
from typing import Optional, Sequence, Tuple


class DynamicTopologyRouter(nn.Module):
    """
    Multi-Head Dynamic Topology Router (Supports QK-Mode and Multi-Prime Adèlic Head Groups).

    Projects token embeddings (or per-head block-pooled post-RoPE QK descriptors)
    into per-head recursive p-adic tree paths. Each head gets its own routing
    decision at every level of the Bruhat-Tits tree, producing a genuinely
    nested hierarchical mask — not a flat partition.

    Args:
        embed_dim: token embedding dimension
        seq_len: maximum sequence length (determines tree depth)
        num_heads: number of independent routing heads
        p: tree arity (2 = binary Bruhat-Tits tree)
        tau: Gumbel-Softmax temperature (higher = softer routing)
        hard: if True, use straight-through estimator for discrete routing
        init_mode: "collapse" (deterministic step-0 identity) or "random"
        levels: optional explicit tree depth
        tree_mode: if True, parameterize full internal nodes of the tree
        qk_mode: if True, routes directly on post-RoPE block-pooled Q/K descriptors
                 of shape (batch, num_heads, num_blocks, 2 * head_dim)
        head_dim: per-head dimension when qk_mode=True (defaults to embed_dim // num_heads)
        prime_arities: optional sequence of prime arities across GQA head groups
                       (e.g. [2, 3]), partitioning heads into contiguous prime groups
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
        qk_mode: bool = False,
        head_dim: Optional[int] = None,
        prime_arities: Optional[Sequence[int]] = None,
        triad_mode: bool = False,
        d_shell: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tau = tau
        self.hard = hard
        self.init_mode = init_mode
        self.tree_mode = tree_mode
        self.qk_mode = qk_mode
        self.triad_mode = bool(triad_mode)
        self.d_shell = max(2, int(d_shell))
        self.head_dim = head_dim if head_dim is not None else max(1, embed_dim // max(1, num_heads))

        if prime_arities is not None and len(prime_arities) > 0:
            self.prime_arities = [int(pa) for pa in prime_arities]
            self.p = max(self.prime_arities)
            num_groups = len(self.prime_arities)
            self.head_arities = []
            for h in range(num_heads):
                g = min((h * num_groups) // max(1, num_heads), num_groups - 1)
                self.head_arities.append(self.prime_arities[g])
        else:
            self.prime_arities = None
            self.p = int(p)
            self.head_arities = [self.p] * num_heads

        if levels is not None:
            self.levels = int(levels)
        else:
            min_p = min(self.head_arities) if self.head_arities else self.p
            self.levels = int(math.ceil(math.log(max(seq_len, 2), max(min_p, 2))))

        self.head_levels = [
            int(math.ceil(math.log(max(seq_len, 2), max(p_h, 2)))) if levels is None else self.levels
            for p_h in self.head_arities
        ]

        in_dim = (2 * self.head_dim) if self.qk_mode else self.embed_dim
        self.in_dim = in_dim

        # Per-head routing: shared backbone, per-head projection heads
        self.backbone = nn.Linear(in_dim, in_dim)
        if self.tree_mode:
            self.num_internal = (self.p ** self.levels - 1) // max(self.p - 1, 1)
            self.out_per_head = self.num_internal * self.p
            self.route_heads = nn.Linear(in_dim, num_heads * self.out_per_head)
        else:
            self.num_internal = 0
            self.out_per_head = self.levels * self.p
            self.route_heads = nn.Linear(in_dim, num_heads * self.out_per_head)

        if self.triad_mode and not self.tree_mode:
            M = self.d_shell
            ctx_dim = self.levels * self.p + M
            self.shell_inject = nn.Parameter(torch.empty(self.num_heads, self.levels, M, in_dim))
            self.triad_weight = nn.Parameter(
                torch.empty(self.num_heads, max(1, self.levels - 1), ctx_dim, M, M)
            )
            self.triad_readout = nn.Parameter(
                torch.empty(self.num_heads, max(1, self.levels - 1), self.p, M)
            )
            self.log_viscosity = nn.Parameter(torch.full((1, self.num_heads, 1, 1), -6.0))
        else:
            self.shell_inject = None
            self.triad_weight = None
            self.triad_readout = None
            self.log_viscosity = None

        self.last_triad_flux: Optional[torch.Tensor] = None

        with torch.no_grad():
            if init_mode == "random":
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

            if self.triad_mode and not self.tree_mode:
                M = self.d_shell
                for h in range(self.num_heads):
                    w_orth = torch.empty(self.levels * M, in_dim)
                    nn.init.orthogonal_(w_orth, gain=1.0)
                    self.shell_inject[h].copy_(w_orth.view(self.levels, M, in_dim))
                nn.init.normal_(self.triad_weight, std=0.5 / math.sqrt(M))
                if init_mode == "random":
                    nn.init.normal_(self.triad_readout, std=0.5 / math.sqrt(M))
                else:
                    # Exact Inviscid Homotopy: zero triad_readout at Step 0 preserves 100% branch-0 collapse
                    # while non-zero shell_inject & triad_weight provide immediate O(1) Step-1 gradients!
                    nn.init.zeros_(self.triad_readout)

    def _project_logits(self, x: torch.Tensor, tau: float = 1.0) -> Tuple[torch.Tensor, int, int]:
        """
        Projects input tensor x into per-head routing logits in float32 (preventing bf16 ULP bias freeze).
        Supports both flat V3 projection and V4 Tao-Katz-Pavlovic skew-symmetric dyadic triad cascade.
        """
        orig_dtype = x.dtype
        x_f = x.float()
        w_bb = self.backbone.weight.float()
        b_bb = self.backbone.bias.float() if self.backbone.bias is not None else None
        h = F.gelu(F.linear(x_f, w_bb, b_bb))

        if x.dim() == 4:
            batch_size, num_heads, seq_len, _ = x.shape
            assert num_heads == self.num_heads, f"Expected {self.num_heads} heads, got {num_heads}"
            w = self.route_heads.weight.float().view(self.num_heads, self.out_per_head, self.in_dim)
            logits_bhso = torch.einsum("bhsi,hoi->bhso", h, w)
            if self.route_heads.bias is not None:
                b = self.route_heads.bias.float().view(1, self.num_heads, 1, self.out_per_head)
                logits_bhso = logits_bhso + b
            base_logits = logits_bhso.permute(0, 2, 1, 3)  # (B, S, H, out_per_head)
            h_4d = h  # (B, H, S, in_dim)
        else:
            batch_size, seq_len, _ = x.shape
            w = self.route_heads.weight.float()
            b = self.route_heads.bias.float() if self.route_heads.bias is not None else None
            base_logits = F.linear(h, w, b)  # (B, S, H * out_per_head)
            h_4d = h.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.in_dim)

        if self.triad_mode and not self.tree_mode and self.levels > 1:
            base_5d = base_logits.view(batch_size, seq_len, self.num_heads, self.levels, self.p).permute(0, 2, 1, 3, 4)
            f_shells = torch.einsum("bhnd,hlmd->bhnlm", h_4d, self.shell_inject.float())  # (B, H, S, L, M)
            W_skew = self.triad_weight.float() - self.triad_weight.float().transpose(-1, -2)
            nu = torch.exp(self.log_viscosity.float())

            shell_logits = [base_5d[:, :, :, 0, :]]
            u_prev = f_shells[:, :, :, 0, :]
            flux_per_level = []

            for ell in range(1, self.levels):
                f_ell = f_shells[:, :, :, ell, :]
                # Detached idempotent parent branch probabilities a^{(0..ell-1)} in [0, 1]^p
                anc_probs = torch.cat(
                    [F.softmax(z.detach() / max(tau, 0.2), dim=-1) for z in shell_logits], dim=-1
                )
                pad_len = self.levels * self.p - anc_probs.shape[-1]
                if pad_len > 0:
                    anc_probs = F.pad(anc_probs, (0, pad_len))
                parent_ctx = torch.cat([anc_probs, F.normalize(u_prev.detach(), p=2, dim=-1)], dim=-1)

                # Skew-symmetric bilinear triad pump B_ell(parent_ctx, f_ell) in so(M)
                triad_pump = torch.einsum("bhni,bhnj,hijm->bhnm", parent_ctx, f_ell, W_skew[:, ell - 1])
                flux_per_level.append(torch.norm(triad_pump, p=2, dim=-1))
                damping = torch.exp(-nu * (float(self.p) ** (2 * ell)) * 1e-3)
                u_next = damping * (u_prev + f_ell + triad_pump)
                triad_scale = math.sqrt(float(self.in_dim) / max(1.0, float(self.d_shell)))
                triad_logit = triad_scale * torch.einsum(
                    "bhnm,hpm->bhnp", triad_pump, self.triad_readout[:, ell - 1].float()
                )
                shell_logits.append(base_5d[:, :, :, ell, :] + damping * triad_logit)
                u_prev = u_next

            self.last_triad_flux = torch.stack(flux_per_level, dim=-1).mean(dim=-1).detach()  # (B, H, S)
            logits_5d = torch.stack(shell_logits, dim=3)  # (B, H, S, L, p)
            logits_out = logits_5d.permute(0, 2, 1, 3, 4).reshape(batch_size, seq_len, -1)
            return logits_out.clamp(min=-50000.0, max=50000.0).to(orig_dtype), batch_size, seq_len

        return base_logits.clamp(min=-50000.0, max=50000.0).to(orig_dtype), batch_size, seq_len

    def forward(
        self, x: torch.Tensor, tau_override: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = tau_override if tau_override is not None else self.tau
        logits, batch_size, seq_len = self._project_logits(x, tau=tau)

        if self.tree_mode:
            logits = logits.view(batch_size, seq_len, self.num_heads, self.num_internal, self.p)
            log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)

            path_log_probs = torch.zeros(
                (batch_size, seq_len, self.num_heads, 1), device=logits.device, dtype=torch.float32
            )
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
                mapping = torch.zeros(
                    (self.p**self.levels, self.levels, self.p), device=logits.device, dtype=torch.float32
                )
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

            if self.prime_arities is not None and len(set(self.head_arities)) > 1:
                # Multi-prime Adèlic routing per head group
                assignments_list = []
                lb_losses = []
                num_groups = len(self.prime_arities)
                for g, p_g in enumerate(self.prime_arities):
                    h_start = (g * self.num_heads) // num_groups
                    h_end = ((g + 1) * self.num_heads) // num_groups
                    if h_start >= h_end:
                        continue
                    sub_logits = logits[:, h_start:h_end, :, :, :p_g]
                    if self.training:
                        flat = sub_logits.reshape(-1, p_g)
                        sampled = F.gumbel_softmax(
                            flat.to(torch.float32), tau=tau, hard=self.hard, dim=-1
                        ).to(logits.dtype)
                        sub_assign = sampled.view_as(sub_logits)
                    else:
                        indices = sub_logits.argmax(dim=-1)
                        sub_assign = F.one_hot(indices, num_classes=p_g).to(logits.dtype)
                    lb_losses.append(self.compute_load_balance_loss(sub_assign))
                    if p_g < self.p:
                        sub_assign = F.pad(sub_assign, (0, self.p - p_g), value=0.0)
                    assignments_list.append(sub_assign)
                assignments = torch.cat(assignments_list, dim=1)
                load_balance_loss = sum(lb_losses) / max(len(lb_losses), 1)
            else:
                if self.training:
                    flat = logits.reshape(-1, self.p)
                    sampled = F.gumbel_softmax(
                        flat.to(torch.float32), tau=tau, hard=self.hard, dim=-1
                    ).to(logits.dtype)
                    assignments = sampled.view_as(logits)
                else:
                    indices = logits.argmax(dim=-1)
                    assignments = F.one_hot(indices, num_classes=self.p).float()

                load_balance_loss = self.compute_load_balance_loss(assignments)

        return assignments, load_balance_loss

    @staticmethod
    def compute_load_balance_loss(assignments: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer-style load balancing loss over root-to-leaf paths (in float32).

        Computes the joint probability of the full root-to-leaf path so product-rule
        gradients wake up all tree levels simultaneously at Step 0 without over-penalizing
        the laminar root shell l=0 relative to deep leaf shells.
        """
        B, H, S, L, p = assignments.shape
        a_f32 = assignments.float()
        if L <= 8 and (p ** L) <= 4096:
            joint_P = a_f32[..., 0, :]
            for l in range(1, L):
                joint_P = joint_P.unsqueeze(-1) * a_f32[..., l, :].unsqueeze(-2)
                joint_P = joint_P.view(B, H, S, -1)
            num_paths = p ** L
            f = (joint_P.detach() > 0.5).float().mean(dim=2)
            P = joint_P.mean(dim=2)
            loss = (f * P).sum(dim=-1).mean() * float(num_paths)
        else:
            f = (a_f32.detach() > 0.5).float().mean(dim=2)
            P = a_f32.mean(dim=2)
            loss = (f * P).sum(dim=-1).mean() * float(p)
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
        """
        if step >= warmup_steps:
            return tau_end
        progress = step / warmup_steps
        return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))


class MultiPrimeTopologyRouter(DynamicTopologyRouter):
    """
    Convenience subclass of DynamicTopologyRouter configured for Multi-Prime
    Adèlic GQA head groups (default prime_arities=(2, 3)).
    """

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int = 2,
        prime_arities: Sequence[int] = (2, 3),
        tau: float = 1.0,
        hard: bool = True,
        init_mode: str = "collapse",
        levels: Optional[int] = None,
        tree_mode: bool = False,
        qk_mode: bool = False,
        head_dim: Optional[int] = None,
        triad_mode: bool = False,
        d_shell: int = 8,
    ):
        super().__init__(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            p=max(prime_arities),
            tau=tau,
            hard=hard,
            init_mode=init_mode,
            levels=levels,
            tree_mode=tree_mode,
            qk_mode=qk_mode,
            head_dim=head_dim,
            prime_arities=prime_arities,
            triad_mode=triad_mode,
            d_shell=d_shell,
        )


class TriadTopologyRouter(DynamicTopologyRouter):
    """
    V4 Tao–Katz–Pavlović Dyadic Triad Cascade Router.

    Replaces flat independent tree-level projections with a recursive skew-symmetric
    bilinear triad cascade B_ell(u^{(0..ell-1)}, f^{(ell)}(x)) in so(d_shell) down
    the Bruhat-Tits tree levels ell = 0 .. L-1, damped by scale-covariant p-adic
    viscosity exp(-nu_h * p^{2*ell}).
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
        qk_mode: bool = True,
        head_dim: Optional[int] = None,
        prime_arities: Optional[Sequence[int]] = None,
        d_shell: int = 8,
    ):
        super().__init__(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            p=p,
            tau=tau,
            hard=hard,
            init_mode=init_mode,
            levels=levels,
            tree_mode=False,
            qk_mode=qk_mode,
            head_dim=head_dim,
            prime_arities=prime_arities,
            triad_mode=True,
            d_shell=d_shell,
        )


def compute_expected_block_distance(assignments: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Computes the differentiable expected p-adic (cophenetic) distance directly at
    block granularity without expanding to N x N tokens.

    Uses `M.cummin` in the forward pass and adds a depth-covariant dyadic STE
    tie-breaking residual in the backward pass (`w_ell = 0.05 * 2^(ell - (L-1))`),
    eliminating Step-0 cummin tie-breaking starvation at deeper shells ell >= 1
    while keeping the root shell ell=0 laminar (preventing the 5x root-shattering
    bias of unweighted prefix products).
    """
    levels = assignments.shape[-2]
    if assignments.dim() == 5:
        M = torch.einsum("bhilp,bhjlp->bhijl", assignments, assignments)
    elif assignments.dim() == 4:
        M = torch.einsum("bilp,bjlp->bijl", assignments, assignments)
    else:
        raise ValueError(f"Expected 4D or 5D assignments tensor, got {assignments.dim()}D")

    P_prefix = M.cummin(dim=-1)[0]
    if M.requires_grad:
        ell_idx = torch.arange(levels, device=M.device, dtype=M.dtype)
        dyadic_w = 0.05 * (2.0 ** (ell_idx - (levels - 1)))
        P_tiebreak = M * dyadic_w
        P_prefix = P_prefix + (P_tiebreak - P_tiebreak.detach())
    sum_P = P_prefix.sum(dim=-1)
    return levels - sum_P



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
        chunk_size = 4
        if H > chunk_size:
            masks = []
            for h_start in range(0, H, chunk_size):
                h_end = min(h_start + chunk_size, H)
                sub_a = assignments[:, h_start:h_end].reshape(-1, S, L, P)
                sub_mask = _compute_distance_mask(sub_a, L, max_dist, local_window)
                masks.append(sub_mask.view(B, h_end - h_start, S, S))
            return torch.cat(masks, dim=1)
        else:
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

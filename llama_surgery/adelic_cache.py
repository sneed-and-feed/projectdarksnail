"""
Surprisal-Guided Two-Tier Adèlic KV-Cache Condenser (Milestone 3.1).

Replaces naive greedy clustering and zero-key padding with a fully vectorized
two-tier condensation pipeline over far-history tokens:
  1. Tier 1 (Outlier / Needle Shield): Protects `outlier_budget` tokens that reside
     in sparse peripheral buckets (occupancy <= k_outlier), exhibit high semantic
     isolation (1 - max_other_cosine_sim), or carry high cumulative attention
     surprisal / norm. Their exact (K, V) representations are preserved untouched.
  2. Tier 2 (Vectorized Medoid + Mean-Value Condenser): Partitions the remaining
     far-history tokens into `condensed_budget` guaranteed non-empty clusters per
     head, computes positional variance Var(P_S), scales the cluster medoid key by
     gamma = exp(-lambda * Var(P_S)), and averages V.

Guarantees an exact, uniform physical cache length across all heads with zero
dummy (K=0, V=0) padding (fixing Trap 5).
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


class SurprisalAdelicCache(DynamicCache):
    """
    Two-Tier Vectorized Adèlic KV-Cache with Needle Protection and Synthetic RoPE Decay.
    Compatible with HuggingFace `DynamicCache` (including transformers 4.x and 5.x).
    """

    def __init__(
        self,
        max_capacity: int = 128,
        local_window: int = 64,
        outlier_budget: Optional[int] = None,
        condensed_budget: Optional[int] = None,
        lambda_decay: float = 0.05,
        k_outlier: int = 2,
        beltrami_mode: bool = True,
        viscous_p: int = 2,
    ):
        super().__init__()
        assert max_capacity > local_window, "max_capacity must be strictly greater than local_window"
        self.max_capacity = int(max_capacity)
        self.local_window = int(local_window)
        far_budget = self.max_capacity - self.local_window

        if outlier_budget is None and condensed_budget is None:
            self.outlier_budget = max(1, far_budget // 4)
            self.condensed_budget = far_budget - self.outlier_budget
        elif outlier_budget is not None and condensed_budget is None:
            self.outlier_budget = int(outlier_budget)
            self.condensed_budget = max(1, far_budget - self.outlier_budget)
        elif outlier_budget is None and condensed_budget is not None:
            self.condensed_budget = int(condensed_budget)
            self.outlier_budget = max(0, far_budget - self.condensed_budget)
        else:
            self.outlier_budget = int(outlier_budget)
            self.condensed_budget = int(condensed_budget)

        assert self.outlier_budget + self.condensed_budget + self.local_window <= self.max_capacity, (
            "outlier_budget + condensed_budget + local_window must not exceed max_capacity"
        )

        self.lambda_decay = float(lambda_decay)
        self.k_outlier = int(k_outlier)
        self.beltrami_mode = bool(beltrami_mode)
        self.viscous_p = max(1, int(viscous_p))

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.pos_cache: List[torch.Tensor] = []
        self.surprisal_cache: List[torch.Tensor] = []
        self._true_seen_tokens = 0

    def _sync_hf_layers(self, layer_idx: int):
        """Synchronizes internal tensors with transformers 5.x `self.layers` if present."""
        if hasattr(self, "layers"):
            while len(self.layers) <= layer_idx:
                from transformers.cache_utils import DynamicLayer
                self.layers.append(DynamicLayer())
            self.layers[layer_idx].keys = self.key_cache[layer_idx]
            self.layers[layer_idx].values = self.value_cache[layer_idx]
            self.layers[layer_idx].is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, S_new, D = key_states.shape
        device = key_states.device

        if layer_idx == 0:
            start_pos = self._true_seen_tokens
            self._true_seen_tokens += S_new
            self._seen_tokens = self._true_seen_tokens
        else:
            start_pos = max(0, self._true_seen_tokens - S_new)

        new_pos = (
            torch.arange(start_pos, start_pos + S_new, dtype=torch.float32, device=device)
            .view(1, 1, S_new)
            .expand(B, H, S_new)
        )

        if cache_kwargs is not None and "surprisal" in cache_kwargs and cache_kwargs["surprisal"] is not None:
            new_surp = cache_kwargs["surprisal"].to(device=device, dtype=torch.float32)
            if new_surp.dim() == 2:
                new_surp = new_surp.unsqueeze(1).expand(B, H, S_new)
        else:
            new_surp = torch.zeros((B, H, S_new), dtype=torch.float32, device=device)

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.pos_cache.append(None)
            self.surprisal_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.pos_cache[layer_idx] = new_pos
            self.surprisal_cache[layer_idx] = new_surp
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.pos_cache[layer_idx] = torch.cat([self.pos_cache[layer_idx], new_pos], dim=-1)
            self.surprisal_cache[layer_idx] = torch.cat([self.surprisal_cache[layer_idx], new_surp], dim=-1)

        current_length = self.key_cache[layer_idx].shape[-2]
        if current_length > self.max_capacity:
            router_leaf_ids = cache_kwargs.get("router_leaf_ids", None) if cache_kwargs is not None else None
            self._condense_layer(layer_idx, router_leaf_ids=router_leaf_ids)

        self._sync_hf_layers(layer_idx)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @staticmethod
    def compute_beltrami_triad_flux(far_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the V4 Beltrami Triad-Flux score across coarse and fine frequency shells of `far_k`:
          - Cross-shell exterior curl: ||u_coarse ^ u_fine||_2 = sqrt(1 - <u_coarse, u_fine>^2)
          - Local fine-shell phase innovation: ||u_fine(t) - u_fine(t-1)||_2
        Protects multi-token passcodes (e.g. '9481-OMEGA') and low-norm semantic needles
        while suppressing high-norm laminar sink tokens where coarse and fine shells are collinear.
        """
        B, H, far_len, D = far_k.shape
        half_d = max(1, D // 2)
        u_coarse = F.normalize(far_k[..., :half_d].float(), p=2, dim=-1)
        u_fine = F.normalize(far_k[..., -half_d:].float(), p=2, dim=-1)
        cos_cf = (u_coarse * u_fine).sum(dim=-1).clamp(min=-1.0, max=1.0)
        cross_shell_curl = torch.sqrt(torch.clamp(1.0 - cos_cf * cos_cf, min=0.0))  # (B, H, far_len)

        fine_innovation = torch.zeros_like(cross_shell_curl)
        if far_len > 1:
            diff_prev = torch.norm(u_fine[:, :, 1:, :] - u_fine[:, :, :-1, :], p=2, dim=-1)
            fine_innovation[:, :, 1:] = diff_prev
            fine_innovation[:, :, :-1] = torch.maximum(fine_innovation[:, :, :-1], diff_prev)
        beltrami_flux = cross_shell_curl * fine_innovation
        return beltrami_flux, cross_shell_curl

    def _condense_layer(
        self,
        layer_idx: int,
        router_leaf_ids: Optional[torch.Tensor] = None,
    ):
        keys = self.key_cache[layer_idx]         # (B, H, S, D)
        values = self.value_cache[layer_idx]     # (B, H, S, D)
        positions = self.pos_cache[layer_idx]    # (B, H, S)
        surprisal = self.surprisal_cache[layer_idx]  # (B, H, S)

        B, H, S, D = keys.shape
        far_len = S - self.local_window
        target_far_len = self.outlier_budget + self.condensed_budget
        if far_len <= target_far_len or far_len <= 0:
            return

        far_k = keys[:, :, :far_len, :]
        far_v = values[:, :, :far_len, :]
        far_pos = positions[:, :, :far_len]
        far_surp = surprisal[:, :, :far_len]

        local_k = keys[:, :, far_len:, :]
        local_v = values[:, :, far_len:, :]
        local_pos = positions[:, :, far_len:]
        local_surp = surprisal[:, :, far_len:]

        K_out = min(self.outlier_budget, far_len - 1) if self.condensed_budget > 0 else min(self.outlier_budget, far_len)
        rem_len = far_len - K_out
        C = min(self.condensed_budget, rem_len)

        # Normalize far_v and far_k for directional similarity & isolation scoring
        norm_v = F.normalize(far_v.float(), p=2, dim=-1)  # (B, H, far_len, D)
        norm_k = F.normalize(far_k.float(), p=2, dim=-1)  # (B, H, far_len, D)

        # Initial bucket occupancy estimation
        num_init_buckets = max(C, 4)
        if router_leaf_ids is not None and router_leaf_ids.shape[-1] >= far_len:
            init_bucket = (router_leaf_ids[:, :H, :far_len].long() % num_init_buckets)
        else:
            anchor_idx = torch.linspace(0, far_len - 1, steps=num_init_buckets, device=keys.device).long()
            anchors = norm_v[:, :, anchor_idx, :]  # (B, H, num_init_buckets, D)
            init_bucket = torch.matmul(norm_v, anchors.transpose(-2, -1)).argmax(dim=-1)  # (B, H, far_len)

        one_hot_b = F.one_hot(init_bucket, num_classes=num_init_buckets).float()
        bucket_occ = one_hot_b.sum(dim=-2)  # (B, H, num_init_buckets)
        token_occ = torch.gather(bucket_occ, dim=-1, index=init_bucket)  # (B, H, far_len)

        # Pairwise similarity to detect isolated semantic needles
        sim_v = torch.matmul(norm_v, norm_v.transpose(-2, -1))
        sim_k = torch.matmul(norm_k, norm_k.transpose(-2, -1))
        joint_sim = 0.5 * (sim_v + sim_k)
        eye = torch.eye(far_len, dtype=torch.bool, device=keys.device).unsqueeze(0).unsqueeze(0)
        max_other_sim = joint_sim.masked_fill(eye, -1.0).max(dim=-1).values  # (B, H, far_len)
        isolation_score = 1.0 - max_other_sim  # High for orthogonal needle tokens

        kv_norm = torch.norm(far_k.float(), p=2, dim=-1) + torch.norm(far_v.float(), p=2, dim=-1)
        kv_norm = kv_norm / (kv_norm.amax(dim=-1, keepdim=True) + 1e-6)
        peripheral_bonus = (token_occ <= self.k_outlier).float() * 1.5

        if self.beltrami_mode and D >= 2:
            beltrami_flux, cross_shell_curl = self.compute_beltrami_triad_flux(far_k)
            # Gate static kv_norm by cross_shell_curl so laminar formatting sinks cannot hijack Tier 1
            gated_kv_norm = kv_norm * (0.15 + 0.85 * cross_shell_curl)
            outlier_score = (
                far_surp
                + 2.5 * isolation_score
                + 3.5 * beltrami_flux
                + gated_kv_norm
                + peripheral_bonus
            )
        else:
            outlier_score = far_surp + 2.5 * isolation_score + kv_norm + peripheral_bonus

        # Always protect Token 0 (Attention Sink) in Tier 1 if K_out > 0
        if K_out > 0:
            outlier_score = outlier_score.clone()
            outlier_score[:, :, 0] = 1e9
            _, outlier_idx = torch.topk(outlier_score, k=K_out, dim=-1, largest=True, sorted=False)
            outlier_idx, _ = torch.sort(outlier_idx, dim=-1)  # Preserve chronological order

            idx_exp = outlier_idx.unsqueeze(-1).expand(B, H, K_out, D)
            tier1_k = torch.gather(far_k, dim=2, index=idx_exp)
            tier1_v = torch.gather(far_v, dim=2, index=idx_exp)
            tier1_pos = torch.gather(far_pos, dim=2, index=outlier_idx)
            tier1_surp = torch.gather(far_surp, dim=2, index=outlier_idx)

            is_outlier = torch.zeros((B, H, far_len), dtype=torch.bool, device=keys.device)
            is_outlier.scatter_(-1, outlier_idx, True)
            all_indices = torch.arange(far_len, device=keys.device).view(1, 1, far_len).expand(B, H, far_len)
            rem_idx = all_indices[~is_outlier].view(B, H, rem_len)

            rem_idx_exp = rem_idx.unsqueeze(-1).expand(B, H, rem_len, D)
            rem_k = torch.gather(far_k, dim=2, index=rem_idx_exp)
            rem_v = torch.gather(far_v, dim=2, index=rem_idx_exp)
            rem_pos = torch.gather(far_pos, dim=2, index=rem_idx)
            rem_surp = torch.gather(far_surp, dim=2, index=rem_idx)
        else:
            tier1_k = far_k[:, :, :0, :]
            tier1_v = far_v[:, :, :0, :]
            tier1_pos = far_pos[:, :, :0]
            tier1_surp = far_surp[:, :, :0]
            rem_k, rem_v, rem_pos, rem_surp = far_k, far_v, far_pos, far_surp

        # Tier 2: Vectorized Medoid + Mean-Value Condenser into exact C clusters
        if C > 0 and rem_len > 0:
            rem_norm_v = F.normalize(rem_v.float(), p=2, dim=-1)  # (B, H, rem_len, D)
            seed_positions = torch.linspace(0, rem_len - 1, steps=C, device=keys.device).long()
            seed_v = rem_norm_v[:, :, seed_positions, :]  # (B, H, C, D)

            sim_to_seeds = torch.matmul(rem_norm_v, seed_v.transpose(-2, -1))  # (B, H, rem_len, C)
            # Pin each seed token to its own cluster so every cluster has occupancy >= 1
            c_arange = torch.arange(C, device=keys.device)
            sim_to_seeds[:, :, seed_positions, :] = -1e9
            sim_to_seeds[:, :, seed_positions, c_arange] = 1e9

            cluster_id = sim_to_seeds.argmax(dim=-1)  # (B, H, rem_len) in [0, C-1]
            W = F.one_hot(cluster_id, num_classes=C).float()  # (B, H, rem_len, C)
            counts = W.sum(dim=-2).clamp(min=1.0)  # (B, H, C)

            # Mean-Value pooling per cluster
            tier2_v = (torch.einsum("bhnc,bhnd->bhcd", W, rem_v.float()) / counts.unsqueeze(-1)).to(values.dtype)
            tier2_pos = torch.einsum("bhnc,bhn->bhc", W, rem_pos) / counts
            tier2_surp = torch.einsum("bhnc,bhn->bhc", W, rem_surp) / counts

            # Positional variance Var(P_S) per cluster (scale-normalized so long contexts do not underflow float32)
            pos_scale = max(float(far_len) / 10.0, 1.0)
            norm_rem_pos = rem_pos / pos_scale
            norm_tier2_pos = tier2_pos / pos_scale
            pos_diff_sq = ((norm_rem_pos.unsqueeze(-1) - norm_tier2_pos.unsqueeze(-2)) ** 2) * W
            pos_var = pos_diff_sq.sum(dim=-2) / (counts - 1.0).clamp(min=1.0)
            pos_var = torch.where(counts > 1.0, pos_var, torch.zeros_like(pos_var))

            # Scale-covariant p-adic heat-kernel viscosity gamma = exp(-nu * p^{2*l_eff} * Var(P_S))
            if self.beltrami_mode:
                # Effective shell depth l_eff in [0, 2] scales with cluster compression ratio
                compression_ratio = (counts / max(float(rem_len) / float(max(C, 1)), 1.0)).clamp(min=0.5, max=4.0)
                viscous_scale = float(self.viscous_p) ** torch.log2(compression_ratio)
                gamma = torch.exp(-self.lambda_decay * viscous_scale * pos_var).clamp(min=0.05, max=1.0)
            else:
                gamma = torch.exp(-self.lambda_decay * pos_var).clamp(min=0.05, max=1.0)  # (B, H, C)

            # Medoid Key selection: token in cluster with highest similarity to cluster mean V (ties -> latest)
            center_norm_v = F.normalize(tier2_v.float(), p=2, dim=-1)
            sim_to_center = torch.einsum("bhnd,bhcd->bhnc", rem_norm_v, center_norm_v)
            # Add tiny positional tie-breaker so equally similar tokens prefer the most recent position
            tie_break = (torch.arange(rem_len, device=keys.device, dtype=torch.float32) * 1e-6).view(1, 1, rem_len, 1)
            sim_in_cluster = (sim_to_center + tie_break).masked_fill(W == 0.0, float("-inf"))
            medoid_idx = sim_in_cluster.argmax(dim=-2)  # (B, H, C)

            medoid_idx_exp = medoid_idx.unsqueeze(-1).expand(B, H, C, D)
            medoid_k = torch.gather(rem_k, dim=2, index=medoid_idx_exp)
            tier2_k = (medoid_k.float() * gamma.unsqueeze(-1)).to(keys.dtype)
        else:
            tier2_k = far_k[:, :, :0, :]
            tier2_v = far_v[:, :, :0, :]
            tier2_pos = far_pos[:, :, :0]
            tier2_surp = far_surp[:, :, :0]

        self.key_cache[layer_idx] = torch.cat([tier1_k, tier2_k, local_k], dim=-2)
        self.value_cache[layer_idx] = torch.cat([tier1_v, tier2_v, local_v], dim=-2)
        self.pos_cache[layer_idx] = torch.cat([tier1_pos, tier2_pos, local_pos], dim=-1)
        self.surprisal_cache[layer_idx] = torch.cat([tier1_surp, tier2_surp, local_surp], dim=-1)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])

    def get_physical_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns physical compressed KV-cache sequence length for `layer_idx`."""
        return self.get_seq_length(layer_idx)

    def get_logical_seq_length(self) -> int:
        """Returns total logical tokens seen before condensation."""
        return int(self._true_seen_tokens)


class BeltramiAdelicCache(SurprisalAdelicCache):
    """
    V4 Beltrami Vortex Soliton Adèlic Cache.

    Shields multi-token passcodes ('9481-OMEGA') and low-norm semantic needles
    ('spherical coordinates') using cross-shell Beltrami triad flux rather than
    static Euclidean key norm, and applies scale-covariant p-adic heat-kernel
    dissipation in Tier 2 medoid condensation.
    """

    def __init__(
        self,
        max_capacity: int = 128,
        local_window: int = 64,
        outlier_budget: Optional[int] = None,
        condensed_budget: Optional[int] = None,
        lambda_decay: float = 0.05,
        k_outlier: int = 2,
        viscous_p: int = 2,
    ):
        super().__init__(
            max_capacity=max_capacity,
            local_window=local_window,
            outlier_budget=outlier_budget,
            condensed_budget=condensed_budget,
            lambda_decay=lambda_decay,
            k_outlier=k_outlier,
            beltrami_mode=True,
            viscous_p=viscous_p,
        )


# Backward-compatible alias
TemporalAdelicCache = SurprisalAdelicCache


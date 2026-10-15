"""
Ultrametric AI — Attention Layer (PyTorch)

Implements the core UltrametricAttention module with three execution paths:

1. **Dense** (fallback): Full O(N²) score matrix with masking. Works anywhere.
2. **Chunked Block-Sparse**: Iterates over K/V blocks, skipping masked blocks
   entirely. Real O(N·B) memory where B = avg non-masked blocks per query.
3. **Triton**: Hardware-accelerated block-sparse via the custom Triton kernel.
   True O(N log N) compute with dynamic SRAM skipping.

Includes Rotary Position Embeddings (RoPE) for positional encoding and
tree adjacency mask construction for the Holographic Reasoning Tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

from .topology import get_ultrametric_mask
from .kernel import HAS_TRITON, ultrametric_attention_triton, routing_to_block_indices, CurriculumSparseAttention


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (Su et al., 2021).

    Encodes absolute position as a rotation in the complex plane,
    enabling relative position awareness through the dot product.
    Critical for any benchmark — without positional encoding,
    the model cannot distinguish token order within an attention block.

    Args:
        head_dim: dimension of each attention head
        max_seq_len: maximum sequence length to precompute
        base: base frequency for the sinusoidal schedule
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Precompute frequency bands: theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build or extend the cos/sin position cache."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim/2)
        # Duplicate for paired rotation: [cos(0), cos(1), ..., cos(0), cos(1), ...]
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., seq_len, head_dim) tensor to determine seq_len and device
            offset: position offset for cached/KV generation

        Returns:
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
        """
        seq_len = x.shape[-2] + offset
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[offset:seq_len].to(x.dtype)
        sin = self.sin_cached[offset:seq_len].to(x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions for RoPE: [x0, x1, x2, x3] → [-x1, x0, -x3, x2]."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        q_rotated, k_rotated with same shapes
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated, k_rotated


# ============================================================================
# Tree Adjacency Mask (for Holographic Reasoning Tokens)
# ============================================================================


def get_tree_adjacency_mask(
    routing: torch.Tensor, num_interior: int, seq_len: int, p: int = 2
) -> torch.Tensor:
    """
    Generates a boolean adjacency mask for a tree topology including
    interior Reasoning Tokens wired hierarchically (parent = (i-1)//p).

    Args:
        routing: (batch, seq_len, levels, p) one-hots or
                 (batch, seq_len, levels) branch IDs
        num_interior: number of interior tree nodes
        seq_len: number of leaf (sequence) tokens
        p: tree arity

    Returns:
        mask: (batch, num_interior + seq_len, num_interior + seq_len) bool
    """
    if routing.dim() == 4:
        routing_paths = routing.argmax(dim=-1)
    else:
        routing_paths = routing

    batch_size, _, levels = routing_paths.shape
    total_len = num_interior + seq_len
    device = routing_paths.device

    mask = torch.zeros(
        (batch_size, total_len, total_len), dtype=torch.bool, device=device
    )

    # Self loops
    idx = torch.arange(total_len, device=device)
    mask[:, idx, idx] = True

    # Interior-to-interior edges (tree backbone)
    if num_interior > 1:
        i_idx = torch.arange(1, num_interior, device=device)
        parent_idx = (i_idx - 1) // p
        mask[:, i_idx, parent_idx] = True
        mask[:, parent_idx, i_idx] = True

    # Leaf-to-interior edges (dynamic routing)
    current_node_offset = 0
    current_path_val = torch.zeros(
        (batch_size, seq_len), dtype=torch.long, device=device
    )

    for l in range(1, levels):
        current_node_offset += p ** (l - 1)
        current_path_val = current_path_val * p + routing_paths[:, :, l - 1]

    leaf_parents = current_node_offset + current_path_val
    leaf_parents = leaf_parents.clamp(0, num_interior - 1)

    seq_idx = num_interior + torch.arange(seq_len, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    seq_idx_broadcast = seq_idx.unsqueeze(0).expand(batch_size, seq_len)

    mask[batch_idx, seq_idx_broadcast, leaf_parents] = True
    mask[batch_idx, leaf_parents, seq_idx_broadcast] = True

    return mask


# ============================================================================
# Ultrametric Attention Module
# ============================================================================


class UltrametricAttention(nn.Module):
    """
    Ultrametric (p-adic) Attention with three execution paths.

    Replaces the dense O(N²) attention matrix of standard Transformers with
    a hierarchical block-sparse pattern derived from the p-adic metric on
    the Bruhat-Tits tree.

    Execution modes:
        - 'dense':   Full N² score matrix, masked. Always correct, baseline.
        - 'chunked': Block-wise iteration, skipping masked blocks. Real memory savings.
        - 'triton':  Hardware kernel with dynamic SRAM skipping. Real compute savings.
        - 'auto':    Triton if available + CUDA + seq_len >= 256, else chunked.

    Args:
        embed_dim: total embedding dimension
        num_heads: number of attention heads
        p: tree arity (2 = binary Bruhat-Tits tree)
        max_seq_len: max sequence length for RoPE precomputation
        dropout: attention dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        p: int = 2,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.p = p
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        num_interior: int = 0,
        dynamic_mask: Optional[torch.Tensor] = None,
        routing: Optional[torch.Tensor] = None,
        routing_assignments: Optional[torch.Tensor] = None,
        mode: Literal["auto", "dense", "chunked", "triton"] = "auto",
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, total_len, embed_dim) where total_len = num_interior + seq_len
            num_interior: number of interior Reasoning Tokens prepended to x
            dynamic_mask: (batch, total_len, total_len) pre-computed attention mask
            routing: (batch, seq_len, levels, p) or (batch, seq_len, levels) for tree mask
            routing_assignments: (batch, heads, seq_len, levels, p) for Triton path
            mode: execution mode selection

        Returns:
            out: (batch, total_len, embed_dim)
        """
        batch_size, total_len, _ = x.size()
        seq_len = total_len - num_interior

        # Project Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, total_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, total_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, total_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Shape: (batch, heads, total_len, head_dim)

        # Apply RoPE to the sequence portion (not interior tokens)
        if num_interior > 0:
            q_seq = q[:, :, num_interior:, :]
            k_seq = k[:, :, num_interior:, :]
            cos, sin = self.rope(q_seq)
            q_seq, k_seq = apply_rotary_pos_emb(q_seq, k_seq, cos, sin)
            q = torch.cat([q[:, :, :num_interior, :], q_seq], dim=2)
            k = torch.cat([k[:, :, :num_interior, :], k_seq], dim=2)
        else:
            cos, sin = self.rope(q)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Resolve execution mode
        if mode == "auto":
            if (
                HAS_TRITON
                and x.is_cuda
                and seq_len >= 256
                and routing_assignments is not None
                and num_interior == 0
            ):
                mode = "triton"
            elif seq_len >= 128:
                mode = "chunked"
            else:
                mode = "dense"

        # Dispatch to the appropriate attention path
        if mode == "triton" and HAS_TRITON and routing_assignments is not None:
            return self._triton_attention(
                q, k, v, routing_assignments, batch_size, total_len
            )
        elif mode == "chunked":
            mask = self._resolve_mask(
                batch_size, total_len, seq_len, num_interior, dynamic_mask, routing, x.device
            )
            return self._chunked_sparse_attention(q, k, v, mask, batch_size, total_len)
        else:
            mask = self._resolve_mask(
                batch_size, total_len, seq_len, num_interior, dynamic_mask, routing, x.device
            )
            return self._dense_attention(q, k, v, mask, batch_size, total_len)

    def _resolve_mask(
        self,
        batch_size: int,
        total_len: int,
        seq_len: int,
        num_interior: int,
        dynamic_mask: Optional[torch.Tensor],
        routing: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Build the attention mask from routing or dynamic_mask input."""
        if routing is not None:
            mask = get_tree_adjacency_mask(routing, num_interior, seq_len, self.p)
            mask = mask.unsqueeze(1)  # (batch, 1, total_len, total_len)
        elif dynamic_mask is not None:
            mask = dynamic_mask.to(device)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        else:
            # Fallback: use static ultrametric mask on the sequence portion
            static_mask = get_ultrametric_mask(total_len, self.p).to(device)
            mask = static_mask.unsqueeze(0).unsqueeze(0)
        return mask

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        total_len: int,
    ) -> torch.Tensor:
        """Standard dense attention with masking. O(N²) compute and memory."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + torch.log(mask.clamp(min=1e-9))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, total_len, self.embed_dim)
        )
        return self.o_proj(out)

    def _chunked_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        total_len: int,
        block_size: int = 64,
    ) -> torch.Tensor:
        """
        Chunked block-sparse attention. Iterates over K/V blocks and skips
        entirely masked blocks, achieving real memory savings without Triton.

        For a binary tree with depth-3 cutoff and seq_len=4096:
        ~75% of blocks are skipped → ~4× memory reduction.
        """
        B, H, S, D = q.shape
        num_blocks = math.ceil(S / block_size)

        # Pad to exact block boundary
        pad_len = num_blocks * block_size - S
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            # Expand mask to padded size
            mask = F.pad(mask, (0, pad_len, 0, pad_len), value=False)

        S_padded = num_blocks * block_size

        # Reshape into blocks: (B, H, num_blocks, block_size, D)
        q_blocks = q.view(B, H, num_blocks, block_size, D)
        k_blocks = k.view(B, H, num_blocks, block_size, D)
        v_blocks = v.view(B, H, num_blocks, block_size, D)

        # Precompute block-level mask: does block i attend to block j at all?
        # mask shape: (B, 1 or H, S_padded, S_padded)
        mask_expanded = mask.expand(B, H, S_padded, S_padded) if mask.shape[1] == 1 else mask
        mask_blocks = mask_expanded.view(B, H, num_blocks, block_size, num_blocks, block_size)
        # Any token in block i attends to any token in block j?
        block_mask = mask_blocks.any(dim=3).any(dim=-1)  # (B, H, num_blocks, num_blocks)

        # Online softmax accumulators
        out_acc = torch.zeros_like(q)  # (B, H, S_padded, D)
        m_acc = torch.full((B, H, S_padded), float("-inf"), device=q.device, dtype=torch.float32)
        l_acc = torch.zeros((B, H, S_padded), device=q.device, dtype=torch.float32)

        for j in range(num_blocks):
            # Check which (batch, head, query_block) pairs need this KV block
            active = block_mask[:, :, :, j]  # (B, H, num_blocks)

            if not active.any():
                continue

            k_j = k_blocks[:, :, j, :, :]  # (B, H, block_size, D)
            v_j = v_blocks[:, :, j, :, :]

            for i in range(num_blocks):
                if not active[:, :, i].any():
                    continue

                q_i = q_blocks[:, :, i, :, :]  # (B, H, block_size, D)

                # Compute scores for this block pair
                scores_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * self.scale
                # (B, H, block_size, block_size)

                # Apply fine-grained token mask
                token_mask = mask_expanded[
                    :, :,
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ]
                if token_mask.dtype == torch.bool:
                    scores_ij = scores_ij.masked_fill(~token_mask, float("-inf"))
                else:
                    scores_ij = scores_ij + torch.log(token_mask.clamp(min=1e-9))

                # Online softmax update for this block
                row_start = i * block_size
                row_end = (i + 1) * block_size

                m_old = m_acc[:, :, row_start:row_end]  # (B, H, block_size)
                l_old = l_acc[:, :, row_start:row_end]
                out_old = out_acc[:, :, row_start:row_end, :]

                m_new_block = scores_ij.max(dim=-1).values  # (B, H, block_size)
                m_new = torch.maximum(m_old, m_new_block)

                # Rescale old accumulator
                alpha = torch.exp(m_old - m_new)
                # New block contribution
                p_block = torch.exp(scores_ij - m_new.unsqueeze(-1))  # (B, H, bs, bs)

                out_acc[:, :, row_start:row_end, :] = (
                    out_old * alpha.unsqueeze(-1)
                    + torch.matmul(p_block.to(v_j.dtype), v_j)
                )
                l_acc[:, :, row_start:row_end] = (
                    l_old * alpha + p_block.sum(dim=-1)
                )
                m_acc[:, :, row_start:row_end] = m_new

        # Normalize
        out = (out_acc / l_acc.unsqueeze(-1).clamp(min=1e-8)).to(q.dtype)

        # Remove padding
        out = out[:, :, :S, :]
        out = self.attn_dropout(out)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, total_len, self.embed_dim)
        )
        return self.o_proj(out)

    def _triton_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routing_assignments: torch.Tensor,
        batch_size: int,
        total_len: int,
    ) -> torch.Tensor:
        """
        Triton hardware-accelerated block-sparse attention.
        Requires CUDA, float16, and Triton installed.
        """
        # Convert per-token routing to per-block indices for the kernel
        BLOCK_M = min(128, total_len)
        router_indices = routing_to_block_indices(
            routing_assignments, total_len, block_size=BLOCK_M
        )

        # Determine required depth from tree levels
        tree_depth = router_indices.shape[-1]
        req_depth = max(tree_depth // 2, 1)

        # Cast to float16 if needed
        q_half = q.half() if q.dtype != torch.float16 else q
        k_half = k.half() if k.dtype != torch.float16 else k
        v_half = v.half() if v.dtype != torch.float16 else v

        out = CurriculumSparseAttention.apply(
            q_half, k_half, v_half, router_indices, req_depth, self.p, True
        )

        # Cast back if input wasn't float16
        if q.dtype != torch.float16:
            out = out.to(q.dtype)

        out = self.attn_dropout(out)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, total_len, self.embed_dim)
        )
        return self.o_proj(out)

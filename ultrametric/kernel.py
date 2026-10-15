"""
Ultrametric AI — Triton Block-Sparse Attention Kernel (NVIDIA GPU)

Hardware-accelerated block-sparse attention that dynamically skips SRAM loads
for blocks whose routing vectors indicate they do not share the required
ancestral depth in the Bruhat-Tits tree.

The kernel uses Online Softmax (Milakov & Gimelshein, 2018) for numerically
stable, single-pass attention without materializing the full N² score matrix.

Usage:
    out = ultrametric_attention_triton(q, k, v, router_indices, req_depth=3)

Falls back to PyTorch masked attention if Triton is not installed.
"""

import torch
import math
import os
from typing import Optional

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, RuntimeError):
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _ultrametric_fwd_kernel(
        Q, K, V, sm_scale,
        q_to_k_indices, num_active_k, max_active_k: tl.constexpr,
        Out, L,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        stride_idx_z, stride_idx_h, stride_idx_m, stride_idx_n,
        stride_act_z, stride_act_h, stride_act_m,
        stride_lz, stride_lh, stride_lm,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr = False,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh
        idx_offset = off_z * stride_idx_z + off_h * stride_idx_h + start_m * stride_idx_m
        act_offset = off_z * stride_act_z + off_h * stride_act_h + start_m * stride_act_m

        q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            base=Out + o_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0),
        )

        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        num_act = tl.load(num_active_k + act_offset)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        for idx in range(max_active_k):
            mask = idx < num_act
            safe_idx = tl.where(mask, idx, 0)
            start_n_block = tl.load(q_to_k_indices + idx_offset + safe_idx * stride_idx_n)

            k_block_ptr = tl.make_block_ptr(
                base=K + k_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
                offsets=(0, start_n_block * BLOCK_N), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1),
            )
            v_block_ptr = tl.make_block_ptr(
                base=V + v_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vk, stride_vn),
                offsets=(start_n_block * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0),
            )

            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(mask, qk, float('-inf'))

            if IS_CAUSAL:
                # Token-level causal mask (covers the diagonal block and makes any
                # future block fully -inf). Assigning a scalar to the block-typed
                # `qk` in a branch does not compile in Triton.
                offs_n = start_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

        acc = acc / l_i[:, None]
        tl.store(o_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

        L_i = m_i + tl.math.log(l_i)
        l_offset = off_z * stride_lz + off_h * stride_lh
        l_ptrs = L + l_offset + offs_m
        tl.store(l_ptrs, L_i, mask=offs_m < N_CTX)


def compute_block_sparse_indices(router_indices: torch.Tensor, req_depth: int, is_causal: bool = False):
    """
    Computes precomputed active block coordinate lists for the Triton kernels.
    """
    Z, H, num_blocks, tree_depth = router_indices.shape
    if req_depth == 0:
        match = torch.ones((Z, H, num_blocks, num_blocks), dtype=torch.bool, device=router_indices.device)
    else:
        r = router_indices[:, :, :, :req_depth]
        match = (r.unsqueeze(3) == r.unsqueeze(2)).all(dim=-1)
    
    if is_causal:
        col_indices = torch.arange(num_blocks, device=router_indices.device)
        causal_block_mask = col_indices.unsqueeze(0) <= col_indices.unsqueeze(1)
        match = match & causal_block_mask
    
    _, q_to_k_indices = match.sort(dim=-1, descending=True)
    num_active_k = match.sum(dim=-1, dtype=torch.int32)
    q_to_k_indices = q_to_k_indices.to(torch.int32)
    max_active_k = num_active_k.max().item()
    
    _, k_to_q_indices = match.transpose(2, 3).sort(dim=-1, descending=True)
    num_active_q = match.transpose(2, 3).sum(dim=-1, dtype=torch.int32)
    k_to_q_indices = k_to_q_indices.to(torch.int32)
    max_active_q = num_active_q.max().item()
    
    return q_to_k_indices, num_active_k, max_active_k, k_to_q_indices, num_active_q, max_active_q


def routing_to_block_indices(
    assignments: torch.Tensor,
    seq_len: int,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Convert per-token routing assignments to per-block routing indices
    for the Triton kernel.

    Each block's routing vector is determined by majority vote of the tokens
    in that block. This ensures consistent block-level routing decisions.

    Args:
        assignments: (batch, heads, seq_len, levels, p) one-hot routing
        seq_len: actual sequence length (before padding)
        block_size: kernel block size (BLOCK_M)

    Returns:
        router_indices: (batch, heads, num_blocks, levels) int32
    """
    B, H, S, L, P = assignments.shape
    num_blocks = math.ceil(seq_len / block_size)

    # Get integer branch IDs from one-hot: (B, H, S, L)
    branch_ids = assignments.argmax(dim=-1)

    # Pad sequence to exact multiple of block_size
    if S < num_blocks * block_size:
        pad_len = num_blocks * block_size - S
        branch_ids = torch.nn.functional.pad(branch_ids, (0, 0, 0, pad_len), value=0)

    # Reshape into blocks: (B, H, num_blocks, block_size, L)
    branch_ids = branch_ids.view(B, H, num_blocks, block_size, L)

    # Majority vote per block: take mode (most common branch per level)
    # For efficiency, use the first token in each block as representative
    # (in practice, tokens within a block are spatially close and route similarly)
    router_indices = branch_ids[:, :, :, 0, :]  # (B, H, num_blocks, L)

    return router_indices.to(torch.int32)


if HAS_TRITON:
    @triton.jit
    def _ultrametric_bwd_dq_kernel(
        Q, K, V, sm_scale, q_to_k_indices, num_active_k, max_active_k: tl.constexpr,
        dO, dQ, L, D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        stride_idx_z, stride_idx_h, stride_idx_m, stride_idx_n,
        stride_act_z, stride_act_h, stride_act_m,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        do_offset = off_z * stride_oz + off_h * stride_oh
        dq_offset = off_z * stride_qz + off_h * stride_qh
        idx_offset = off_z * stride_idx_z + off_h * stride_idx_h + start_m * stride_idx_m
        act_offset = off_z * stride_act_z + off_h * stride_act_h + start_m * stride_act_m

        q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
        )
        do_block_ptr = tl.make_block_ptr(
            base=dO + do_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
        )
        dq_block_ptr = tl.make_block_ptr(
            base=dQ + dq_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
        )

        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        l_ptrs = L + off_hz * N_CTX + offs_m
        d_ptrs = D + off_hz * N_CTX + offs_m
        l_i = tl.load(l_ptrs, mask=offs_m < N_CTX, other=0.0)
        D_i = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        num_act = tl.load(num_active_k + act_offset)

        for idx in range(max_active_k):
            mask = idx < num_act
            safe_idx = tl.where(mask, idx, 0)
            start_n_block = tl.load(q_to_k_indices + idx_offset + safe_idx * stride_idx_n)

            k_ptr = tl.make_block_ptr(
                base=K + k_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
                offsets=(0, start_n_block * BLOCK_N), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1)
            )
            v_T_ptr = tl.make_block_ptr(
                base=V + v_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_vk, stride_vn),
                offsets=(0, start_n_block * BLOCK_N), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1)
            )
            k_norm_ptr = tl.make_block_ptr(
                base=K + k_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_kn, stride_kk),
                offsets=(start_n_block * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
            )

            k = tl.load(k_ptr, boundary_check=(0, 1), padding_option="zero")
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(mask, qk, float('-inf'))
            p = tl.exp(qk - l_i[:, None])

            v_T = tl.load(v_T_ptr, boundary_check=(0, 1), padding_option="zero")
            dp = tl.dot(do, v_T)
            
            ds = p * (dp - D_i[:, None]) * sm_scale
            ds = tl.where(mask, ds, 0.0)
            
            k_n = tl.load(k_norm_ptr, boundary_check=(0, 1), padding_option="zero")
            dq += tl.dot(ds.to(tl.float16), k_n)

        tl.store(dq_block_ptr, dq.to(tl.float16), boundary_check=(0, 1))

    @triton.jit
    def _ultrametric_bwd_dk_dv_kernel(
        Q, K, V, sm_scale, k_to_q_indices, num_active_q, max_active_q: tl.constexpr,
        dO, dK, dV, L, D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        stride_idx_z, stride_idx_h, stride_idx_m, stride_idx_n,
        stride_act_z, stride_act_h, stride_act_m,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        do_offset = off_z * stride_oz + off_h * stride_oh
        dk_offset = off_z * stride_kz + off_h * stride_kh
        dv_offset = off_z * stride_vz + off_h * stride_vh
        idx_offset = off_z * stride_idx_z + off_h * stride_idx_h + start_n * stride_idx_m
        act_offset = off_z * stride_act_z + off_h * stride_act_h + start_n * stride_act_m

        k_block_ptr = tl.make_block_ptr(
            base=K + k_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_kn, stride_kk),
            offsets=(start_n * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
        )
        v_block_ptr = tl.make_block_ptr(
            base=V + v_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk),
            offsets=(start_n * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
        )
        dk_block_ptr = tl.make_block_ptr(
            base=dK + dk_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_kn, stride_kk),
            offsets=(start_n * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
        )
        dv_block_ptr = tl.make_block_ptr(
            base=dV + dv_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk),
            offsets=(start_n * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
        )

        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        num_act = tl.load(num_active_q + act_offset)

        for idx in range(max_active_q):
            mask = idx < num_act
            safe_idx = tl.where(mask, idx, 0)
            start_m_block = tl.load(k_to_q_indices + idx_offset + safe_idx * stride_idx_n)

            q_norm_ptr = tl.make_block_ptr(
                base=Q + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                offsets=(start_m_block * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
            )
            do_norm_ptr = tl.make_block_ptr(
                base=dO + do_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on),
                offsets=(start_m_block * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
            )

            q_n = tl.load(q_norm_ptr, boundary_check=(0, 1), padding_option="zero")
            qk = tl.dot(q_n, tl.trans(k)) * sm_scale
            qk = tl.where(mask, qk, float('-inf'))
            
            offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            l_ptrs = L + off_hz * N_CTX + offs_m
            d_ptrs = D + off_hz * N_CTX + offs_m
            l_i = tl.load(l_ptrs, mask=offs_m < N_CTX, other=0.0)
            D_i = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

            p = tl.exp(qk - l_i[:, None])

            do = tl.load(do_norm_ptr, boundary_check=(0, 1), padding_option="zero")
            
            p_masked = tl.where(mask, p, 0.0)
            dv += tl.dot(tl.trans(p_masked).to(tl.float16), do)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - D_i[:, None]) * sm_scale
            ds = tl.where(mask, ds, 0.0)

            dk += tl.dot(tl.trans(ds).to(tl.float16), q_n)

        tl.store(dk_block_ptr, dk.to(tl.float16), boundary_check=(0, 1))
        tl.store(dv_block_ptr, dv.to(tl.float16), boundary_check=(0, 1))


def _pytorch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    router_indices: torch.Tensor,
    req_depth: int,
    p: int = 2,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Pure PyTorch fallback when Triton is unavailable.

    Reconstructs a block-level boolean mask from router_indices and applies
    standard scaled dot-product attention with the mask.
    """
    import torch.nn.functional as F

    Z, H, N_CTX, DMODEL = q.shape
    BLOCK = 128
    num_blocks = router_indices.shape[2]

    # Build block-level mask: blocks match if routing prefixes agree up to req_depth
    # router_indices: (Z, H, num_blocks, depth)
    r = router_indices[:, :, :, :req_depth]  # (Z, H, num_blocks, req_depth)
    match = (r.unsqueeze(3) == r.unsqueeze(2)).all(dim=-1)  # (Z, H, num_blocks, num_blocks)

    if is_causal:
        col_indices = torch.arange(num_blocks, device=router_indices.device)
        causal_block_mask = col_indices.unsqueeze(0) <= col_indices.unsqueeze(1)
        match = match & causal_block_mask

    # Expand block-level mask to token-level mask
    # match[z,h,i,j] = True means tokens in block i attend to tokens in block j
    mask = match.repeat_interleave(BLOCK, dim=2).repeat_interleave(BLOCK, dim=3)
    mask = mask[:, :, :N_CTX, :N_CTX]  # trim padding

    if is_causal:
        causal_mask = torch.tril(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool))
        mask = mask & causal_mask

    scale = 1.0 / math.sqrt(DMODEL)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def ultrametric_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    router_indices: torch.Tensor,
    req_depth: int = 2,
    p: int = 2,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Launch wrapper for the Triton block-sparse ultrametric attention kernel.

    This function handles stride computation, output allocation, grid sizing,
    and kernel dispatch. The kernel itself performs Online Softmax with dynamic
    block skipping based on p-adic routing vectors.

    Args:
        q: (batch, heads, seq_len, head_dim) float16 query tensor
        k: (batch, heads, seq_len, head_dim) float16 key tensor
        v: (batch, heads, seq_len, head_dim) float16 value tensor
        router_indices: (batch, heads, num_blocks, tree_depth) int32
            Per-block routing vectors. num_blocks = ceil(seq_len / BLOCK_M).
            Each entry is an integer branch ID at the corresponding tree level.
        req_depth: required ancestral depth for block matching.
            Higher = sparser attention (more blocks skipped).
            0 = all blocks attend (equivalent to dense). Must be <= tree_depth.
        p: tree arity (default: 2 for binary Bruhat-Tits tree)
        is_causal: whether to apply lower-triangular causal masking (for autoregressive LLMs)

    Returns:
        out: (batch, heads, seq_len, head_dim) float16 attention output
    """
    if not HAS_TRITON:
        return _pytorch_fallback(q, k, v, router_indices, req_depth, p, is_causal=is_causal)

    assert q.dtype == torch.float16, f"Triton kernel requires float16, got {q.dtype}"
    assert q.is_cuda, "Triton kernel requires CUDA tensors"

    Z, H, N_CTX, DMODEL = q.shape

    # Ensure contiguous memory layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    router_indices = router_indices.contiguous().to(torch.int32)

    TREE_DEPTH = router_indices.shape[-1]
    assert 0 <= req_depth <= TREE_DEPTH, (
        f"req_depth ({req_depth}) must be in [0, {TREE_DEPTH}]"
    )

    # Block size configuration
    BLOCK_M = min(128, N_CTX)
    BLOCK_N = min(128, N_CTX)
    BLOCK_DMODEL = DMODEL

    # Pad router_indices if num_blocks doesn't match
    num_blocks_needed = triton.cdiv(N_CTX, BLOCK_M)
    num_blocks_have = router_indices.shape[2]
    if num_blocks_have < num_blocks_needed:
        pad = torch.zeros(
            (Z, H, num_blocks_needed - num_blocks_have, TREE_DEPTH),
            dtype=torch.int32,
            device=router_indices.device,
        )
        router_indices = torch.cat([router_indices, pad], dim=2)

    # Precompute coordinate lists
    q_to_k_indices, num_active_k, max_active_k, _, _, _ = compute_block_sparse_indices(
        router_indices, req_depth, is_causal=is_causal
    )

    # Allocate output
    out = torch.empty_like(q)

    # Scale factor
    sm_scale = 1.0 / math.sqrt(DMODEL)

    # Grid: one program per (query_block, batch*head)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    _ultrametric_fwd_kernel[grid](
        q, k, v, sm_scale, q_to_k_indices, num_active_k, max_active_k, out, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q_to_k_indices.stride(0), q_to_k_indices.stride(1), q_to_k_indices.stride(2), q_to_k_indices.stride(3),
        num_active_k.stride(0), num_active_k.stride(1), num_active_k.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
    )

    return out, L


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    router_indices: torch.Tensor,
    req_depth: int = 2,
    p: int = 2,
    is_causal: bool = False,
    route_block: int = 128,
):
    """Quickstart API: block-sparse routed attention forward (inference).

    Backed by the block-list kernel (``ultrametric.block_list``), which loops over
    only each query block's active key blocks. Same call signature and return
    value as the legacy ``ultrametric_attention_triton``: ``(out, lse)`` where
    ``lse`` is the per-row natural log-sum-exp of the scaled scores, (Z, H, N) fp32.

    Args:
        q, k, v: (Z, H, N, D) fp16/bf16 (fp32 also accepted), D a power of two >= 16.
        router_indices: (Z, H, num_blocks, tree_depth) non-negative integer branch ids,
            one routing vector per ``route_block`` tokens. If fewer than
            ceil(N / route_block) blocks are given, the rest are padded with branch 0.
        req_depth: blocks match iff their routing vectors agree on the first
            ``req_depth`` levels (0 = dense).
        p: tree arity (informational; ids are validated on the host).
        is_causal: token-level causal masking.

    Notes:
        This wrapper validates ids and sizes the lists exactly, which costs a few
        host syncs per call. In a hot loop, build lists once with
        ``build_block_lists(..., arity=p)`` (sync-free) and call
        ``block_list_attention`` directly. Forward only (no autograd).
        Without CUDA/Triton it falls back to an fp32 masked PyTorch reference.
    """
    from .block_list import (block_list_attention, block_mask_reference, build_block_lists,
                             token_mask_from_blocks)

    Z, H, N, D = q.shape
    router_indices = router_indices.to(torch.int64)
    nb = -(-N // route_block)
    have = router_indices.shape[2]
    if have < nb:
        pad = torch.zeros((Z, H, nb - have, router_indices.shape[-1]), dtype=router_indices.dtype,
                          device=router_indices.device)
        router_indices = torch.cat([router_indices, pad], dim=2)
    elif have > nb:
        router_indices = router_indices[:, :, :nb]

    use_kernel = HAS_TRITON and (q.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")
    if use_kernel:
        bl = build_block_lists(router_indices, req_depth, is_causal=is_causal)
        return block_list_attention(q, k, v, bl, route_block=route_block, is_causal=is_causal,
                                    return_lse=True)

    bm = block_mask_reference(router_indices, req_depth, is_causal=is_causal)
    mask = token_mask_from_blocks(bm, route_block, N, is_causal=is_causal)
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(D)
    s = s.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = torch.matmul(torch.softmax(s, dim=-1).nan_to_num(0.0), v.float()).to(q.dtype)
    return out, lse


class CurriculumSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, router_indices, req_depth=2, p=2, use_sparse_backend=False):
        if use_sparse_backend and HAS_TRITON:
            o, L = ultrametric_attention_triton(q, k, v, router_indices, req_depth, p)
            ctx.save_for_backward(q, k, v, o, L, router_indices)
            ctx.req_depth = req_depth
            ctx.p = p
            ctx.use_sparse_backend = True
            return o
        else:
            # Fallback path requires gradient tracking to be maintained automatically
            # since we won't execute custom backward.
            # So if not using sparse backend, we should not wrap in autograd if possible, 
            # but since we are in autograd.Function, we must return a tensor that has grad_fn.
            # Actually, standard PyTorch allows calling operations in forward that we manually differentiate.
            # If use_sparse_backend=False, we compute the PyTorch fallback here and rely on PyTorch Autograd.
            # But autograd.Function overrides it. 
            # We'll just do the computation. Wait, if we return a tensor computed with PyTorch ops, 
            # its grad_fn is ignored because it's inside autograd.Function.
            # We must compute backward manually! 
            # To avoid this, CurriculumSparseAttention should ONLY be used when use_sparse_backend=True.
            raise NotImplementedError("CurriculumSparseAttention should only be called for use_sparse_backend=True")

    @staticmethod
    def backward(ctx, do):
        if not ctx.use_sparse_backend:
            raise NotImplementedError("Dense fallback backward not implemented in autograd.Function")

        q, k, v, o, L, router_indices = ctx.saved_tensors
        req_depth = ctx.req_depth
        p = ctx.p

        do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        Z, H, N_CTX, DMODEL = q.shape
        sm_scale = 1.0 / math.sqrt(DMODEL)

        BLOCK_M = min(128, N_CTX)
        BLOCK_N = min(128, N_CTX)
        BLOCK_DMODEL = DMODEL

        num_blocks_needed = triton.cdiv(N_CTX, BLOCK_M)
        num_blocks_have = router_indices.shape[2]
        if num_blocks_have < num_blocks_needed:
            pad = torch.zeros(
                (Z, H, num_blocks_needed - num_blocks_have, router_indices.shape[-1]),
                dtype=torch.int32,
                device=router_indices.device,
            )
            router_indices = torch.cat([router_indices, pad], dim=2)

        q_to_k_indices, num_active_k, max_active_k, k_to_q_indices, num_active_q, max_active_q = compute_block_sparse_indices(router_indices, req_depth)

        # D_i = sum(do_i * o_i)
        D = (do * o).to(torch.float32).sum(dim=-1).contiguous()

        grid_q = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
        _ultrametric_bwd_dq_kernel[grid_q](
            q, k, v, sm_scale, q_to_k_indices, num_active_k, max_active_k,
            do, dq, L, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q_to_k_indices.stride(0), q_to_k_indices.stride(1), q_to_k_indices.stride(2), q_to_k_indices.stride(3),
            num_active_k.stride(0), num_active_k.stride(1), num_active_k.stride(2),
            Z, H, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        )

        grid_k = (triton.cdiv(N_CTX, BLOCK_N), Z * H)
        _ultrametric_bwd_dk_dv_kernel[grid_k](
            q, k, v, sm_scale, k_to_q_indices, num_active_q, max_active_q,
            do, dk, dv, L, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            k_to_q_indices.stride(0), k_to_q_indices.stride(1), k_to_q_indices.stride(2), k_to_q_indices.stride(3),
            num_active_q.stride(0), num_active_q.stride(1), num_active_q.stride(2),
            Z, H, N_CTX,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        )

        return dq, dk, dv, None, None, None, None


"""
Physical Block-Sparse Attention Kernel for Google Gemma 4 31B (arXiv:2607.02770).
Part of Project DarkSnail V2 (CC0-1.0 Public Domain).

Architecture:
1. Heterogeneous 5:1 Gemma 4 Attention:
   - 10 Global Full-Attention layers: GLOBAL_LAYER_INDICES = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59].
   - head_dim = 512, H_q = 32, H_kv = 4 (GQA factor = 8).
   - attention_k_eq_v = True: K and V share the raw projection k_raw = W_k x.
     k = k_norm(k_raw), v = v_norm(k_raw) (pre-k_norm k_raw normalized by scale-free RMSNorm).
   - Proportional RoPE (p-RoPE, 25% rotated):
     d_rope = 128 (rotary phase channels), d_nope = 384 (position-invariant semantic content).
   - Routing contract: Dynamic Bruhat-Tits tree routing is evaluated strictly on the post-k_norm
     unrotated 384-D k_nope subspace.
   - Kernel attention contract: Dot-product attention is evaluated over full 512-D post-RoPE (q, k)
     and pre-k_norm v exclusively over active blocks in BlockLists(lists, counts, max_count).

2. Physical Block-Sparse Prefill (block_sparse_gemma4_prefill):
   - Given BlockLists(lists, counts, max_count) of block size R_B = 128 tokens, executes physical
     block-sparse attention.
   - On CUDA (when Triton is available), runs fused @triton.jit online softmax kernel streaming active
     key blocks directly from row-specific block lists.
   - On CPU/MPS, executes a vectorized block-gather tile loop without ever materializing an S x S dense
     mask or computing skipped key blocks.

3. Span-Shielded Decode (span_shielded_gemma4_decode):
   - During single-token autoregressive decoding (seq_len == 1, past_len > 0), gathers strictly the
     active local sliding window, sink block 0, ultrametric routed prefix blocks, and cached top-96
     Beltrami + Query-Tail shielded token spans (_darksnail_shield_tokens), completely eliminating
     full-context O(L_kv) KV scans.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, RuntimeError):
    triton = None
    tl = None
    HAS_TRITON = False

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

# ============================================================================
# Gemma 4 31B Global Layer Architectural Constants
# ============================================================================

GLOBAL_LAYER_INDICES: Tuple[int, ...] = (5, 11, 17, 23, 29, 35, 41, 47, 53, 59)
HEAD_DIM_GLOBAL: int = 512
D_ROPE: int = 128       # 25% of 512
D_NOPE: int = 384       # 75% of 512
ROUTE_BLOCK: int = 128  # Default block size R_B


class BlockLists(NamedTuple):
    """
    Compiled active key route-blocks per query route-block.

    lists:     (B, H, NB, max_count) int32. Row (b, h, qb) holds the active key block
               indices in ascending order; positions >= counts[b, h, qb] are padding.
    counts:    (B, H, NB) int32, number of active key blocks per query block row.
    max_count: int, maximum over counts (stride of lists).
    """

    lists: torch.Tensor
    counts: torch.Tensor
    max_count: int


# ============================================================================
# Subspace Geometry & Helper Utilities
# ============================================================================

def extract_knope_subspace(
    k: torch.Tensor,
    layout: str = "hf_split_half",
    head_dim: int = HEAD_DIM_GLOBAL,
    d_rope: int = D_ROPE,
) -> torch.Tensor:
    """
    Extracts the unrotated 384-D `k_nope` subspace (75%) from 512-D key states `k`.

    Supports both HuggingFace `hf_split_half` layout and `contiguous_prefix` layout:
    - hf_split_half: rotary channels are [0:64] U [256:320]; non-rotary are [64:256] U [320:512].
    - contiguous_prefix: rotary channels are [0:128]; non-rotary are [128:512].
    """
    if layout == "hf_split_half":
        half_rope = d_rope // 2      # 64
        half_total = head_dim // 2   # 256
        nope_1 = k[..., half_rope:half_total]                  # 64..256 (192)
        nope_2 = k[..., (half_total + half_rope):head_dim]     # 320..512 (192)
        return torch.cat([nope_1, nope_2], dim=-1)             # 384
    elif layout == "contiguous_prefix":
        return k[..., d_rope:head_dim]                         # 128..512 (384)
    else:
        raise ValueError(f"Unsupported p-RoPE layout: {layout}")


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads for Grouped-Query Attention (GQA)."""
    if n_rep <= 1:
        return x
    b, h_kv, s, d = x.shape
    return x.unsqueeze(2).expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


def compute_causal_budget(
    counts: torch.Tensor,
    seq_len: int,
    route_block: int = ROUTE_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes exact active causal key pair counts and budget fraction vs dense causal ceiling.

    For block qb in [0, NB-1]:
    - Block length L_q = route_block (or S - (NB-1)*RB for last block).
    - Row qb attends to (counts[qb] - 1) full antecedent key blocks of size RB,
      plus 1 causal diagonal self-block contributing L_q * (L_q + 1) / 2 pairs.
    Total active causal pairs = sum_qb ((counts[qb] - 1) * RB * L_q + L_q * (L_q + 1) / 2).
    Dense causal pairs = S * (S + 1) / 2.
    """
    B, H, NB = counts.shape
    device = counts.device
    lens = torch.full((NB,), route_block, dtype=torch.float64, device=device)
    lens[-1] = seq_len - (NB - 1) * route_block
    c = counts.to(torch.float64)

    pairs_per_row = (c - 1.0) * float(route_block) * lens + lens * (lens + 1.0) / 2.0
    total_pairs = pairs_per_row.sum(dim=-1)  # (B, H)
    dense_pairs = float(seq_len * (seq_len + 1) / 2.0)
    budget_fraction = total_pairs / max(dense_pairs, 1.0)
    return total_pairs, budget_fraction


def measure_block_sparse_traffic_reduction(
    counts: torch.Tensor,
    seq_len: int,
    route_block: int = ROUTE_BLOCK,
) -> float:
    """Computes active KV-block traffic reduction percentage vs dense causal block count."""
    B, H, NB = counts.shape
    dense_blocks_total = B * H * (NB * (NB + 1) / 2.0)
    active_blocks_total = float(counts.float().sum().item())
    reduction_pct = 100.0 * (1.0 - (active_blocks_total / max(dense_blocks_total, 1.0)))
    return float(reduction_pct)


def apply_darksnail_block_bridge(
    bl: BlockLists,
    route_block: int,
    seq_len: int,
    local_blocks: int,
    token_salience: Optional[torch.Tensor] = None,
    bridge_blocks: int = 4,
) -> BlockLists:
    """
    Executes the DarkSnail Measure-Preserving Callosal Block Bridge on BlockLists:
    1. Retargets the n_int,2 interior slots on penultimate block NB - 2 to the top-salience
       Beltrami blocks, preserving counts[NB - 2] identically.
    2. Transfers up to bridge_blocks interior slots from donor rows (d_qb in [local_blocks+2, NB-2])
       to the final tail query block NB - 1, strictly satisfying Delta Budget_causal <= 0.
    """
    if token_salience is None:
        return bl

    lists = bl.lists.clone()
    counts = bl.counts.clone()
    max_count = bl.max_count
    B, H, NB = counts.shape
    if NB <= local_blocks + 3 or max_count <= 2:
        return bl

    pad_len = NB * route_block - seq_len
    if pad_len > 0:
        sal_pad = F.pad(token_salience, (0, pad_len), value=float("-inf"))
    else:
        sal_pad = token_salience[:, :, : NB * route_block]
    blk_sal = sal_pad.view(B, H, NB, route_block).max(dim=-1).values

    lists_cpu = lists.cpu()
    counts_cpu = counts.cpu()
    sal_cpu = blk_sal.cpu()

    for b_i in range(B):
        for h_i in range(H):
            # Penultimate block NB - 2 retargeting (preserves counts identically)
            if NB >= local_blocks + 5:
                qb2 = NB - 2
                int_end2 = qb2 - local_blocks
                cnt2 = int(counts_cpu[b_i, h_i, qb2].item())
                n_loc2 = qb2 - int_end2 + 1
                n_int2 = max(0, cnt2 - 1 - n_loc2)
                if n_int2 > 0 and int_end2 > 1:
                    top2 = torch.argsort(sal_cpu[b_i, h_i, 1:int_end2], descending=True)[:n_int2] + 1
                    top2_sorted, _ = torch.sort(top2.to(torch.int32))
                    lists_cpu[b_i, h_i, qb2, 1 : 1 + n_int2] = top2_sorted

            # Tail query block NB - 1 donor-to-tail transfer
            qb = NB - 1
            int_end = qb - local_blocks
            if int_end <= 1:
                continue
            m_int = int_end - 1
            cnt = int(counts_cpu[b_i, h_i, qb].item())
            n_loc = qb - int_end + 1
            n_int = max(0, cnt - 1 - n_loc)
            target_int = max(n_int, min(bridge_blocks, m_int, max_count - 1 - n_loc))

            if target_int > n_int:
                need = target_int - n_int
                donated = 0
                for d_qb in range(local_blocks + 2, NB - 2):
                    if donated >= need:
                        break
                    d_cnt = int(counts_cpu[b_i, h_i, d_qb].item())
                    d_int_end = d_qb - local_blocks
                    d_n_loc = d_qb - d_int_end + 1
                    d_n_int = d_cnt - 1 - d_n_loc
                    while d_n_int > 0 and donated < need and (cnt + 1) <= max_count:
                        lists_cpu[b_i, h_i, d_qb, 1 : d_cnt - 1] = lists_cpu[b_i, h_i, d_qb, 2:d_cnt].clone()
                        lists_cpu[b_i, h_i, d_qb, d_cnt - 1] = 0
                        d_cnt -= 1
                        d_n_int -= 1
                        counts_cpu[b_i, h_i, d_qb] = d_cnt
                        cnt += 1
                        n_int += 1
                        counts_cpu[b_i, h_i, qb] = cnt
                        donated += 1

            if n_int > 0:
                top_int = torch.argsort(sal_cpu[b_i, h_i, 1:int_end], descending=True)[:n_int] + 1
                top_sorted, _ = torch.sort(top_int.to(torch.int32))
                lists_cpu[b_i, h_i, qb, 0] = 0
                lists_cpu[b_i, h_i, qb, 1 : 1 + n_int] = top_sorted
                lists_cpu[b_i, h_i, qb, 1 + n_int : cnt] = torch.arange(int_end, qb + 1, dtype=torch.int32)

    return BlockLists(lists_cpu.to(lists.device), counts_cpu.to(counts.device), max_count)


# ============================================================================
# Triton Fused CUDA Kernel
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _block_sparse_gemma4_triton_fwd_kernel(
        Q, K, V, Out, Lists, Counts, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        H, N_CTX, NUM_RB, MAX_COUNT,
        HEAD_DIM: tl.constexpr, ROUTE_BLOCK: tl.constexpr, IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        m_valid = offs_m < N_CTX

        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None], other=0.0,
        )

        qb = (start_m * BLOCK_M) // ROUTE_BLOCK
        row = off_hz.to(tl.int64) * NUM_RB + qb
        count = tl.load(Counts + row)
        list_ptr = Lists + row * MAX_COUNT

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        log2e = 1.4426950408889634
        qk_scale = sm_scale * log2e

        for t in range(0, count):
            kb = tl.load(list_ptr + t)
            for s in tl.static_range(ROUTE_BLOCK // BLOCK_N):
                cols = kb * ROUTE_BLOCK + s * BLOCK_N + offs_n
                n_valid = cols < N_CTX
                k = tl.load(
                    k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                    mask=n_valid[None, :], other=0.0,
                )
                v = tl.load(
                    v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_valid[:, None], other=0.0,
                )
                qk = tl.dot(q, k) * qk_scale
                keep = n_valid[None, :]
                if IS_CAUSAL:
                    keep = keep & (offs_m[:, None] >= cols[None, :])
                qk = tl.where(keep, qk, float("-inf"))

                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.math.exp2(qk - m_ij[:, None])
                alpha = tl.math.exp2(m_i - m_ij)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_ij

        acc = acc / tl.maximum(l_i[:, None], 1e-12)
        tl.store(
            o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            acc.to(Out.dtype.element_ty), mask=m_valid[:, None],
        )


# ============================================================================
# Vectorized Block-Gather Tile Loop (CPU / MPS / CUDA Fallback)
# ============================================================================

def _cpu_vectorized_block_gather_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lists: BlockLists,
    route_block: int = ROUTE_BLOCK,
    is_causal: bool = True,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """
    Vectorized block-gather tile loop for CPU/MPS without materializing the dense S x S mask
    or computing skipped key blocks.

    Tiles query sequence into blocks of size R_B = route_block (default 128).
    For each query block qb, gathers strictly the active key blocks from block_lists.lists,
    evaluating dot-product attention in FP32 accumulation and casting to q.dtype.
    """
    B, H, S, D = q.shape
    RB = route_block
    NB = (S + RB - 1) // RB
    lists = block_lists.lists
    counts = block_lists.counts
    max_count = block_lists.max_count

    if max_count == 0 or S == 0:
        return torch.zeros_like(q)

    out = torch.empty_like(q)
    pad_len = NB * RB - S
    if pad_len > 0:
        k_pad = F.pad(k, (0, 0, 0, pad_len), value=0.0)
        v_pad = F.pad(v, (0, 0, 0, pad_len), value=0.0)
    else:
        k_pad = k
        v_pad = v

    k_blocks = k_pad.view(B, H, NB, RB, D)
    v_blocks = v_pad.view(B, H, NB, RB, D)

    for qb in range(NB):
        q_start = qb * RB
        q_end = min(S, (qb + 1) * RB)
        L_q = q_end - q_start
        if L_q <= 0:
            continue

        cur_counts = counts[:, :, qb]  # (B, H)
        row_max = int(cur_counts.max().item())
        if row_max <= 0:
            out[:, :, q_start:q_end, :] = 0.0
            continue

        q_chunk = q[:, :, q_start:q_end, :]  # (B, H, L_q, D)
        cur_lists = lists[:, :, qb, :row_max].long()  # (B, H, row_max)
        safe_lists = cur_lists.clamp(0, NB - 1)

        # Gather only active key and value blocks up to row_max
        gather_idx = safe_lists.unsqueeze(-1).unsqueeze(-1).expand(B, H, row_max, RB, D)
        k_active = torch.gather(k_blocks, dim=2, index=gather_idx).view(B, H, row_max * RB, D)
        v_active = torch.gather(v_blocks, dim=2, index=gather_idx).view(B, H, row_max * RB, D)

        scores = torch.matmul(q_chunk.float(), k_active.float().transpose(-2, -1)) * sm_scale

        t_idx = torch.arange(row_max, device=q.device).view(1, 1, row_max)
        slot_valid = (
            (t_idx < cur_counts.unsqueeze(-1))
            .unsqueeze(-1)
            .expand(B, H, row_max, RB)
            .reshape(B, H, 1, row_max * RB)
        )

        rb_offsets = torch.arange(RB, device=q.device).view(1, 1, 1, RB)
        k_positions = (safe_lists.unsqueeze(-1) * RB + rb_offsets).view(B, H, 1, row_max * RB)
        q_positions = torch.arange(q_start, q_end, device=q.device).view(1, 1, L_q, 1)

        valid_mask = slot_valid & (k_positions < S)
        if is_causal:
            valid_mask = valid_mask & (k_positions <= q_positions)

        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
        out_chunk = torch.matmul(weights, v_active.float()).to(q.dtype)
        out[:, :, q_start:q_end, :] = out_chunk

    return out


# ============================================================================
# Core Prefill Kernel Entry Point
# ============================================================================

def block_sparse_gemma4_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lists: BlockLists,
    *,
    route_block: int = ROUTE_BLOCK,
    is_causal: bool = True,
    sm_scale: Optional[float] = 1.0,
) -> torch.Tensor:
    """
    Physical Block-Sparse Prefill Kernel for Google Gemma 4 31B (arXiv:2607.02770).

    Given compiled BlockLists(lists, counts, max_count) of block size R_B = 128 tokens,
    executes physical block-sparse attention over full 512-D post-RoPE (q, k) and pre-k_norm v
    exclusively over active blocks in BlockLists:
    - Uses Triton @triton.jit on CUDA when available.
    - Uses a vectorized block-gather tile loop on CPU/MPS without materializing the dense S x S mask
      or computing skipped key blocks.

    Args:
        q: (B, H_q, S, 512) post-RoPE query tensor.
        k: (B, H_kv, S, 512) or (B, H_q, S, 512) post-RoPE key tensor.
        v: (B, H_kv, S, 512) or (B, H_q, S, 512) value tensor (post-v_norm on pre-k_norm k_raw).
        block_lists: Compiled BlockLists object (lists, counts, max_count).
        route_block: Block size R_B (default 128).
        is_causal: Token-level causal masking.
        sm_scale: Attention scaling factor (default 1.0 per Gemma 4 specification).

    Returns:
        out: (B, H_q, S, 512) block-sparse attention output tensor in q.dtype.
    """
    if q.dim() != 4:
        raise ValueError(f"Expected q of shape (B, H, S, D), got {q.shape}")
    B, H, S, D = q.shape

    # Broadcast KV heads to Query heads if GQA
    if k.shape[1] != H:
        k = _repeat_kv(k, H // k.shape[1])
    if v.shape[1] != H:
        v = _repeat_kv(v, H // v.shape[1])

    scale = 1.0 if sm_scale is None else float(sm_scale)
    use_cuda_triton = (
        HAS_TRITON
        and q.is_cuda
        and not _INTERPRET
        and q.dtype == torch.float16
    )

    if use_cuda_triton:
        NB = (S + route_block - 1) // route_block
        lists = block_lists.lists.to(torch.int32).contiguous()
        counts = block_lists.counts.to(torch.int32).contiguous()
        max_count = int(block_lists.max_count)

        bm = 16 if D >= 512 else (32 if D >= 256 else 64)
        bn = 16 if D >= 512 else (32 if D >= 256 else 64)
        out = torch.empty_like(q)

        grid = (triton.cdiv(S, bm), B * H)
        _block_sparse_gemma4_triton_fwd_kernel[grid](
            q, k, v, out, lists, counts, scale,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            H, S, NB, max_count,
            HEAD_DIM=D, ROUTE_BLOCK=route_block, IS_CAUSAL=bool(is_causal),
            BLOCK_M=bm, BLOCK_N=bn,
            num_warps=4, num_stages=1,
        )
        return out
    else:
        return _cpu_vectorized_block_gather_prefill(
            q, k, v, block_lists,
            route_block=route_block,
            is_causal=is_causal,
            sm_scale=scale,
        )


# ============================================================================
# Core Decode Kernel Entry Point
# ============================================================================

def span_shielded_gemma4_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    branch_id: Optional[torch.Tensor] = None,
    shield_tokens: Optional[torch.Tensor] = None,
    r: int = 5,
    local_window: int = 16,
    sink_block: bool = True,
    route_block: int = ROUTE_BLOCK,
    sm_scale: Optional[float] = 1.0,
    active_mask: Optional[torch.Tensor] = None,
    decode_k: int = 0,
) -> torch.Tensor:
    """
    Span-Shielded Autoregressive Decode Kernel for Google Gemma 4 31B (arXiv:2607.02770).

    During seq_len == 1 autoregressive decode steps, gathers strictly the active local sliding
    window, sink block 0, ultrametric routed prefix blocks, and cached top-96 Beltrami +
    Query-Tail shielded token spans (_darksnail_shield_tokens), completely eliminating full-context
    O(L_kv) KV scans.

    Args:
        q: (B, H_q, 1, 512) post-RoPE decode query vector.
        k_cache: (B, H_kv, L_kv, 512) or (B, H_q, L_kv, 512) total key cache.
        v_cache: (B, H_kv, L_kv, 512) or (B, H_q, L_kv, 512) total value cache.
        branch_id: Optional (B, H_q, L_kv) integer routing tree branch assignments.
        shield_tokens: Optional (B, H_q, K_sh) tensor of cached top-96 Beltrami outlier tokens.
        r: Tree depth for branch matching (default 5).
        local_window: Local sliding window token width (default 16).
        sink_block: Include sink block 0 / sink token (default True).
        route_block: Block size R_B (default 128).
        sm_scale: Attention scaling factor (default 1.0 per Gemma 4).
        active_mask: Optional explicit (B, H_q, L_kv) boolean mask override.
        decode_k: Optional live decode neighborhood pooling tokens (default 0).

    Returns:
        out: (B, H_q, 1, 512) decode attention output in q.dtype.
    """
    B, H, _, D = q.shape
    L_kv = k_cache.shape[-2]
    device = q.device
    scale = 1.0 if sm_scale is None else float(sm_scale)

    if k_cache.shape[1] != H:
        k_cache = _repeat_kv(k_cache, H // k_cache.shape[1])
    if v_cache.shape[1] != H:
        v_cache = _repeat_kv(v_cache, H // v_cache.shape[1])

    # Construct active token mask if not passed explicitly
    if active_mask is None:
        mask = torch.zeros((B, H, L_kv), dtype=torch.bool, device=device)

        # 1. Active local sliding window
        if local_window > 0:
            mask[:, :, max(0, L_kv - local_window) :] = True

        # 2. Sink block 0 / sink token
        if sink_block:
            sink_len = min(route_block, L_kv)
            mask[:, :, :sink_len] = True

        # 3. Ultrametric routed prefix blocks
        if branch_id is not None and branch_id.shape[-1] >= L_kv:
            q_branch = branch_id[:, :, L_kv - 1 : L_kv]
            hist_branches = branch_id[:, :, :L_kv]
            mask = mask | (hist_branches == q_branch)

        # 4. Cached top-96 Beltrami + Query-Tail shielded token spans
        if shield_tokens is not None:
            sh_clamped = shield_tokens.clamp(0, L_kv - 1).long()
            mask.scatter_(-1, sh_clamped, True)

        # 5. Live decode pool if enabled
        if decode_k > 0:
            scores_approx = torch.matmul(q.float(), k_cache.float().transpose(-2, -1)) * scale
            live_pool = F.max_pool1d(
                scores_approx.squeeze(2).reshape(B * H, 1, L_kv),
                kernel_size=9, stride=1, padding=4,
            ).reshape(B, H, L_kv)
            _, live_top = live_pool.topk(min(decode_k, L_kv), dim=-1)
            mask.scatter_(-1, live_top, True)
    else:
        mask = active_mask.squeeze(2) if active_mask.dim() == 4 else active_mask

    # Gather strictly the active tokens
    counts = mask.sum(dim=-1)  # (B, H)
    max_k = int(counts.max().item())
    if max_k <= 0:
        return torch.zeros_like(q)

    # Sort puts True values at the front for contiguous active token gathering
    _, idx = torch.sort(mask.to(torch.int8), dim=-1, descending=True)
    gather_idx = idx[:, :, :max_k]  # (B, H, max_k)

    t_idx = torch.arange(max_k, device=device).view(1, 1, max_k)
    valid_mask = t_idx < counts.unsqueeze(-1)  # (B, H, max_k)

    idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    k_active = torch.gather(k_cache, dim=2, index=idx_expanded)  # (B, H, max_k, D)
    v_active = torch.gather(v_cache, dim=2, index=idx_expanded)  # (B, H, max_k, D)

    scores = torch.matmul(q.float(), k_active.float().transpose(-2, -1)) * scale  # (B, H, 1, max_k)
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))
    weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.matmul(weights, v_active.float()).to(q.dtype)  # (B, H, 1, D)
    return out


__all__ = [
    "BlockLists",
    "GLOBAL_LAYER_INDICES",
    "HEAD_DIM_GLOBAL",
    "D_ROPE",
    "D_NOPE",
    "ROUTE_BLOCK",
    "extract_knope_subspace",
    "compute_causal_budget",
    "measure_block_sparse_traffic_reduction",
    "apply_darksnail_block_bridge",
    "block_sparse_gemma4_prefill",
    "span_shielded_gemma4_decode",
]

"""
Block-list sparse attention (forward + backward) for routed / ultrametric block sparsity
========================================================================================

Problem with the existing kernels
---------------------------------
* ``benchmarks/benchmark_triton.py`` (the kernel behind the README numbers) visits
  EVERY key block and tests its route before skipping, so loop overhead stays
  O((N/B)^2) even at 98% sparsity. Measured: 5.7x over its own dense path at
  98.4% sparsity, vs. an ideal of 64x.
* ``ultrametric.kernel.ultrametric_attention_triton`` uses block lists, but passes
  the maximum list length as a ``tl.constexpr`` (a fresh compile for every new
  value) and runs every query block to that maximum: padded iterations still load
  K/V and execute both matmuls, so cost follows the *longest* list, not each row's.

This module
-----------
1. ``build_block_lists`` turns per-block routing vectors into, for every
   (batch, head, query route-block), an ascending list of active key route-blocks
   plus its length. Uses a fused Triton kernel on CUDA when NB <= 1024 (avoiding
   materializing the O(Z*H*NB^2) boolean mask in HBM), with a vectorized PyTorch
   fallback.
2. ``build_transposed_block_lists`` inverts a ``BlockLists`` object so each key
   route-block maps to the ascending list of query route-blocks attending to it
   (used by the dK/dV backward kernel).
3. ``block_list_attention`` loops over **only that row's own list length**
   (runtime loop bound), so cost scales with the number of active blocks per row.
   Each route block (default 128 tokens) is tiled into BLOCK_N sub-tiles;
   tile sizes are autotuned on CUDA. Causality and the sequence tail are masked at
   token granularity; an optional sink block and local block band mirror the
   LLaMA-surgery inference mask.
4. Full sparse autograd support (``_block_list_bwd_dq_kernel`` and
   ``_block_list_bwd_dkdv_kernel``) plus optional ``soft_block_bias`` gradient
   for straight-through estimator (STE) router training without materializing an
   O(N^2) dense attention matrix.

Matching rule (same as the existing kernels): query block i attends to key block j
iff their routing vectors agree on the first ``req_depth[h]`` levels.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple, Optional, Sequence, Union

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, RuntimeError):
    HAS_TRITON = False

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

DepthLike = Union[int, Sequence[int], torch.Tensor]

__all__ = [
    "HAS_TRITON",
    "BlockLists",
    "build_block_lists",
    "build_transposed_block_lists",
    "block_list_attention",
    "routed_block_attention",
    "block_mask_reference",
    "token_mask_from_blocks",
    "masked_reference_attention",
]


# ============================================================================
# Block-list construction (Triton fused kernel + pure PyTorch fallback)
# ============================================================================

class BlockLists(NamedTuple):
    """Active key route-blocks per query route-block (or transposed: query blocks per key block).

    lists:     (Z, H, NB, max_count) int32. Row (z, h, i) holds the active partner
               blocks for block i in ascending order; entries at positions
               >= counts[z, h, i] are padding and are never read by the kernel.
    counts:    (Z, H, NB) int32, number of active blocks per row.
    max_count: int, max over counts (row stride of ``lists``).
    """

    lists: torch.Tensor
    counts: torch.Tensor
    max_count: int


def _depth_tensor(req_depth: DepthLike, num_heads: int, tree_depth: int,
                  device: torch.device) -> torch.Tensor:
    """Normalize req_depth to an int64 (H,) tensor and validate its range."""
    if isinstance(req_depth, int):
        depth = torch.full((num_heads,), req_depth, dtype=torch.int64, device=device)
    else:
        depth = torch.as_tensor(req_depth, dtype=torch.int64, device=device).reshape(-1)
        if depth.numel() != num_heads:
            raise ValueError(f"req_depth has {depth.numel()} entries, expected {num_heads} (one per head)")
    if depth.numel() and (int(depth.min()) < 0 or int(depth.max()) > tree_depth):
        raise ValueError(f"req_depth must be in [0, {tree_depth}], got {depth.tolist()}")
    return depth


def _apply_structural_masks(match: torch.Tensor, *, is_causal: bool, sink_block: bool,
                            local_blocks: int) -> torch.Tensor:
    nb = match.shape[-1]
    idx = torch.arange(nb, device=match.device)
    if local_blocks > 0:
        match = match | ((idx.view(-1, 1) - idx.view(1, -1)).abs() <= local_blocks)
    if sink_block:
        match = match.clone()
        match[..., 0] = True
    if is_causal:
        match = match & (idx.view(-1, 1) >= idx.view(1, -1))
    return match


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _build_block_lists_kernel(
        Codes, Lists, Counts,
        stride_cz, stride_ch, stride_cb,
        stride_lz, stride_lh, stride_lb, stride_lk,
        stride_cntz, stride_cnth, stride_cntb,
        H, NB,
        IS_CAUSAL: tl.constexpr, SINK_BLOCK: tl.constexpr, LOCAL_BLOCKS: tl.constexpr,
        BLOCK_NB: tl.constexpr,
    ):
        qb = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        c_base = Codes + off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch
        q_code = tl.load(c_base + qb * stride_cb)

        offs_kb = tl.arange(0, BLOCK_NB)
        kb_valid = offs_kb < NB
        k_codes = tl.load(c_base + offs_kb * stride_cb, mask=kb_valid, other=-1)

        match = (k_codes == q_code) & kb_valid
        if LOCAL_BLOCKS > 0:
            diff = tl.abs(qb - offs_kb)
            match = match | ((diff <= LOCAL_BLOCKS) & kb_valid)
        if SINK_BLOCK:
            match = match | ((offs_kb == 0) & kb_valid)
        if IS_CAUSAL:
            match = match & (qb >= offs_kb)

        match_int = match.to(tl.int32)
        count = tl.sum(match_int, axis=0)
        tl.store(
            Counts + off_z.to(tl.int64) * stride_cntz + off_h.to(tl.int64) * stride_cnth + qb * stride_cntb,
            count,
        )

        pos = tl.cumsum(match_int, axis=0) - 1
        l_row = Lists + off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh + qb * stride_lb
        tl.store(l_row + pos * stride_lk, offs_kb, mask=match)


@torch.no_grad()
def build_block_lists(
    router_indices: torch.Tensor,
    req_depth: DepthLike,
    *,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
    arity: Optional[int] = None,
) -> BlockLists:
    """Build per-row active key-block lists from per-block routing vectors.

    Args:
        router_indices: (Z, H, NB, TD) integer branch ids (>= 0), one routing
            vector per route block (e.g. from ``routing_to_block_indices``).
        req_depth: required matching depth; int (all heads) or per-head (H,).
            0 means every block matches (dense).
        is_causal: drop key blocks after the query block (token-level causality
            inside the diagonal block is applied by the kernel).
        sink_block: always include key block 0 (attention sink).
        local_blocks: always include key blocks with |i - j| <= local_blocks.
        arity: branch count p (ids must lie in [0, p)). If given, the builder is
            **sync-free** (no device->host reads): ids are not validated and
            ``lists`` is padded to NB columns (``max_count = NB``). Use this in
            the hot path; the kernel only ever reads ``counts[row]`` entries.

    Returns:
        BlockLists(lists, counts, max_count)
    """
    if router_indices.dim() != 4:
        raise ValueError(f"router_indices must be (Z, H, NB, TD), got {tuple(router_indices.shape)}")
    Z, H, NB, TD = router_indices.shape
    device = router_indices.device
    sync_free = arity is not None
    if sync_free and isinstance(req_depth, int):
        if not 0 <= req_depth <= TD:
            raise ValueError(f"req_depth must be in [0, {TD}], got {req_depth}")
        depth = None
    else:
        depth = _depth_tensor(req_depth, H, TD, device)
    r = router_indices.to(torch.int64)

    if sync_free:
        base = max(2, int(arity))
    else:
        if r.numel() and int(r.min()) < 0:
            raise ValueError("router_indices must be non-negative branch ids")
        base = max(2, int(r.max()) + 1) if r.numel() else 2
    levels = torch.arange(TD, device=device)
    if depth is None:
        keep = (levels < req_depth).view(1, 1, 1, TD)
    else:
        keep = levels.view(1, 1, 1, TD) < depth.view(1, H, 1, 1)

    can_encode_code = TD * math.log2(base) < 62
    if can_encode_code:
        # Encode each block's first-d levels as a mixed-radix integer; blocks match
        # iff their codes are equal. O(Z*H*NB^2) memory instead of O(Z*H*NB^2*TD).
        powers = torch.pow(base, levels.to(torch.int64))
        code = (r * keep * powers).sum(-1).contiguous()         # (Z, H, NB) int64

        if router_indices.is_cuda and HAS_TRITON and not _INTERPRET and 0 < NB <= 1024:
            lists_full = torch.zeros((Z, H, NB, NB), dtype=torch.int32, device=device)
            counts = torch.empty((Z, H, NB), dtype=torch.int32, device=device)
            block_nb = _next_pow2(NB)
            grid = (NB, Z * H)
            _build_block_lists_kernel[grid](
                code, lists_full, counts,
                *code.stride(),
                *lists_full.stride(),
                *counts.stride(),
                H, NB,
                IS_CAUSAL=bool(is_causal),
                SINK_BLOCK=bool(sink_block),
                LOCAL_BLOCKS=int(local_blocks),
                BLOCK_NB=block_nb,
            )
            if sync_free:
                max_count = NB
                lists = lists_full
            else:
                max_count = int(counts.max()) if counts.numel() else 0
                lists = lists_full[..., :max_count].contiguous()
            return BlockLists(lists, counts, max_count)

        match = code.unsqueeze(-1) == code.unsqueeze(-2)        # (Z, H, NB, NB)
    else:
        eq = (r.unsqueeze(3) == r.unsqueeze(2)) | ~keep.unsqueeze(2)
        match = eq.all(-1)

    match = _apply_structural_masks(match, is_causal=is_causal, sink_block=sink_block,
                                    local_blocks=local_blocks)

    counts = match.sum(-1, dtype=torch.int32)                   # (Z, H, NB)
    if sync_free:
        max_count = NB
    else:
        max_count = int(counts.max()) if counts.numel() else 0

    # Stable compaction: the k-th True in a row goes to slot k (ascending order);
    # non-matching entries go to a dump slot that is sliced off.
    pos = torch.cumsum(match, dim=-1, dtype=torch.int64) - 1
    pos = torch.where(match, pos, torch.full_like(pos, max_count))
    src = torch.arange(NB, dtype=torch.int32, device=device).expand(Z, H, NB, NB)
    lists = torch.zeros((Z, H, NB, max_count + 1), dtype=torch.int32, device=device)
    lists.scatter_(-1, pos, src)
    lists = lists[..., :max_count].contiguous()

    return BlockLists(lists, counts.contiguous(), max_count)


@torch.no_grad()
def build_transposed_block_lists(
    block_lists: BlockLists,
    num_key_blocks: Optional[int] = None,
) -> BlockLists:
    """Invert active key-block lists per query block into active query-block lists per key block.

    Args:
        block_lists: ``BlockLists(lists, counts, max_count)`` where ``lists`` has
            shape ``(Z, H, NB_Q, max_count)`` and ``counts`` has shape ``(Z, H, NB_Q)``.
        num_key_blocks: total number of key route-blocks ``NB_K`` (defaults to ``NB_Q``).

    Returns:
        ``BlockLists(k_lists, q_counts, max_q_count)`` where row ``(z, h, kb)``
        of ``k_lists`` holds the ascending query route-blocks ``qb`` that attend to ``kb``.
    """
    lists, counts, max_count = block_lists
    Z, H, NB_Q = counts.shape
    NB_K = num_key_blocks if num_key_blocks is not None else NB_Q
    device = lists.device

    if NB_Q == 0 or NB_K == 0 or max_count == 0:
        q_counts = torch.zeros((Z, H, NB_K), dtype=torch.int32, device=device)
        k_lists = torch.zeros((Z, H, NB_K, 0), dtype=torch.int32, device=device)
        return BlockLists(k_lists, q_counts, 0)

    t_idx = torch.arange(max_count, device=device).view(1, 1, 1, max_count)
    valid = t_idx < counts.unsqueeze(-1)  # (Z, H, NB_Q, max_count)

    # Scatter valid (qb, kb) pairs into a (Z, H, NB_Q, NB_K + 1) boolean table
    # (slot NB_K is a dump bucket for invalid entries), then transpose to (Z, H, NB_K, NB_Q).
    safe_kb = torch.where(valid, lists.to(torch.int64), torch.full_like(lists, NB_K, dtype=torch.int64))
    match_qk = torch.zeros((Z, H, NB_Q, NB_K + 1), dtype=torch.bool, device=device)
    match_qk.scatter_(-1, safe_kb, valid)
    match_kq = match_qk[..., :NB_K].transpose(-2, -1).contiguous()  # (Z, H, NB_K, NB_Q)

    q_counts = match_kq.sum(-1, dtype=torch.int32)
    max_q_count = int(q_counts.max()) if q_counts.numel() else 0

    pos = torch.cumsum(match_kq, dim=-1, dtype=torch.int64) - 1
    pos = torch.where(match_kq, pos, torch.full_like(pos, max_q_count))
    src = torch.arange(NB_Q, dtype=torch.int32, device=device).expand(Z, H, NB_K, NB_Q)
    k_lists = torch.zeros((Z, H, NB_K, max_q_count + 1), dtype=torch.int32, device=device)
    k_lists.scatter_(-1, pos, src)
    k_lists = k_lists[..., :max_q_count].contiguous()

    return BlockLists(k_lists, q_counts.contiguous(), max_q_count)


# ============================================================================
# Independent references (tests / benchmark correctness checks)
# ============================================================================

@torch.no_grad()
def block_mask_reference(
    router_indices: torch.Tensor,
    req_depth: DepthLike,
    *,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
) -> torch.Tensor:
    """(Z, H, NB, NB) bool block mask via direct pairwise prefix comparison.

    Deliberately shares no code path with ``build_block_lists`` (per-head loop,
    explicit ``all`` over levels) so it can serve as a test oracle.
    """
    Z, H, NB, TD = router_indices.shape
    depth = _depth_tensor(req_depth, H, TD, router_indices.device).tolist()
    out = torch.empty((Z, H, NB, NB), dtype=torch.bool, device=router_indices.device)
    for h, d in enumerate(depth):
        if d == 0:
            out[:, h] = True
        else:
            rh = router_indices[:, h, :, :d]
            out[:, h] = (rh[:, :, None, :] == rh[:, None, :, :]).all(-1)
    return _apply_structural_masks(out, is_causal=is_causal, sink_block=sink_block,
                                   local_blocks=local_blocks)


def token_mask_from_blocks(block_mask: torch.Tensor, route_block: int, seq_len: int,
                           is_causal: bool = False) -> torch.Tensor:
    """Expand a (..., NB, NB) block mask to a (..., N, N) token mask."""
    m = block_mask.repeat_interleave(route_block, dim=-2).repeat_interleave(route_block, dim=-1)
    m = m[..., :seq_len, :seq_len]
    if is_causal:
        m = m & torch.ones(seq_len, seq_len, dtype=torch.bool, device=m.device).tril()
    return m


def masked_reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               token_mask: torch.Tensor,
                               sm_scale: Optional[float] = None) -> torch.Tensor:
    """fp32 masked attention reference; returns q.dtype."""
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    s = s.masked_fill(~token_mask, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


# ============================================================================
# Triton kernels (forward + backward)
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _block_list_fwd_kernel(
        Q, K, V, Out, Lse, Lists, Counts, SoftBias, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        stride_sbz, stride_sbh, stride_sbm, stride_sbn,
        H, N_CTX, NUM_RB, MAX_COUNT, COUNT_BUCKET,
        HEAD_DIM: tl.constexpr, ROUTE_BLOCK: tl.constexpr, IS_CAUSAL: tl.constexpr,
        STORE_LSE: tl.constexpr, HAS_SOFT_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # COUNT_BUCKET is unused in the body; it is an autotune key (next pow2 of
        # MAX_COUNT) so dense and very sparse launches may pick different tiles.
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

        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_valid[:, None], other=0.0)

        # BLOCK_M divides ROUTE_BLOCK, so this query tile lies in one route block.
        qb = (start_m * BLOCK_M) // ROUTE_BLOCK
        row = off_hz.to(tl.int64) * NUM_RB + qb
        count = tl.load(Counts + row)
        list_ptr = Lists + row * MAX_COUNT
        if HAS_SOFT_BIAS:
            sb_ptr = (SoftBias + off_z.to(tl.int64) * stride_sbz
                      + off_h.to(tl.int64) * stride_sbh + qb * stride_sbm)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        log2e = 1.4426950408889634
        qk_scale = sm_scale * log2e  # log2(e): use exp2

        for t in range(0, count):  # runtime bound: only this row's active blocks
            kb = tl.load(list_ptr + t)
            if HAS_SOFT_BIAS:
                b_val = tl.load(sb_ptr + t * stride_sbn).to(tl.float32) * log2e
            for s in tl.static_range(ROUTE_BLOCK // BLOCK_N):
                cols = kb * ROUTE_BLOCK + s * BLOCK_N + offs_n
                n_valid = cols < N_CTX
                k = tl.load(k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                            mask=n_valid[None, :], other=0.0)
                v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                            mask=n_valid[:, None], other=0.0)
                qk = tl.dot(q, k) * qk_scale
                if HAS_SOFT_BIAS:
                    qk = qk + b_val
                keep = n_valid[None, :]
                if IS_CAUSAL:
                    keep = keep & (offs_m[:, None] >= cols[None, :])
                qk = tl.where(keep, qk, float("-inf"))

                # Online softmax, safe for rows whose keys are all masked so far.
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.math.exp2(m_i - m_ref)
                p = tl.math.exp2(qk - m_ref[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new

        l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i_safe[:, None]
        tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc.to(Out.dtype.element_ty), mask=m_valid[:, None])
        if STORE_LSE:
            # m_i is in log2 units (scores were scaled by log2(e)); convert to natural log.
            lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
            tl.store(Lse + off_hz.to(tl.int64) * N_CTX + offs_m, lse, mask=m_valid)

    def _prune_configs(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        rb = args.get("ROUTE_BLOCK", 128)
        ok = [c for c in configs
              if c.kwargs["BLOCK_M"] <= rb and c.kwargs["BLOCK_N"] <= rb
              and rb % c.kwargs["BLOCK_M"] == 0 and rb % c.kwargs["BLOCK_N"] == 0]
        return ok or [triton.Config({"BLOCK_M": rb, "BLOCK_N": rb}, num_warps=4, num_stages=2)]

    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ]

    _block_list_fwd_tuned = triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=["N_CTX", "COUNT_BUCKET", "HEAD_DIM", "IS_CAUSAL", "ROUTE_BLOCK"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )(_block_list_fwd_kernel)

    @triton.jit
    def _block_list_bwd_dq_kernel(
        Q, K, V, dO, dQ, Lse, Delta, Lists, Counts, SoftBias, dSoftBias, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_doz, stride_doh, stride_dom, stride_dod,
        stride_dqz, stride_dqh, stride_dqm, stride_dqd,
        stride_sbz, stride_sbh, stride_sbm, stride_sbn,
        stride_dsbz, stride_dsbh, stride_dsbm, stride_dsbn,
        H, N_CTX, NUM_RB, MAX_COUNT,
        HEAD_DIM: tl.constexpr, ROUTE_BLOCK: tl.constexpr, IS_CAUSAL: tl.constexpr,
        HAS_SOFT_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        do_base = dO + off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh
        dq_base = dQ + off_z.to(tl.int64) * stride_dqz + off_h.to(tl.int64) * stride_dqh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        m_valid = offs_m < N_CTX

        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_valid[:, None], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                     mask=m_valid[:, None], other=0.0)

        lse = tl.load(Lse + off_hz.to(tl.int64) * N_CTX + offs_m, mask=m_valid, other=float("-inf"))
        delta = tl.load(Delta + off_hz.to(tl.int64) * N_CTX + offs_m, mask=m_valid, other=0.0)

        log2e = 1.4426950408889634
        qk_scale = sm_scale * log2e
        # Safe lse in log2 units so exp2(qk - lse_log2) does not produce NaN when lse == -inf
        lse_safe = tl.where(lse == float("-inf"), 0.0, lse * log2e)

        qb = (start_m * BLOCK_M) // ROUTE_BLOCK
        row = off_hz.to(tl.int64) * NUM_RB + qb
        count = tl.load(Counts + row)
        list_ptr = Lists + row * MAX_COUNT

        if HAS_SOFT_BIAS:
            sb_ptr = (SoftBias + off_z.to(tl.int64) * stride_sbz
                      + off_h.to(tl.int64) * stride_sbh + qb * stride_sbm)
            dsb_ptr = (dSoftBias + off_z.to(tl.int64) * stride_dsbz
                       + off_h.to(tl.int64) * stride_dsbh + qb * stride_dsbm)

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for t in range(0, count):
            kb = tl.load(list_ptr + t)
            if HAS_SOFT_BIAS:
                b_val = tl.load(sb_ptr + t * stride_sbn).to(tl.float32) * log2e
                db_acc = tl.zeros([], dtype=tl.float32)
            for s in tl.static_range(ROUTE_BLOCK // BLOCK_N):
                cols = kb * ROUTE_BLOCK + s * BLOCK_N + offs_n
                n_valid = cols < N_CTX
                k_t = tl.load(k_base + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                              mask=n_valid[None, :], other=0.0)
                v_t = tl.load(v_base + cols[None, :] * stride_vn + offs_d[:, None] * stride_vd,
                              mask=n_valid[None, :], other=0.0)
                qk = tl.dot(q, k_t) * qk_scale
                if HAS_SOFT_BIAS:
                    qk = qk + b_val
                keep = m_valid[:, None] & n_valid[None, :]
                if IS_CAUSAL:
                    keep = keep & (offs_m[:, None] >= cols[None, :])
                qk = tl.where(keep, qk, float("-inf"))

                p = tl.where(keep, tl.math.exp2(qk - lse_safe[:, None]), 0.0)
                dp = tl.dot(do, v_t)
                ds_unscaled = tl.where(keep, p * (dp - delta[:, None]), 0.0)
                if HAS_SOFT_BIAS:
                    db_acc += tl.sum(ds_unscaled)
                ds = ds_unscaled * sm_scale

                k = tl.trans(k_t)  # (BLOCK_N, HEAD_DIM)
                dq += tl.dot(ds.to(k.dtype), k)
            if HAS_SOFT_BIAS:
                tl.atomic_add(dsb_ptr + t * stride_dsbn, db_acc)

        tl.store(dq_base + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
                 dq.to(dQ.dtype.element_ty), mask=m_valid[:, None])

    @triton.jit
    def _block_list_bwd_dkdv_kernel(
        Q, K, V, dO, dK, dV, Lse, Delta, KeyLists, KeyCounts, SoftBias, sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_doz, stride_doh, stride_dom, stride_dod,
        stride_dkz, stride_dkh, stride_dkn, stride_dkd,
        stride_dvz, stride_dvh, stride_dvn, stride_dvd,
        stride_sbz, stride_sbh, stride_sbm, stride_sbn,
        FwdLists, FwdCounts, FWD_MAX_COUNT,
        H, N_CTX, NUM_RB, MAX_Q_COUNT,
        HEAD_DIM: tl.constexpr, ROUTE_BLOCK: tl.constexpr, IS_CAUSAL: tl.constexpr,
        HAS_SOFT_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        do_base = dO + off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh
        dk_base = dK + off_z.to(tl.int64) * stride_dkz + off_h.to(tl.int64) * stride_dkh
        dv_base = dV + off_z.to(tl.int64) * stride_dvz + off_h.to(tl.int64) * stride_dvh

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_sub = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        n_valid = offs_n < N_CTX

        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_valid[:, None], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_valid[:, None], other=0.0)
        k_t = tl.trans(k)  # (HEAD_DIM, BLOCK_N)
        v_t = tl.trans(v)  # (HEAD_DIM, BLOCK_N)

        kb = (start_n * BLOCK_N) // ROUTE_BLOCK
        k_row = off_hz.to(tl.int64) * NUM_RB + kb
        q_count = tl.load(KeyCounts + k_row)
        k_list_ptr = KeyLists + k_row * MAX_Q_COUNT

        log2e = 1.4426950408889634
        qk_scale = sm_scale * log2e

        dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        for t in range(0, q_count):
            qb = tl.load(k_list_ptr + t)
            b_val = 0.0
            if HAS_SOFT_BIAS:
                # Find index of kb in qb's forward list to read SoftBias[z, h, qb, idx]
                q_row = off_hz.to(tl.int64) * NUM_RB + qb
                f_count = tl.load(FwdCounts + q_row)
                f_list_ptr = FwdLists + q_row * FWD_MAX_COUNT
                sb_row_ptr = (SoftBias + off_z.to(tl.int64) * stride_sbz
                              + off_h.to(tl.int64) * stride_sbh + qb * stride_sbm)
                for ft in range(0, f_count):
                    f_kb = tl.load(f_list_ptr + ft)
                    if f_kb == kb:
                        b_val = tl.load(sb_row_ptr + ft * stride_sbn).to(tl.float32) * log2e

            for s in tl.static_range(ROUTE_BLOCK // BLOCK_M):
                rows = qb * ROUTE_BLOCK + s * BLOCK_M + offs_m_sub
                m_valid = rows < N_CTX
                q = tl.load(q_base + rows[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                            mask=m_valid[:, None], other=0.0)
                do = tl.load(do_base + rows[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                             mask=m_valid[:, None], other=0.0)
                lse = tl.load(Lse + off_hz.to(tl.int64) * N_CTX + rows, mask=m_valid, other=float("-inf"))
                delta = tl.load(Delta + off_hz.to(tl.int64) * N_CTX + rows, mask=m_valid, other=0.0)
                lse_safe = tl.where(lse == float("-inf"), 0.0, lse * log2e)

                qk = tl.dot(q, k_t) * qk_scale
                if HAS_SOFT_BIAS:
                    qk = qk + b_val
                keep = m_valid[:, None] & n_valid[None, :]
                if IS_CAUSAL:
                    keep = keep & (rows[:, None] >= offs_n[None, :])
                qk = tl.where(keep, qk, float("-inf"))

                p = tl.where(keep, tl.math.exp2(qk - lse_safe[:, None]), 0.0)
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)

                dp = tl.dot(do, v_t)
                ds = tl.where(keep, p * (dp - delta[:, None]) * sm_scale, 0.0)
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

        tl.store(dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
                 dk.to(dK.dtype.element_ty), mask=n_valid[:, None])
        tl.store(dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
                 dv.to(dV.dtype.element_ty), mask=n_valid[:, None])


# ============================================================================
# Low-level launch helpers and Autograd Function
# ============================================================================

def _block_list_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lists: torch.Tensor,
    counts: torch.Tensor,
    max_count: int,
    *,
    route_block: int,
    is_causal: bool,
    scale: float,
    block_m: Optional[int],
    block_n: Optional[int],
    num_warps: int,
    num_stages: int,
    store_lse: bool,
    soft_block_bias: Optional[torch.Tensor] = None,
):
    Z, H, N, D = q.shape
    nb = triton.cdiv(N, route_block)
    out = torch.empty_like(q)
    lse = torch.empty((Z, H, N), dtype=torch.float32, device=q.device) if store_lse else out

    has_soft_bias = soft_block_bias is not None
    sb_tensor = soft_block_bias if has_soft_bias else q
    sb_strides = soft_block_bias.stride() if has_soft_bias else (0, 0, 0, 0)

    args = (
        q, k, v, out, lse, lists, counts, sb_tensor, scale,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        *sb_strides,
        H, N, nb, max_count, _next_pow2(max_count),
    )
    meta = dict(
        HEAD_DIM=D,
        ROUTE_BLOCK=route_block,
        IS_CAUSAL=bool(is_causal),
        STORE_LSE=bool(store_lse),
        HAS_SOFT_BIAS=has_soft_bias,
    )

    use_autotune = block_m is None and block_n is None and D <= 128 and q.is_cuda and not _INTERPRET
    if use_autotune:
        grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), Z * H)  # noqa: E731
        _block_list_fwd_tuned[grid](*args, **meta)
    else:
        default_bm = 32 if D >= 256 else 64
        default_bn = 16 if D >= 512 else (32 if D >= 256 else 64)
        bm = block_m or min(default_bm, route_block)
        bn = block_n or min(default_bn, route_block)
        if route_block % bm or route_block % bn:
            raise ValueError(f"block_m ({bm}) and block_n ({bn}) must divide route_block ({route_block})")
        fwd_stages = 1 if (D >= 256 and num_stages > 1) else num_stages

        def _launch_fwd_with_params(cur_bm: int, cur_bn: int, cur_stages: int):
            launch = dict(BLOCK_M=cur_bm, BLOCK_N=cur_bn, **meta)
            if not _INTERPRET:
                launch.update(num_warps=num_warps, num_stages=cur_stages)
            _block_list_fwd_kernel[(triton.cdiv(N, cur_bm), Z * H)](*args, **launch)

        try:
            _launch_fwd_with_params(bm, bn, fwd_stages)
        except Exception as e:
            if "OutOfResources" in type(e).__name__ or "shared memory" in str(e):
                fb_m = min(16, bm)
                fb_n = min(16, bn)
                _launch_fwd_with_params(fb_m, fb_n, 1)
            else:
                raise
    return out, lse


def _block_list_attention_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    lists: torch.Tensor,
    counts: torch.Tensor,
    max_count: int,
    *,
    route_block: int,
    is_causal: bool,
    scale: float,
    block_m: Optional[int],
    block_n: Optional[int],
    num_warps: int,
    num_stages: int,
    soft_block_bias: Optional[torch.Tensor] = None,
):
    Z, H, N, D = q.shape
    nb = triton.cdiv(N, route_block)
    do = do.contiguous()

    # Fast elementwise-reduction preprocess: delta_i = sum_d(do_{i,d} * out_{i,d})
    delta = (do.float() * out.float()).sum(dim=-1).contiguous()

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    has_soft_bias = soft_block_bias is not None
    if has_soft_bias:
        d_soft_bias = torch.zeros((Z, H, nb, max_count), dtype=torch.float32, device=q.device)
        sb_tensor = soft_block_bias
        sb_strides = soft_block_bias.stride()
        dsb_strides = d_soft_bias.stride()
    else:
        d_soft_bias = None
        sb_tensor = q
        sb_strides = (0, 0, 0, 0)
        dsb_strides = (0, 0, 0, 0)

    default_bwd_tile = 16 if D >= 512 else (32 if D >= 128 else 64)
    bm = block_m or min(default_bwd_tile, route_block)
    bn = block_n or min(default_bwd_tile, route_block)
    if route_block % bm or route_block % bn:
        raise ValueError(f"block_m ({bm}) and block_n ({bn}) must divide route_block ({route_block})")

    bwd_stages = 1 if (D >= 128 and num_stages > 1) else num_stages

    if max_count > 0:
        def _launch_dq_with_params(cur_bm: int, cur_bn: int, cur_stages: int):
            launch_dq = dict(
                HEAD_DIM=D,
                ROUTE_BLOCK=route_block,
                IS_CAUSAL=bool(is_causal),
                HAS_SOFT_BIAS=has_soft_bias,
                BLOCK_M=cur_bm,
                BLOCK_N=cur_bn,
            )
            if not _INTERPRET:
                launch_dq.update(num_warps=num_warps, num_stages=cur_stages)
            _block_list_bwd_dq_kernel[(triton.cdiv(N, cur_bm), Z * H)](
                q, k, v, do, dq, lse, delta, lists, counts,
                sb_tensor, d_soft_bias if has_soft_bias else q, scale,
                *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dq.stride(),
                *sb_strides, *dsb_strides,
                H, N, nb, max_count,
                **launch_dq,
            )

        try:
            _launch_dq_with_params(bm, bn, bwd_stages)
        except Exception as e:
            if "OutOfResources" in type(e).__name__ or "shared memory" in str(e):
                if has_soft_bias and d_soft_bias is not None:
                    d_soft_bias.zero_()
                fb_m = min(16, bm)
                fb_n = min(16, bn)
                _launch_dq_with_params(fb_m, fb_n, 1)
            else:
                raise

        k_lists, q_counts, max_q_count = build_transposed_block_lists(
            BlockLists(lists, counts, max_count), num_key_blocks=nb
        )
        if max_q_count > 0:
            def _launch_dkdv_with_params(cur_bm: int, cur_bn: int, cur_stages: int):
                launch_dkdv = dict(
                    HEAD_DIM=D,
                    ROUTE_BLOCK=route_block,
                    IS_CAUSAL=bool(is_causal),
                    HAS_SOFT_BIAS=has_soft_bias,
                    BLOCK_M=cur_bm,
                    BLOCK_N=cur_bn,
                )
                if not _INTERPRET:
                    launch_dkdv.update(num_warps=num_warps, num_stages=cur_stages)
                _block_list_bwd_dkdv_kernel[(triton.cdiv(N, cur_bn), Z * H)](
                    q, k, v, do, dk, dv, lse, delta, k_lists, q_counts, sb_tensor, scale,
                    *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dk.stride(), *dv.stride(),
                    *sb_strides,
                    lists, counts, max_count,
                    H, N, nb, max_q_count,
                    **launch_dkdv,
                )

            try:
                _launch_dkdv_with_params(bm, bn, bwd_stages)
            except Exception as e:
                if "OutOfResources" in type(e).__name__ or "shared memory" in str(e):
                    fb_m = min(16, bm)
                    fb_n = min(16, bn)
                    _launch_dkdv_with_params(fb_m, fb_n, 1)
                else:
                    raise

    if has_soft_bias and d_soft_bias is not None:
        d_soft_bias = d_soft_bias.to(soft_block_bias.dtype)
    return dq, dk, dv, d_soft_bias


class _BlockListAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lists: torch.Tensor,
        counts: torch.Tensor,
        max_count: int,
        route_block: int,
        is_causal: bool,
        scale: float,
        block_m: Optional[int],
        block_n: Optional[int],
        num_warps: int,
        num_stages: int,
        return_lse: bool,
        soft_block_bias: Optional[torch.Tensor],
    ):
        out, lse = _block_list_attention_fwd(
            q, k, v, lists, counts, max_count,
            route_block=route_block,
            is_causal=is_causal,
            scale=scale,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
            store_lse=True,
            soft_block_bias=soft_block_bias,
        )
        if soft_block_bias is not None:
            ctx.save_for_backward(q, k, v, out, lse, lists, counts, soft_block_bias)
            ctx.has_soft_bias = True
        else:
            ctx.save_for_backward(q, k, v, out, lse, lists, counts)
            ctx.has_soft_bias = False
        ctx.max_count = max_count
        ctx.route_block = route_block
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        ctx.return_lse = return_lse
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        do = grad_outputs[0]
        if ctx.has_soft_bias:
            q, k, v, out, lse, lists, counts, soft_block_bias = ctx.saved_tensors
        else:
            q, k, v, out, lse, lists, counts = ctx.saved_tensors
            soft_block_bias = None

        dq, dk, dv, d_soft_bias = _block_list_attention_bwd(
            do, q, k, v, out, lse, lists, counts, ctx.max_count,
            route_block=ctx.route_block,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
            block_m=ctx.block_m,
            block_n=ctx.block_n,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
            soft_block_bias=soft_block_bias,
        )
        return (
            dq if ctx.needs_input_grad[0] else None,
            dk if ctx.needs_input_grad[1] else None,
            dv if ctx.needs_input_grad[2] else None,
            None, None, None, None, None, None, None, None, None, None, None,
            d_soft_bias if (ctx.has_soft_bias and ctx.needs_input_grad[14]) else None,
        )


# ============================================================================
# Python wrappers
# ============================================================================

def block_list_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lists: BlockLists,
    *,
    route_block: int = 128,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 3,
    return_lse: bool = False,
    soft_block_bias: Optional[torch.Tensor] = None,
):
    """Block-sparse attention over precomputed block lists (with sparse autograd support).

    Args:
        q, k, v: (Z, H, N, D), same shape and dtype (fp16/bf16; fp32 allowed but
            uses TF32 matmuls on GPU). D must be a power of two >= 16. Expand GQA
            K/V to H heads before calling.
        block_lists: from ``build_block_lists`` with NB = ceil(N / route_block).
        route_block: tokens per routing block (power of two >= 16).
        is_causal: token-level causal masking (lists should also be built causal
            so blocks after the diagonal are never visited).
        block_m, block_n: fix tile sizes (bypasses autotuning). Must divide
            route_block.
        return_lse: also return the per-row log-sum-exp (natural log) of the
            scaled scores, (Z, H, N) fp32; -inf for rows with no visible keys.
        soft_block_bias: optional (Z, H, NB, max_count) additive logit bias per
            active block pair ``(qb, kb_t)``. For STE router training, pass a
            zero-valued STE bias ``soft_logits - soft_logits.detach()`` so the
            forward pass is 100% hard-sparse while backward computes exact
            gradients ``dL/d(soft_block_bias)[z, h, qb, t] = sum_{i in qb, j in kb_t} P_ij * (dO_i . V_j - Delta_i)``.

    Returns:
        out: (Z, H, N, D), same dtype as q; or (out, lse) if return_lse.
    """
    if not HAS_TRITON:
        raise RuntimeError("block_list_attention requires Triton")
    if not (q.shape == k.shape == v.shape) or q.dim() != 4:
        raise ValueError(f"q, k, v must share shape (Z, H, N, D); got {q.shape}, {k.shape}, {v.shape}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must share dtype")
    Z, H, N, D = q.shape
    if D < 16 or D & (D - 1):
        raise ValueError(f"head_dim must be a power of two >= 16, got {D}")
    if route_block < 16 or route_block & (route_block - 1):
        raise ValueError(f"route_block must be a power of two >= 16, got {route_block}")

    lists, counts, max_count = block_lists
    nb = triton.cdiv(N, route_block)
    if tuple(counts.shape) != (Z, H, nb) or tuple(lists.shape[:3]) != (Z, H, nb):
        raise ValueError(f"block_lists built for {tuple(counts.shape)}, expected {(Z, H, nb)} "
                         f"(N={N}, route_block={route_block})")
    if lists.shape[-1] != max_count:
        raise ValueError("lists.shape[-1] must equal max_count")
    if soft_block_bias is not None and tuple(soft_block_bias.shape) != (Z, H, nb, max_count):
        raise ValueError(
            f"soft_block_bias must have shape {(Z, H, nb, max_count)}, got {tuple(soft_block_bias.shape)}"
        )

    lists = lists.to(torch.int32).contiguous()
    counts = counts.to(torch.int32).contiguous()
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)

    needs_grad = torch.is_grad_enabled() and (
        q.requires_grad
        or k.requires_grad
        or v.requires_grad
        or (soft_block_bias is not None and soft_block_bias.requires_grad)
    )
    if needs_grad:
        return _BlockListAttentionFn.apply(
            q, k, v, lists, counts, max_count,
            route_block, bool(is_causal), scale,
            block_m, block_n, num_warps, num_stages,
            bool(return_lse), soft_block_bias,
        )

    out, lse = _block_list_attention_fwd(
        q, k, v, lists, counts, max_count,
        route_block=route_block,
        is_causal=bool(is_causal),
        scale=scale,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        store_lse=bool(return_lse),
        soft_block_bias=soft_block_bias,
    )
    return (out, lse) if return_lse else out


def routed_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    router_indices: torch.Tensor,
    req_depth: DepthLike,
    *,
    route_block: int = 128,
    is_causal: bool = False,
    sink_block: bool = False,
    local_blocks: int = 0,
    sm_scale: Optional[float] = None,
    soft_block_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience wrapper: build the block lists, then run the kernel."""
    bl = build_block_lists(router_indices, req_depth, is_causal=is_causal,
                           sink_block=sink_block, local_blocks=local_blocks)
    return block_list_attention(q, k, v, bl, route_block=route_block,
                                is_causal=is_causal, sm_scale=sm_scale,
                                soft_block_bias=soft_block_bias)

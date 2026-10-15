"""
Ultrametric AI — V3 Triton Block-Sparse Attention Kernel

Hardware-accelerated block-sparse attention that dynamically skips SRAM loads
for blocks whose routing vectors indicate they do not share the required
ancestral depth in the Bruhat-Tits tree.

Upgraded for V3 to support:
1. Attention Sinks (Token 0 is permanently visible)
2. Local Grammar Windows (Tokens within local_window are permanently visible)
"""

import torch
import math
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
        Q, K, V, sm_scale, router_indices, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        stride_rz, stride_rh, stride_rm, stride_rd,
        req_depth, shift_size, Z, H, N_CTX,
        LOCAL_WINDOW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
        P_ARY: tl.constexpr, TREE_DEPTH_P2: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Base pointers
        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh
        r_offset = off_z * stride_rz + off_h * stride_rh

        q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

        o_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

        depth_offsets = tl.arange(0, TREE_DEPTH_P2)
        m_router_ptrs = router_indices + r_offset + start_m * stride_rm + depth_offsets * stride_rd
        m_routing_vec = tl.load(m_router_ptrs)

        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        k_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )

        v_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )

        num_n_blocks = start_m + 1
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        region_m = offs_m >= (N_CTX - shift_size)

        for start_n_block in range(0, num_n_blocks):
            n_router_ptrs = router_indices + r_offset + start_n_block * stride_rm + depth_offsets * stride_rd
            n_routing_vec = tl.load(n_router_ptrs)

            # TOPOLOGICAL DISTANCE
            mismatch = (m_routing_vec != n_routing_vec) & (depth_offsets < req_depth)
            has_mismatch = tl.max(mismatch.to(tl.int32), axis=0)

            # V3 FIX: Attention Sink (Block 0 is always visible)
            is_sink = (start_n_block == 0)
            
            # V3 FIX: Local Grammar Window (Blocks within local_window are always visible)
            is_local = (start_m - start_n_block) <= LOCAL_WINDOW_BLOCKS

            if has_mismatch == 0 or is_sink or is_local:
                k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

                qk = tl.dot(q, k) * sm_scale

                offs_n = start_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                region_n = offs_n >= (N_CTX - shift_size)
                
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                region_mask = region_m[:, None] == region_n[None, :]
                valid_mask = causal_mask & region_mask
                
                qk = tl.where(valid_mask, qk, float("-inf"))

                m_i_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.exp(m_i - m_i_new)
                p_weights = tl.exp(qk - m_i_new[:, None])

                acc = acc * alpha[:, None] + tl.dot(p_weights.to(tl.float16), v)
                l_i = l_i * alpha + tl.sum(p_weights, 1)
                m_i = m_i_new

            k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
            v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

        acc = acc / l_i[:, None]
        tl.store(o_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))



    @triton.jit
    def _ternary_ultrametric_fwd_kernel(
        Q, K, V, K_scale, V_scale, sm_scale, router_indices, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_ksz, stride_ksh, stride_ksn,
        stride_vsz, stride_vsh, stride_vsn,
        stride_oz, stride_oh, stride_om, stride_on,
        stride_rz, stride_rh, stride_rm, stride_rd,
        req_depth, shift_size, Z, H, N_CTX,
        LOCAL_WINDOW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
        P_ARY: tl.constexpr, TREE_DEPTH_P2: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        ks_offset = off_z * stride_ksz + off_h * stride_ksh
        vs_offset = off_z * stride_vsz + off_h * stride_vsh
        o_offset = off_z * stride_oz + off_h * stride_oh
        r_offset = off_z * stride_rz + off_h * stride_rh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_d_packed = offs_d // 16
        shift_amt = (offs_d % 16) * 2

        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on

        depth_offsets = tl.arange(0, TREE_DEPTH_P2)
        m_router_ptrs = router_indices + r_offset + start_m * stride_rm + depth_offsets * stride_rd
        m_routing_vec = tl.load(m_router_ptrs)

        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        num_n_blocks = start_m + 1
        region_m = offs_m >= (N_CTX - shift_size)

        for start_n_block in range(0, num_n_blocks):
            n_router_ptrs = router_indices + r_offset + start_n_block * stride_rm + depth_offsets * stride_rd
            n_routing_vec = tl.load(n_router_ptrs)

            mismatch = (m_routing_vec != n_routing_vec) & (depth_offsets < req_depth)
            has_mismatch = tl.max(mismatch.to(tl.int32), axis=0)

            is_sink = (start_n_block == 0)
            is_local = (start_m - start_n_block) <= LOCAL_WINDOW_BLOCKS

            if has_mismatch == 0 or is_sink or is_local:
                offs_n = start_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                
                # Load packed K and scale
                k_ptrs = K + k_offset + offs_d_packed[:, None] * stride_kk + offs_n[None, :] * stride_kn
                k_packed = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX, other=0)
                ks_ptrs = K_scale + ks_offset + offs_n * stride_ksn
                k_scale = tl.load(ks_ptrs, mask=offs_n < N_CTX, other=0.0)

                # Unpack K: 2 -> -1.0, 1 -> 1.0, 0 -> 0.0
                k_val = (k_packed >> shift_amt[:, None]) & 3
                k_float = tl.where(k_val == 2, -1.0, k_val.to(tl.float32))
                k = (k_float * k_scale[None, :]).to(tl.float16)

                qk = tl.dot(q, k) * sm_scale

                region_n = offs_n >= (N_CTX - shift_size)
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                region_mask = region_m[:, None] == region_n[None, :]
                valid_mask = causal_mask & region_mask
                
                qk = tl.where(valid_mask, qk, float("-inf"))

                m_i_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.exp(m_i - m_i_new)
                p_weights = tl.exp(qk - m_i_new[:, None])

                # Load packed V and scale
                v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d_packed[None, :] * stride_vk
                v_packed = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0)
                vs_ptrs = V_scale + vs_offset + offs_n * stride_vsn
                v_scale = tl.load(vs_ptrs, mask=offs_n < N_CTX, other=0.0)

                # Unpack V
                v_val = (v_packed >> shift_amt[None, :]) & 3
                v_float = tl.where(v_val == 2, -1.0, v_val.to(tl.float32))
                v = (v_float * v_scale[:, None]).to(tl.float16)

                acc = acc * alpha[:, None] + tl.dot(p_weights.to(tl.float16), v)
                l_i = l_i * alpha + tl.sum(p_weights, 1)
                m_i = m_i_new

        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

def routing_to_block_indices(
    assignments: torch.Tensor,
    seq_len: int,
    block_size: int = 128,
) -> torch.Tensor:
    B, H, S, L, P = assignments.shape
    num_blocks = math.ceil(seq_len / block_size)

    branch_ids = assignments.argmax(dim=-1)

    if S < num_blocks * block_size:
        pad_len = num_blocks * block_size - S
        branch_ids = torch.nn.functional.pad(branch_ids, (0, 0, 0, pad_len), value=0)

    branch_ids = branch_ids.view(B, H, num_blocks, block_size, L)
    router_indices = branch_ids[:, :, :, 0, :]  

    return router_indices.to(torch.int32)


def ultrametric_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    router_indices: torch.Tensor,
    local_window: int = 128,
    req_depth: int = 2,
    shift_size: int = 0,
    p: int = 2,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")

    assert q.dtype == torch.float16, f"Triton kernel requires float16, got {q.dtype}"
    assert q.is_cuda, "Triton kernel requires CUDA tensors"

    Z, H, N_CTX, DMODEL = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    router_indices = router_indices.contiguous().to(torch.int32)

    TREE_DEPTH = router_indices.shape[-1]
    assert 0 <= req_depth <= TREE_DEPTH, (
        f"req_depth ({req_depth}) must be in [0, {TREE_DEPTH}]"
    )

    TREE_DEPTH_P2 = 1 << (TREE_DEPTH - 1).bit_length() if TREE_DEPTH > 1 else 1

    BLOCK_M = min(128, N_CTX)
    BLOCK_N = min(128, N_CTX)
    BLOCK_DMODEL = DMODEL
    
    LOCAL_WINDOW_BLOCKS = math.ceil(local_window / BLOCK_N)

    num_blocks_needed = triton.cdiv(N_CTX, BLOCK_M)
    num_blocks_have = router_indices.shape[2]
    if num_blocks_have < num_blocks_needed:
        pad = torch.zeros(
            (Z, H, num_blocks_needed - num_blocks_have, TREE_DEPTH),
            dtype=torch.int32,
            device=router_indices.device,
        )
        router_indices = torch.cat([router_indices, pad], dim=2)

    if TREE_DEPTH < TREE_DEPTH_P2:
        depth_pad = torch.zeros(
            (*router_indices.shape[:-1], TREE_DEPTH_P2 - TREE_DEPTH),
            dtype=torch.int32,
            device=router_indices.device,
        )
        router_indices = torch.cat([router_indices, depth_pad], dim=-1)

    router_indices = router_indices.contiguous()
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(DMODEL)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _ultrametric_fwd_kernel[grid](
        q, k, v, sm_scale, router_indices, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        router_indices.stride(0), router_indices.stride(1), router_indices.stride(2), router_indices.stride(3),
        req_depth, shift_size, Z, H, N_CTX,
        LOCAL_WINDOW_BLOCKS=LOCAL_WINDOW_BLOCKS,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        P_ARY=p, TREE_DEPTH_P2=TREE_DEPTH_P2,
        num_warps=4, num_stages=3,
    )

    return out


def ternary_ultrametric_attention_triton(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    router_indices: torch.Tensor,
    local_window: int = 128,
    req_depth: int = 2,
    shift_size: int = 0,
    p: int = 2,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")

    assert q.dtype == torch.float16, f"Triton kernel requires float16 Q, got {q.dtype}"
    assert k_packed.dtype == torch.int32, f"Triton kernel requires int32 packed K, got {k_packed.dtype}"

    Z, H, N_CTX, DMODEL = q.shape

    q = q.contiguous()
    k_packed = k_packed.contiguous()
    v_packed = v_packed.contiguous()
    k_scale = k_scale.contiguous()
    v_scale = v_scale.contiguous()
    router_indices = router_indices.contiguous().to(torch.int32)

    TREE_DEPTH = router_indices.shape[-1]
    TREE_DEPTH_P2 = 1 << (TREE_DEPTH - 1).bit_length() if TREE_DEPTH > 1 else 1

    BLOCK_M = min(128, N_CTX)
    BLOCK_N = min(128, N_CTX)
    BLOCK_DMODEL = DMODEL
    
    LOCAL_WINDOW_BLOCKS = math.ceil(local_window / BLOCK_N)

    num_blocks_needed = triton.cdiv(N_CTX, BLOCK_M)
    num_blocks_have = router_indices.shape[2]
    if num_blocks_have < num_blocks_needed:
        pad = torch.zeros(
            (Z, H, num_blocks_needed - num_blocks_have, TREE_DEPTH),
            dtype=torch.int32,
            device=router_indices.device,
        )
        router_indices = torch.cat([router_indices, pad], dim=2)

    if TREE_DEPTH < TREE_DEPTH_P2:
        depth_pad = torch.zeros(
            (*router_indices.shape[:-1], TREE_DEPTH_P2 - TREE_DEPTH),
            dtype=torch.int32,
            device=router_indices.device,
        )
        router_indices = torch.cat([router_indices, depth_pad], dim=-1)

    router_indices = router_indices.contiguous()
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(DMODEL)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _ternary_ultrametric_fwd_kernel[grid](
        q, k_packed, v_packed, k_scale, v_scale, sm_scale, router_indices, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        router_indices.stride(0), router_indices.stride(1), router_indices.stride(2), router_indices.stride(3),
        req_depth, shift_size, Z, H, N_CTX,
        LOCAL_WINDOW_BLOCKS=LOCAL_WINDOW_BLOCKS,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        P_ARY=p, TREE_DEPTH_P2=TREE_DEPTH_P2,
        num_warps=4, num_stages=3,
    )

    return out
class CurriculumSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, router_indices, req_depth=2, p=2, use_sparse_backend=False):
        if use_sparse_backend and HAS_TRITON:
            o = ultrametric_attention_triton(q, k, v, router_indices, req_depth=req_depth, p=p)
            ctx.save_for_backward(q, k, v, o, router_indices)
            ctx.req_depth = req_depth
            ctx.p = p
            ctx.use_sparse_backend = True
            return o
        else:
            raise NotImplementedError("Fallback not supported in autograd wrapper")

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not implemented for V3 Triton kernel yet")

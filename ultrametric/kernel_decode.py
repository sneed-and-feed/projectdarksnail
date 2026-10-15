import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def _next_power_of_2(n: int) -> int:
    if HAS_TRITON and hasattr(triton, "next_power_of_2"):
        return int(triton.next_power_of_2(max(n, 1)))
    p = 1
    while p < max(n, 1):
        p <<= 1
    return p


if HAS_TRITON:
    @triton.jit
    def _ultrametric_decode_kernel(
        Q, K_cache, V_cache,
        block_tables, context_lens,
        router_indices, q_router_indices,
        Out,
        stride_q_b, stride_q_h, stride_q_d,
        stride_k_b, stride_k_h, stride_k_bs, stride_k_d,
        stride_v_b, stride_v_h, stride_v_bs, stride_v_d,
        stride_bt_b, stride_bt_l,
        stride_ri_b, stride_ri_h, stride_ri_d,
        stride_qri_b, stride_qri_h, stride_qri_d,
        stride_o_b, stride_o_h, stride_o_d,
        sm_scale,
        req_depth,
        local_blocks,
        SINK_BLOCK: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        TREE_DEPTH: tl.constexpr,
        TREE_DEPTH_P2: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        # Context length for this batch element
        ctx_len = tl.load(context_lens + batch_idx)
        if ctx_len == 0:
            return

        # Load query vector
        q_offset = batch_idx * stride_q_b + head_idx * stride_q_h
        d_offsets = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + q_offset + d_offsets * stride_q_d
        q = tl.load(q_ptrs)

        # Load query routing vector (masked for non-power-of-two TREE_DEPTH)
        q_ri_offset = batch_idx * stride_qri_b + head_idx * stride_qri_h
        tree_offsets = tl.arange(0, TREE_DEPTH_P2)
        valid_tree = tree_offsets < TREE_DEPTH
        q_ri_ptrs = q_router_indices + q_ri_offset + tree_offsets * stride_qri_d
        q_routing_vec = tl.load(q_ri_ptrs, mask=valid_tree, other=0)

        # Initialize Online Softmax variables
        m_i = tl.zeros([1], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_logical_blocks = tl.cdiv(ctx_len, BLOCK_SIZE)

        for logical_idx in range(num_logical_blocks):
            # Load physical block index
            bt_ptr = block_tables + batch_idx * stride_bt_b + logical_idx * stride_bt_l
            physical_block_idx = tl.load(bt_ptr)

            # Load block routing vector
            kv_ri_offset = physical_block_idx * stride_ri_b + head_idx * stride_ri_h
            kv_ri_ptrs = router_indices + kv_ri_offset + tree_offsets * stride_ri_d
            kv_routing_vec = tl.load(kv_ri_ptrs, mask=valid_tree, other=0)

            # Topology check (prefix match up to req_depth, plus optional sink block & local window blocks)
            active_level = (tree_offsets < req_depth) & valid_tree
            mismatch = (q_routing_vec != kv_routing_vec) & active_level
            has_mismatch = tl.max(mismatch.to(tl.int32), axis=0)

            is_sink = SINK_BLOCK and (logical_idx == 0)
            is_local = (local_blocks > 0) and (logical_idx >= (num_logical_blocks - local_blocks))

            if (has_mismatch == 0) or is_sink or is_local:
                # Load K and V
                k_offset = physical_block_idx * stride_k_b + head_idx * stride_k_h
                v_offset = physical_block_idx * stride_v_b + head_idx * stride_v_h

                bs_offsets = tl.arange(0, BLOCK_SIZE)

                k_ptrs = K_cache + k_offset + bs_offsets[:, None] * stride_k_bs + d_offsets[None, :] * stride_k_d
                v_ptrs = V_cache + v_offset + bs_offsets[:, None] * stride_v_bs + d_offsets[None, :] * stride_v_d

                # Mask for the last block
                seq_offsets = logical_idx * BLOCK_SIZE + bs_offsets
                mask = seq_offsets < ctx_len

                k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)

                qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
                qk = tl.where(mask, qk, float('-inf'))

                # Online Softmax
                m_i_new = tl.maximum(m_i, tl.max(qk, axis=0))
                alpha = tl.exp(m_i - m_i_new)
                p = tl.exp(qk - m_i_new)

                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                m_i = m_i_new

        # Epilogue
        out_offset = batch_idx * stride_o_b + head_idx * stride_o_h
        out_ptrs = Out + out_offset + d_offsets * stride_o_d
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(out_ptrs, acc / safe_l)
else:
    _ultrametric_decode_kernel = None


def _ultrametric_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    router_indices: torch.Tensor,
    q_router_indices: torch.Tensor,
    req_depth: int,
    local_blocks: int = 0,
    sink_block: bool = False,
) -> torch.Tensor:
    """Exact PyTorch reference fallback for CPU execution when Triton CUDA/interpret is inactive."""
    batch_size, num_heads, head_dim = q.shape
    tree_depth = q_router_indices.shape[-1]
    r = min(max(int(req_depth), 0), tree_depth)
    block_size = k_cache.shape[2]
    sm_scale = 1.0 / (head_dim ** 0.5)
    out = torch.zeros_like(q)

    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        if ctx_len <= 0:
            continue
        num_logical = (ctx_len + block_size - 1) // block_size
        phys_ids = block_tables[b, :num_logical].long()
        for h in range(num_heads):
            q_vec = q[b, h].float()  # (D,)
            q_route = q_router_indices[b, h, :r]  # (r,)
            kv_routes = router_indices[phys_ids, h, :r]  # (num_logical, r)
            if r > 0:
                match = (kv_routes == q_route.unsqueeze(0)).all(dim=-1)
            else:
                match = torch.ones(num_logical, dtype=torch.bool, device=q.device)
            if sink_block and num_logical > 0:
                match[0] = True
            if local_blocks > 0:
                match[max(0, num_logical - local_blocks):] = True

            if not match.any():
                continue

            k_sel = []
            v_sel = []
            for l_idx in range(num_logical):
                if not match[l_idx]:
                    continue
                p_idx = int(phys_ids[l_idx].item())
                valid_tokens = min(block_size, ctx_len - l_idx * block_size)
                k_sel.append(k_cache[p_idx, h, :valid_tokens].float())
                v_sel.append(v_cache[p_idx, h, :valid_tokens].float())
            k_cat = torch.cat(k_sel, dim=0)
            v_cat = torch.cat(v_sel, dim=0)
            scores = torch.matmul(k_cat, q_vec) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = torch.matmul(probs, v_cat).to(q.dtype)
    return out


def ultrametric_decode(
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    router_indices,
    q_router_indices,
    req_depth,
    local_blocks: int = 0,
    sink_block: bool = False,
):
    import os
    batch_size, num_heads, head_dim = q.shape
    tree_depth = int(q_router_indices.shape[-1])
    assert tree_depth >= 1, "tree_depth must be >= 1"

    interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"
    if (not HAS_TRITON) or (not q.is_cuda and not interpret):
        return _ultrametric_decode_reference(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            router_indices,
            q_router_indices,
            req_depth,
            local_blocks=local_blocks,
            sink_block=sink_block,
        )

    TREE_DEPTH_P2 = _next_power_of_2(tree_depth)
    out = torch.empty_like(q)
    sm_scale = 1.0 / (head_dim ** 0.5)
    BLOCK_SIZE = k_cache.shape[2]
    grid = (batch_size, num_heads)

    _ultrametric_decode_kernel[grid](
        q, k_cache, v_cache,
        block_tables, context_lens,
        router_indices, q_router_indices,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        router_indices.stride(0), router_indices.stride(1), router_indices.stride(2),
        q_router_indices.stride(0), q_router_indices.stride(1), q_router_indices.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        sm_scale,
        req_depth,
        local_blocks,
        SINK_BLOCK=bool(sink_block),
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=head_dim,
        TREE_DEPTH=tree_depth,
        TREE_DEPTH_P2=TREE_DEPTH_P2,
    )
    return out

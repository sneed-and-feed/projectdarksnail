import os
import torch
import triton
import triton.language as tl

@triton.jit
def fused_eaas_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, r_ptr, cg_ptr, out_ptr,
    row_ptr, col_idx,
    num_nodes, num_heads, head_dim: tl.constexpr,
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_rn, stride_rd,
    stride_on, stride_oh, stride_od,
    BLOCK_D: tl.constexpr,
):
    node_i = tl.program_id(0)
    head_h = tl.program_id(1)
    
    if node_i >= num_nodes:
        return
        
    # load query
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    q_offs = node_i * stride_qn + head_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptr + q_offs, mask=mask_d, other=0.0)
    
    # row ptrs
    start_idx = tl.load(row_ptr + node_i)
    end_idx = tl.load(row_ptr + node_i + 1)
    
    # online softmax variables
    m_i = -float('inf')
    l_i = 0.0
    
    # accumulator for EAAS values
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # CG coefficients (simulated as per-dimension scaling for even L_Sigma EAAS operator)
    cg = tl.load(cg_ptr + offs_d, mask=mask_d, other=1.0)
    
    for idx in range(start_idx, end_idx):
        node_j = tl.load(col_idx + idx)
        
        # load key
        k_offs = node_j * stride_kn + head_h * stride_kh + offs_d * stride_kd
        k = tl.load(k_ptr + k_offs, mask=mask_d, other=0.0)
        
        # compute score
        s_ij = tl.sum(q * k, axis=0)
        
        # load r_j (geometry feature or Wigner-D aligned feature multiplier)
        r_offs = node_j * stride_rn + offs_d * stride_rd
        r_j = tl.load(r_ptr + r_offs, mask=mask_d, other=1.0)
        
        # load value
        v_offs = node_j * stride_vn + head_h * stride_vh + offs_d * stride_vd
        v = tl.load(v_ptr + v_offs, mask=mask_d, other=0.0)
        
        # --- EAAS OPERATION (Source Fusion) ---
        # 1. Align/Transform: h'_j = v * r_j (representing \tilde{h} = h D_R)
        # 2. Re-index / CG scale: P(\tilde{h}) = cg * h'_j
        v_eaas = cg * (v * r_j)
        
        # --- Online Softmax Accumulation ---
        m_ij = tl.maximum(m_i, s_ij)
        alpha = tl.exp(s_ij - m_ij)
        beta = tl.exp(m_i - m_ij)
        
        l_i = l_i * beta + alpha
        acc = acc * beta + v_eaas * alpha
        m_i = m_ij

    # Target Fusion: apply target geometry r_i
    r_i_offs = node_i * stride_rn + offs_d * stride_rd
    r_i = tl.load(r_ptr + r_i_offs, mask=mask_d, other=1.0)
    
    # normalize
    acc = acc / l_i
    
    # Target-term coupling: m_i \otimes R(r_i)
    out = acc * r_i
    
    # write output
    out_offs = node_i * stride_on + head_h * stride_oh + offs_d * stride_od
    tl.store(out_ptr + out_offs, out, mask=mask_d)


def run_eaas_attention(q, k, v, r, cg, row_ptr, col_idx):
    num_nodes, num_heads, head_dim = q.shape
    
    out = torch.empty_like(q)
    
    # Pad to nearest power of 2 for block size
    BLOCK_D = 1
    while BLOCK_D < head_dim:
        BLOCK_D *= 2
    
    grid = (num_nodes, num_heads)
    
    fused_eaas_attention_fwd_kernel[grid](
        q, k, v, r, cg, out,
        row_ptr, col_idx,
        num_nodes, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        r.stride(0), r.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=BLOCK_D,
    )
    
    return out

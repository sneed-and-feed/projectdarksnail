"""
Ultrametric AI — PyTorch / Triton Stack

True Fractal attention with dynamic Gumbel-Softmax routing,
Triton block-sparse GPU kernels, and Holographic Reasoning Tokens.
"""

from .topology import (
    DynamicTopologyRouter,
    get_ultrametric_mask,
    get_dynamic_ultrametric_mask,
    compute_p_adic_distance,
)
from .layer import (
    UltrametricAttention,
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    get_tree_adjacency_mask,
)
from .model import (
    UltrametricTransformerBlock,
    UltrametricTransformer,
)
from .kernel import (
    HAS_TRITON,
    routing_to_block_indices,
    ultrametric_attention_triton,
    block_sparse_attention,
)
from .block_list import (
    BlockLists,
    build_block_lists,
    build_transposed_block_lists,
    block_list_attention,
    routed_block_attention,
)

__all__ = [
    # Topology & Routing
    "DynamicTopologyRouter",
    "get_ultrametric_mask",
    "get_dynamic_ultrametric_mask",
    "compute_p_adic_distance",
    # Attention Layer
    "UltrametricAttention",
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
    "get_tree_adjacency_mask",
    # Model
    "UltrametricTransformerBlock",
    "UltrametricTransformer",
    # Kernel utilities
    "HAS_TRITON",
    "routing_to_block_indices",
    "ultrametric_attention_triton",
    "block_sparse_attention",
    # Block-list kernel (runtime-length per-row lists)
    "BlockLists",
    "build_block_lists",
    "build_transposed_block_lists",
    "block_list_attention",
    "routed_block_attention",
]

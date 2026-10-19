# DARKSNAIL V2 -> GEMMA 4 31B SURGICAL HOOK SPECIFICATION
# Target Backbone: google/gemma-4-31B (60 layers, 5:1 Local/Global interleave)
# Global Layer Indices: [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]

import torch
import torch.nn as nn

class DarkSnailGemma4GlobalHook(nn.Module):
    """
    Hooks strictly into the 10 Global p-RoPE layers of a frozen Gemma 4 31B backbone.
    Leaves the 50 Local Sliding-Window (1024-token) layers untouched.
    """
    def __init__(self, head_dim: int, d_model: int, bridge_dim: int = 64, p_rope_ratio: float = 0.25):
        super().__init__()
        self.d_rope = int(head_dim * p_rope_ratio)       # First 25% channels: Rotary phase entangled
        self.d_nope = head_dim - self.d_rope             # Last 75% channels: Pure position-invariant content
        
        # 1. brain-ai Lateralized Callosal Bridge (operates on residual stream + unrotated subspace)
        self.callosal_down = nn.Linear(d_model, bridge_dim, bias=False)
        self.callosal_up   = nn.Linear(bridge_dim, d_model, bias=False)
        
        # 2. Skew-Symmetric Triad Operator (J = A - A^T in 64-D continuous manifold)
        self.triad_gen = nn.Parameter(torch.randn(bridge_dim, bridge_dim) * 0.02)

    def get_skew_triad(self) -> torch.Tensor:
        return self.triad_gen - self.triad_gen.transpose(-1, -2)

    def forward_global_kv_hook(
        self, 
        k_global: torch.Tensor,   # Shape: [B, H_kv, N, head_dim] (Note: V == K in Gemma 4 Global layers)
        hidden_states: torch.Tensor, # Shape: [B, N, d_model]
        dt: float = 0.1,
        steps: int = 4
    ):
        # --- STEP 1: Split p-RoPE Rotated (25%) vs Unrotated Content (75%) Subspaces ---
        k_rope = k_global[..., :self.d_rope]   # Preserve exact 1M-base positional phase
        k_nope = k_global[..., self.d_rope:]   # Pure semantic/spatial channels for DarkSnail surgery

        # --- STEP 2: sparse-ai Ultrametric / Spectral Distractor Score on k_nope ---
        # Compute spectral energy strictly on position-invariant channels to avoid distance bias
        spectral_salience = torch.norm(k_nope, p=2, dim=-1).mean(dim=1)  # [B, N]
        keep_mask = self.compute_ultrametric_mask(spectral_salience)     # [B, N] boolean/soft gate

        # Apply mask once -> automatically prunes BOTH K and V because V = K on Global layers!
        k_pruned = k_global * keep_mask.unsqueeze(1).unsqueeze(-1)
        v_pruned = k_pruned  # Zero extra memory copy; inherited by downstream Shared-KV layers

        # --- STEP 3: brain-ai 64-D Callosal Bridge + Continuous System Dynamics ---
        z = self.callosal_down(hidden_states)  # Project into 64-D bridge: [B, N, 64]
        J = self.get_skew_triad()              # Skew-symmetric conservation operator [64, 64]
        
        # Multi-step continuous field integration in 64-D latent space
        for _ in range(steps):
            dz = torch.matmul(z, J) * keep_mask.unsqueeze(-1)
            z = z + dt * torch.tanh(dz)

        # Inject continuous spatial state back into frozen residual stream
        delta_h = self.callosal_up(z)
        return k_pruned, v_pruned, hidden_states + delta_h, keep_mask
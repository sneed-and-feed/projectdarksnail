"""
Ultrametric AI — Complete Transformer Model (PyTorch)

A full, trainable language model built on fractal p-adic attention.
Supports causal language modeling with:
  - Token + learned position embeddings
  - N stacked UltrametricTransformerBlocks with RoPE
  - Dynamic per-head routing via DynamicTopologyRouter
  - Tied embedding/LM-head weights
  - Autoregressive generation (greedy, top-k, top-p sampling)

Usage:
    model = UltrametricTransformer(vocab_size=32000, num_layers=6)
    logits, loss = model(input_ids, targets=targets)
    generated = model.generate(prompt_ids, max_new_tokens=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .layer import UltrametricAttention
from .topology import DynamicTopologyRouter


class UltrametricTransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with sparse p-adic UltrametricAttention.

    Supports three attention modes (dense, chunked-sparse, triton) and
    optional Holographic Reasoning Tokens (interior tree nodes that
    facilitate hierarchical message passing).

    Args:
        embed_dim: model dimension
        num_heads: number of attention heads
        p: tree arity for Bruhat-Tits topology
        mlp_ratio: MLP hidden dim multiplier
        max_seq_len: max sequence length for RoPE
        dropout: residual and attention dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        p: int = 2,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim

        # Learnable interior token for Holographic Reasoning Tokens
        self.interior_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = UltrametricAttention(
            embed_dim, num_heads, p, max_seq_len=max_seq_len, dropout=dropout
        )

        self.ln_2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        routing: Optional[torch.Tensor] = None,
        routing_assignments: Optional[torch.Tensor] = None,
        mode: str = "auto",
        use_interior: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: optional pre-computed attention mask
            routing: routing paths for tree adjacency mask
            routing_assignments: per-head routing for Triton/chunked paths
            mode: attention execution mode
            use_interior: whether to prepend Holographic Reasoning Tokens

        Returns:
            out: (batch, seq_len, embed_dim) — interior tokens stripped if used
        """
        batch_size, seq_len, _ = x.size()
        num_interior = 0

        if use_interior:
            levels = int(math.ceil(math.log(max(seq_len, 2), self.p)))
            pad_len = self.p**levels
            num_interior = (pad_len - 1) // (self.p - 1)
            interior_nodes = self.interior_token.expand(
                batch_size, num_interior, self.embed_dim
            )
            x_full = torch.cat([interior_nodes, x], dim=1)
        else:
            x_full = x

        # Pre-LN Attention
        h = self.ln_1(x_full)
        attn_out = self.attn(
            h,
            num_interior=num_interior,
            dynamic_mask=mask,
            routing=routing,
            routing_assignments=routing_assignments,
            mode=mode,
        )
        x_full = x_full + attn_out

        # Pre-LN MLP
        x_full = x_full + self.mlp(self.ln_2(x_full))

        # Strip interior tokens, return only sequence tokens
        if use_interior and num_interior > 0:
            return x_full[:, num_interior:, :]
        return x_full


class UltrametricTransformer(nn.Module):
    """
    Complete Ultrametric Transformer language model.

    A GPT-style autoregressive model where standard dense self-attention
    is replaced by hierarchical block-sparse attention on a Bruhat-Tits tree.

    Architecture:
        Input IDs → Token Embedding → [N × UltrametricTransformerBlock] → LayerNorm → LM Head

    The DynamicTopologyRouter produces per-head routing paths that determine
    which tokens attend to which. Routing is learned end-to-end via
    Gumbel-Softmax with temperature annealing.

    Args:
        vocab_size: vocabulary size
        num_layers: number of transformer blocks
        embed_dim: model embedding dimension
        num_heads: number of attention heads
        p: tree arity (2 = binary, 3 = ternary, etc.)
        max_seq_len: maximum sequence length
        mlp_ratio: MLP hidden dimension multiplier
        dropout: dropout probability
        tie_weights: whether to tie embedding and LM head weights
        use_interior: whether to use Holographic Reasoning Tokens
        attn_mode: default attention execution mode
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        num_layers: int = 6,
        embed_dim: int = 512,
        num_heads: int = 8,
        p: int = 2,
        max_seq_len: int = 2048,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        tie_weights: bool = True,
        use_interior: bool = False,
        attn_mode: str = "auto",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.p = p
        self.use_interior = use_interior
        self.attn_mode = attn_mode

        # Token embedding (no learned position embedding — RoPE handles positions)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Dynamic routing (one per layer)
        self.routers = nn.ModuleList([
            DynamicTopologyRouter(
                embed_dim=embed_dim,
                seq_len=max_seq_len,
                num_heads=num_heads,
                p=p,
                tau=1.0,
                hard=True,
            ) for _ in range(num_layers)
        ])

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                UltrametricTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    p=p,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language model head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights: embedding and LM head share parameters
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        tau_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with optional language modeling loss.

        Args:
            input_ids: (batch, seq_len) integer token IDs
            targets: (batch, seq_len) target token IDs for loss computation.
                     If None, loss is not computed.
            tau_override: optional Gumbel-Softmax temperature for the router

        Returns:
            logits: (batch, seq_len, vocab_size) predicted logit distribution
            loss: scalar cross-entropy loss (None if targets not provided)
            aux_loss: scalar routing load-balance loss (for training)
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )

        # Token embeddings (RoPE applied inside attention layers)
        x = self.tok_emb(input_ids)  # (batch, seq_len, embed_dim)
        x = self.emb_dropout(x)

        # Build causal mask
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.float32, device=x.device)
        ).unsqueeze(0).unsqueeze(0) # (1, 1, S, S)

        total_aux_loss = 0.0
        self.layer_routings = []
        from .topology import get_dynamic_ultrametric_mask

        for router, block in zip(self.routers, self.blocks):
            routing_assignments, layer_aux_loss = router(x, tau_override=tau_override)
            total_aux_loss = total_aux_loss + layer_aux_loss
            self.layer_routings.append(routing_assignments)
            
            # Use dynamic mask with local window for all non-triton modes
            dyn_mask = get_dynamic_ultrametric_mask(
                routing_assignments, p=self.p, local_window=32
            ).to(x.device)
            layer_mask = dyn_mask * causal
            
            x = block(
                x,
                mask=layer_mask,
                routing_assignments=routing_assignments,
                mode=self.attn_mode,
                use_interior=self.use_interior,
            )

        aux_loss = total_aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # Compute cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            input_ids: (batch, prompt_len) starting token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = neutral, <1 = greedy, >1 = creative)
            top_k: top-k sampling (None to disable)
            top_p: nucleus sampling threshold (None to disable)
            eos_token_id: stop generation at this token (None to disable)

        Returns:
            output_ids: (batch, prompt_len + generated_len) complete sequence
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            context = generated[:, -self.max_seq_len :]

            # Forward pass (no loss)
            logits, _, _ = self.forward(context)

            # Get logits for the last position
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True
                )
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                # Scatter back
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample from the distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS if specified
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"UltrametricTransformer(\n"
            f"  vocab_size={self.vocab_size}, "
            f"layers={self.num_layers}, "
            f"dim={self.embed_dim}, "
            f"heads={self.num_heads}, "
            f"p={self.p}\n"
            f"  params={params:,} ({params/1e6:.1f}M)\n"
            f"  max_seq_len={self.max_seq_len}, "
            f"mode={self.attn_mode}\n"
            f")"
        )

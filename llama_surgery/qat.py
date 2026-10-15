import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union

# ==============================================================================
# 1. Straight-Through Estimator (STE) Fake Quantization
# ==============================================================================

class FakeQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for low-bit symmetric quantization.
    Forward pass: quantizes to target bits using the provided scale and clamps.
    Backward pass: bypasses rounding by copying input gradients straight through.
    """
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x, bits, scale, min_val, max_val):
        x_f32 = x.float()
        scale_f32 = scale.float().clamp(min=1e-5) # Prevent division by zero
        
        # Round-to-nearest and clamp to boundary
        x_q = torch.round(x_f32 / scale_f32).clamp(min_val, max_val)
        x_dq = x_q * scale_f32
        return x_dq.to(x.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        # STE: Pass the gradient back unmodified for the quantized input,
        # and None for non-trainable args (bits, scale, min_val, max_val)
        return grad_output, None, None, None, None


# ==============================================================================
# 2. QAT Linear Layer Wrapper
# ==============================================================================

class QATLinear(nn.Module):
    """
    A Quantization-Aware Training wrapper for nn.Linear.
    Supports channel-wise scaling, static activation tracking (EMA), and custom bitwidths.
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        bits: int = 4,
        static_activations: bool = False,
        channel_wise: bool = True,
        quantize_weights: bool = True,
        quantize_activations: bool = True
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight # Shares parameters
        self.bias = original_linear.bias     # Shares bias parameter
        
        self.bits = bits
        self.static_activations = static_activations
        self.channel_wise = channel_wise
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        
        self.q_min = - (2 ** (bits - 1))
        self.q_max = (2 ** (bits - 1)) - 1
        
        # Static Activations (EMA scale tracking)
        if self.static_activations and self.quantize_activations:
            scale_shape = (1, 1, self.in_features) if self.channel_wise else (1,)
            self.register_buffer("running_scale", torch.ones(scale_shape))
            self.momentum = 0.99
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_qat = self.weight
        
        # Weight Quantization
        if self.quantize_weights:
            if self.channel_wise:
                # W shape: (out_features, in_features)
                # Compute scale per output channel (dim=1)
                w_scale = self.weight.abs().max(dim=1, keepdim=True).values / self.q_max
            else:
                w_scale = self.weight.abs().max() / self.q_max
            w_qat = FakeQuantizeSTE.apply(self.weight, self.bits, w_scale, self.q_min, self.q_max)
            
        x_qat = x
        
        # Activation/Input Quantization
        if self.quantize_activations:
            if self.training:
                if self.channel_wise:
                    # x shape: (batch, seq_len, in_features)
                    # Compute scale per token/channel (dim=-1)
                    x_scale = x.abs().max(dim=-1, keepdim=True).values / self.q_max
                else:
                    x_scale = x.abs().max() / self.q_max
                    
                if self.static_activations:
                    # Update running scale via EMA during training
                    mean_scale = x_scale.detach().mean(dim=(0, 1), keepdim=True) if self.channel_wise else x_scale.detach()
                    self.running_scale.copy_(self.momentum * self.running_scale + (1 - self.momentum) * mean_scale)
            else:
                if self.static_activations:
                    # Use the pre-calculated frozen scale at inference (Static Activations)
                    x_scale = self.running_scale
                else:
                    if self.channel_wise:
                        x_scale = x.abs().max(dim=-1, keepdim=True).values / self.q_max
                    else:
                        x_scale = x.abs().max() / self.q_max
                        
            x_qat = FakeQuantizeSTE.apply(x, self.bits, x_scale, self.q_min, self.q_max)
            
        return F.linear(x_qat, w_qat, self.bias)


# ==============================================================================
# 3. Model Injection Utility
# ==============================================================================

def inject_qat(
    model: nn.Module,
    bits: int = 4,
    static_activations: bool = False,
    channel_wise: bool = True,
    quantize_weights: bool = True,
    quantize_activations: bool = True,
    qat_config: Optional[Dict[str, Union[int, Dict[str, bool]]]] = None
) -> nn.Module:
    """
    Surgically replaces nn.Linear layers in the model with QATLinear wrappers.
    
    Args:
        model: The PyTorch model (e.g. LlamaForCausalLM / GemmaForCausalLM).
        bits: Global quantization bitwidth (default: 4).
        static_activations: Pre-calculate scaling factors (Gemma 4 QAT style).
        channel_wise: Perform channel/row-wise scaling to preserve accuracy.
        quantize_weights: Enable weight quantization.
        quantize_activations: Enable input activation quantization.
        qat_config: Optional dict mapping layer names/patterns to custom config.
                    Example: {
                        "lm_head": 2,          # 2-bit targeted projection
                        "o_proj": {"bits": 2}, # Custom args
                        "router": {"quantize_weights": False} # Skip router
                    }
    """
    
    def _replace_recursive(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Resolve layer-specific configurations
            layer_bits = bits
            layer_static = static_activations
            layer_channel = channel_wise
            layer_w_q = quantize_weights
            layer_act_q = quantize_activations
            
            if qat_config is not None:
                # Direct match or partial string matching
                match_pattern = None
                for pattern in qat_config.keys():
                    if pattern in full_name:
                        match_pattern = pattern
                        break
                        
                if match_pattern is not None:
                    cfg = qat_config[match_pattern]
                    if isinstance(cfg, int):
                        layer_bits = cfg
                    elif isinstance(cfg, dict):
                        layer_bits = cfg.get("bits", bits)
                        layer_static = cfg.get("static_activations", static_activations)
                        layer_channel = cfg.get("channel_wise", channel_wise)
                        layer_w_q = cfg.get("quantize_weights", quantize_weights)
                        layer_act_q = cfg.get("quantize_activations", quantize_activations)
            
            if isinstance(child, nn.Linear):
                # Skip dynamically generated layers if needed, otherwise wrap
                if "router" in full_name:
                    # Routers need full precision to preserve routing integrity
                    continue
                    
                qat_linear = QATLinear(
                    original_linear=child,
                    bits=layer_bits,
                    static_activations=layer_static,
                    channel_wise=layer_channel,
                    quantize_weights=layer_w_q,
                    quantize_activations=layer_act_q
                )
                setattr(module, name, qat_linear)
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model)
    
    # Inject config settings
    model.config.quantize_kv_cache = True
    model.config.kv_cache_bits = bits
    
    print(f"[QAT] Successfully injected Quantization-Aware Training wrappers into model.")
    if qat_config:
        print(f"      Layer configurations applied: {qat_config}")
    return model

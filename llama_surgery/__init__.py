from .llama_patcher import (
    inject_surgery,
    patch_llama_model,
    save_router_sidecar,
    load_router_sidecar,
)
from .surgery import SurgicalLlamaAttention, SurgeryLossRamp
from .surgery_trainer import SurgeryTrainer, TauAnnealingCallback
from .topology import (
    DynamicTopologyRouter,
    MultiPrimeTopologyRouter,
    TriadTopologyRouter,
    compute_expected_block_distance,
)
from .qat import QATLinear, inject_qat, FakeQuantizeSTE
from .adelic_cache import SurprisalAdelicCache, BeltramiAdelicCache, TemporalAdelicCache
from .speculative import SpeculativeSurgicalGenerator, ZenoSpeculativeGenerator

try:
    from .multimodal_injector import MultimodalEncoder, VisionProjection
except Exception:
    MultimodalEncoder = None
    VisionProjection = None

__all__ = [
    "inject_surgery",
    "patch_llama_model",
    "save_router_sidecar",
    "load_router_sidecar",
    "SurgicalLlamaAttention",
    "SurgeryLossRamp",
    "SurgeryTrainer",
    "TauAnnealingCallback",
    "DynamicTopologyRouter",
    "MultiPrimeTopologyRouter",
    "TriadTopologyRouter",
    "compute_expected_block_distance",
    "QATLinear",
    "inject_qat",
    "FakeQuantizeSTE",
    "SurprisalAdelicCache",
    "BeltramiAdelicCache",
    "TemporalAdelicCache",
    "SpeculativeSurgicalGenerator",
    "ZenoSpeculativeGenerator",
    "MultimodalEncoder",
    "VisionProjection",
]

"""
darksnail: Bi-Hemispheric x Ultrametric Triad Synthesis Package for ARC-AGI.

Exports the core modules of Project DarkSnail V2:
- `shell`: `BeltramiCallosalShell`, `compute_grid_prompt_zeno_flux`, `extract_beltrami_lh_latents`
- `readout`: `SkewSymmetricTriadOperator`, `verify_pointwise_orthogonality`,
  `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, `RoPE2DMedoidPooler`,
  `ContinuousSpectralFieldReadout`, `get_continuous_spectral_and_ray_basis`
- `antenna`: `ZenoAmygdalaController`, `DarkSnailSchedule`, `DarkSnailSystem2Adaptor`
"""

__version__ = "0.1.0"

from .antenna import (
    DarkSnailSchedule,
    DarkSnailSystem2Adaptor,
    ZenoAmygdalaController,
)
from .longbench import (
    attach_layer16_beltrami_hook,
    build_niah_context,
    build_qasper_prompt_ids,
    collect_measured_budget,
    evaluate_banal_niah_comparison,
    evaluate_qasper_comparison,
    load_qasper_dataset,
    load_v4_baseline_references,
    prepare_darksnail_llama_model,
    score_qasper_f1,
    set_long_context_condition,
)
from .readout import (
    ContinuousSpectralFieldReadout,
    DarkSnailCanvasReadout,
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    get_continuous_spectral_and_ray_basis,
    verify_pointwise_orthogonality,
)
from .shell import (
    BeltramiCallosalShell,
    PullbackCanvasResult,
    compute_grid_prompt_zeno_flux,
    extract_beltrami_lh_latents,
    extract_beltrami_pullback_canvas,
    extract_grid_beltrami_shape_descriptors,
    infer_arc_output_shape_beltrami,
)

__all__ = [
    "__version__",
    "BeltramiCallosalShell",
    "PullbackCanvasResult",
    "compute_grid_prompt_zeno_flux",
    "extract_beltrami_lh_latents",
    "extract_beltrami_pullback_canvas",
    "extract_grid_beltrami_shape_descriptors",
    "infer_arc_output_shape_beltrami",
    "SkewSymmetricTriadOperator",
    "verify_pointwise_orthogonality",
    "DarkSnailCanvasReadout",
    "SlowFastTriadCascade",
    "RoPE2DMedoidPooler",
    "ContinuousSpectralFieldReadout",
    "get_continuous_spectral_and_ray_basis",
    "DarkSnailSchedule",
    "ZenoAmygdalaController",
    "DarkSnailSystem2Adaptor",
    "attach_layer16_beltrami_hook",
    "build_niah_context",
    "build_qasper_prompt_ids",
    "collect_measured_budget",
    "evaluate_banal_niah_comparison",
    "evaluate_qasper_comparison",
    "load_qasper_dataset",
    "load_v4_baseline_references",
    "prepare_darksnail_llama_model",
    "score_qasper_f1",
    "set_long_context_condition",
]

from .gemma4 import (
    GEMMA4_31B_SPEC,
    GLOBAL_LAYER_INDICES,
    SLIDING_LAYER_INDICES,
    DarkSnailGemma4GlobalHook,
    Gemma4BeltramiCallosalShell,
    Gemma4RMSNorm,
    Gemma4RMSNormNoScale,
    Gemma4_31B_Spec,
    ProportionalRoPESubspaceSplitter,
    SurgicalGemma4Attention,
    apply_gemma4_prope,
    attach_gemma4_beltrami_hook,
    build_gemma4_prope_cache,
    discover_gemma4_decoder_layers,
    is_gemma4_global_layer,
    patch_gemma4_model,
    prepare_darksnail_gemma4_model,
    warmup_gemma4_routers_kl,
)

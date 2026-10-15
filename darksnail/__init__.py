"""
darksnail: Bi-Hemispheric x Ultrametric Triad Synthesis Package for ARC-AGI.

Exports the core modules of Project DarkSnail V2:
- `shell`: `BeltramiCallosalShell`, `compute_grid_prompt_zeno_flux`, `extract_beltrami_lh_latents`
- `readout`: `SkewSymmetricTriadOperator`, `verify_pointwise_orthogonality`,
  `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, `RoPE2DMedoidPooler`
- `antenna`: `ZenoAmygdalaController`, `DarkSnailSchedule`, `DarkSnailSystem2Adaptor`
"""

__version__ = "0.1.0"

from .antenna import (
    DarkSnailSchedule,
    DarkSnailSystem2Adaptor,
    ZenoAmygdalaController,
)
from .readout import (
    DarkSnailCanvasReadout,
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    verify_pointwise_orthogonality,
)
from .shell import (
    BeltramiCallosalShell,
    compute_grid_prompt_zeno_flux,
    extract_beltrami_lh_latents,
)

__all__ = [
    "__version__",
    "BeltramiCallosalShell",
    "compute_grid_prompt_zeno_flux",
    "extract_beltrami_lh_latents",
    "SkewSymmetricTriadOperator",
    "verify_pointwise_orthogonality",
    "DarkSnailCanvasReadout",
    "SlowFastTriadCascade",
    "RoPE2DMedoidPooler",
    "DarkSnailSchedule",
    "ZenoAmygdalaController",
    "DarkSnailSystem2Adaptor",
]

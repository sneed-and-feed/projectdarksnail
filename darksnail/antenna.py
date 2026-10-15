"""
darksnail.antenna: Tri-Signal Zeno-Amygdala Controller & Dynamic System 2 TTA Scheduler.

Implements Upgrade 4 of Project DarkSnail V2:
- Replaces static cosine conflict (`C in [0.441, 0.449]`) and static TTA hyperparameters
  (`num_steps=30, lr=0.05, anchor_weight=0.1, ray_quantile=0.20`) with `ZenoAmygdalaController`.
- Fuses three zero-leakage signals available at Step 0:
    1. Left-Hemisphere Zeno Triad Flux (`zeno_flux` in [0, 1]) & Cophenetic LCA Variance (`lca_variance`),
    2. Step-0 Demonstration Loss (`step0_demo_loss`), Gradient Norm (`step0_grad_norm`),
       and Empirical System 1 Demonstration Fit (`s1_demo_fit` in [0, 1]),
    3. Right-Hemisphere Test-Grid Predictive Entropy (`test_entropy`) & Canvas Area (`canvas_cells`).
- Computes the calibrated Zeno Turbulence Index `T_zeno in [0, 1]` and schedules
  `DarkSnailSchedule(num_steps, lr, anchor_weight, triad_gain, ray_quantiles)`.
- Also provides `DarkSnailSystem2Adaptor`, a drop-in subclass of `System2LatentAdaptor`
  integrating `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, and `ZenoAmygdalaController`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .readout import DarkSnailCanvasReadout, SlowFastTriadCascade
from .shell import compute_grid_prompt_zeno_flux


@dataclass
class DarkSnailSchedule:
    """
    Dynamic System 2 TTA & Velocity Ray Extrapolation Schedule produced by `ZenoAmygdalaController`.
    """

    num_steps: int
    lr: float
    anchor_weight: float
    triad_gain: float
    ray_quantiles: List[float] = field(default_factory=lambda: [0.15])
    turbulence_index: float = 0.5
    regime: str = "balanced_deliberation"
    bypass_tta: bool = False
    asym_gain: float = 1.5
    cascade_gain: float = 1.0

    @property
    def ray_quantile(self) -> float:
        """Primary velocity ray extrapolation quantile (scalar convenience accessor)."""
        return float(self.ray_quantiles[0]) if self.ray_quantiles else 0.15

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ray_quantile"] = self.ray_quantile
        return d

    def __getitem__(self, key: str) -> Any:
        if key == "ray_quantile":
            return self.ray_quantile
        return getattr(self, key)


class ZenoAmygdalaController(nn.Module):
    """
    Tri-Signal Zeno-Amygdala Controller for Dynamic System 2 TTA & Ray Quantile Scheduling.

    Combines Tao's Zeno clock criterion (`SpeculativeSurgicalGenerator._compute_zeno_next_depth`)
    with `ReflexFirstCascadedRouter` to schedule:
      - `num_steps` in `[0, 40]`
      - `lr` in `[0.02, 0.06]`
      - `anchor_weight` in `[0.05, 0.30]`
      - `triad_gain` in `[0.25, 1.5]`
      - `ray_quantiles` including surgical 1-2 cell quantiles (`1/N_cells`, `2/N_cells`)
        when near-solved (`s1_demo_fit >= 0.90` or `step0_demo_loss < 2.5`) or when
        2-color occlusion ties are detected.
    """

    def __init__(
        self,
        min_steps: int = 0,
        max_steps: int = 40,
        min_lr: float = 0.02,
        max_lr: float = 0.06,
        min_anchor: float = 0.05,
        max_anchor: float = 0.30,
        min_triad_gain: float = 0.25,
        max_triad_gain: float = 1.50,
        near_solved_loss_threshold: float = 2.5,
        high_turb_loss_threshold: float = 4.5,
        near_solved_fit_threshold: float = 0.90,
        allow_zero_step_bypass: bool = False,
    ):
        super().__init__()
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.min_anchor = float(min_anchor)
        self.max_anchor = float(max_anchor)
        self.min_triad_gain = float(min_triad_gain)
        self.max_triad_gain = float(max_triad_gain)
        self.near_solved_loss_threshold = float(near_solved_loss_threshold)
        self.high_turb_loss_threshold = float(high_turb_loss_threshold)
        self.near_solved_fit_threshold = float(near_solved_fit_threshold)
        self.allow_zero_step_bypass = bool(allow_zero_step_bypass)

        self.last_turbulence_index: float = 0.0
        self.last_schedule: Optional[DarkSnailSchedule] = None

    @staticmethod
    def compute_predictive_entropy(
        logits: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Computes mean predictive Shannon entropy `H(p)` (in nats) across active canvas tokens.
        """
        logits_f = logits.detach().float()
        if logits_f.dim() == 3 and logits_f.shape[-1] >= 12:
            color_logits = logits_f[..., 2:12]
        elif logits_f.dim() == 2 and logits_f.shape[-1] >= 12:
            color_logits = logits_f[:, 2:12]
        else:
            color_logits = logits_f

        probs = F.softmax(color_logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-12))
        ent = -(probs * log_probs).sum(dim=-1)

        if active_mask is not None and active_mask.shape == ent.shape and active_mask.any():
            return float(ent[active_mask].mean().item())
        return float(ent.mean().item())

    def compute_turbulence_index(
        self,
        zeno_flux: float = 0.35,
        lca_variance: float = 0.25,
        step0_demo_loss: float = 3.0,
        step0_grad_norm: float = 1.0,
        s1_demo_fit: float = 0.75,
        test_entropy: float = 0.5,
    ) -> float:
        """
        Computes the composite Zeno Turbulence Index `T_zeno in [0.0, 1.0]`.
        Spans the full `[0, 1]` dynamic range across laminar near-solved tasks and
        high-turbulence compositional rules.
        """
        # 1. Left-Hemisphere structural turbulence signal in [0, 1]
        flux_clamped = max(0.0, min(1.0, float(zeno_flux)))
        lca_norm = math.tanh(max(0.0, float(lca_variance)) / 1.5)
        s_lh = 0.75 * flux_clamped + 0.25 * lca_norm

        # 2. Step-0 demonstration discrepancy signal in [0, 1]
        loss_norm = max(0.0, min(1.0, (float(step0_demo_loss) - 0.15) / 6.0))
        err_demo = max(0.0, min(1.0, 1.0 - float(s1_demo_fit)))
        grad_norm = math.tanh(max(0.0, float(step0_grad_norm)) / 2.5)
        s_demo = 0.50 * loss_norm + 0.35 * err_demo + 0.15 * grad_norm

        # 3. Right-Hemisphere test-grid predictive entropy signal in [0, 1]
        max_ent = math.log(10.0)
        s_rh = max(0.0, min(1.0, float(test_entropy) / max_ent))

        t_zeno = 0.35 * s_lh + 0.45 * s_demo + 0.20 * s_rh
        return float(max(0.0, min(1.0, t_zeno)))

    def forward(
        self,
        zeno_flux: Union[float, Mapping[str, Any]] = 0.35,
        lca_variance: float = 0.25,
        step0_demo_loss: float = 3.0,
        step0_grad_norm: float = 1.0,
        s1_demo_fit: float = 0.75,
        test_entropy: float = 0.5,
        canvas_cells: int = 100,
        has_occlusion_tie: bool = False,
    ) -> DarkSnailSchedule:
        """
        Computes the dynamic `DarkSnailSchedule` from Step-0 tri-signal telemetry.
        Accepts either individual telemetry scalars or a single dictionary as the first argument.
        """
        if isinstance(zeno_flux, Mapping):
            d = zeno_flux
            lca_variance = float(d.get("lca_variance", lca_variance))
            step0_demo_loss = float(d.get("step0_demo_loss", d.get("step0_loss", step0_demo_loss)))
            step0_grad_norm = float(d.get("step0_grad_norm", d.get("grad_norm_step0", step0_grad_norm)))
            s1_demo_fit = float(d.get("s1_demo_fit", s1_demo_fit))
            test_entropy = float(d.get("test_entropy", test_entropy))
            canvas_cells = int(d.get("canvas_cells", canvas_cells))
            has_occlusion_tie = bool(d.get("has_occlusion_tie", has_occlusion_tie))
            zeno_flux = float(d.get("zeno_flux", 0.35))

        n_cells = max(1, int(canvas_cells))
        t_zeno = self.compute_turbulence_index(
            zeno_flux=float(zeno_flux),
            lca_variance=float(lca_variance),
            step0_demo_loss=float(step0_demo_loss),
            step0_grad_norm=float(step0_grad_norm),
            s1_demo_fit=float(s1_demo_fit),
            test_entropy=float(test_entropy),
        )
        self.last_turbulence_index = t_zeno

        # Surgical 1-cell and 2-cell quantiles for exact pixel occlusion tie-breaking
        q_1cell = min(0.25, max(1.0 / float(n_cells), 1e-4))
        q_2cell = min(0.30, max(2.0 / float(n_cells), 2e-4))
        # Area factor in [0, 1]: larger grids (up to 30x30 = 900) get stronger anchor protection
        area_factor = min(1.0, max(0.0, float(n_cells - 25) / 875.0))

        # Regime 0: Optional Zero-Step Reflex Bypass when 100% solved with vanishing loss & entropy
        if (
            self.allow_zero_step_bypass
            and float(s1_demo_fit) >= 0.999
            and float(step0_demo_loss) < 0.15
            and float(test_entropy) < 0.02
            and not has_occlusion_tie
        ):
            sched = DarkSnailSchedule(
                num_steps=0,
                lr=self.min_lr,
                anchor_weight=self.max_anchor,
                triad_gain=self.min_triad_gain,
                ray_quantiles=[q_1cell],
                turbulence_index=t_zeno,
                regime="reflex_bypass",
                bypass_tta=True,
                asym_gain=0.5,
                cascade_gain=0.25,
            )
            self.last_schedule = sched
            return sched

        is_near_solved = (
            float(step0_demo_loss) < self.near_solved_loss_threshold
            or float(s1_demo_fit) >= self.near_solved_fit_threshold
        )
        is_high_turbulence = (
            not is_near_solved
            and (
                float(step0_demo_loss) > self.high_turb_loss_threshold
                or (float(zeno_flux) >= 0.55 and t_zeno >= 0.55)
            )
        )

        if is_near_solved:
            # Low-Turbulence / Near-Solved Regime (e.g., af22c60d, 0b17323b, ce039d91, ca8f78db, 929ab4e9)
            # Conservative step count (12..18), conservative lr (0.02..0.035), high anchor (0.22..0.30),
            # surgical 1-2 cell ray quantiles so 98-99% correct canvas is never over-adapted.
            num_steps = int(round(12 + 6 * t_zeno))
            num_steps = max(self.min_steps, min(self.max_steps, num_steps))
            lr = float(np.clip(0.022 + 0.014 * t_zeno, self.min_lr, self.max_lr))
            anchor_weight = float(
                np.clip(0.22 + 0.06 * area_factor + 0.02 * (1.0 - t_zeno), self.min_anchor, self.max_anchor)
            )
            # If an occlusion tie is present on a near-solved grid, boost triad_gain to break the tie cleanly
            base_triad = 0.95 if has_occlusion_tie else 0.55
            triad_gain = float(np.clip(base_triad + 0.35 * float(zeno_flux), self.min_triad_gain, self.max_triad_gain))
            ray_quantiles = [q_1cell, q_2cell, 0.02]
            # Deduplicate while preserving order
            dedup_q: List[float] = []
            for q in ray_quantiles:
                if all(abs(q - existing) > 1e-6 for existing in dedup_q):
                    dedup_q.append(float(q))
            sched = DarkSnailSchedule(
                num_steps=num_steps,
                lr=lr,
                anchor_weight=anchor_weight,
                triad_gain=triad_gain,
                ray_quantiles=dedup_q,
                turbulence_index=t_zeno,
                regime="surgical_near_solved",
                bypass_tta=False,
                asym_gain=1.65,
                cascade_gain=0.75,
            )
        elif is_high_turbulence:
            # High-Turbulence / Compositional Rule Regime (e.g., split-panel Boolean or Kronecker rules)
            # Full deliberation (34..40 steps), strong lr (0.05..0.06), relaxed anchor (0.05..0.11),
            # high triad coupling gain (1.0..1.5), broad + surgical ray quantiles.
            num_steps = int(round(33 + 7 * t_zeno))
            num_steps = max(self.min_steps, min(self.max_steps, num_steps))
            lr = float(np.clip(0.048 + 0.012 * t_zeno, self.min_lr, self.max_lr))
            anchor_weight = float(
                np.clip(0.055 + 0.045 * area_factor, self.min_anchor, self.max_anchor)
            )
            triad_gain = float(
                np.clip(1.0 + 0.50 * t_zeno, self.min_triad_gain, self.max_triad_gain)
            )
            ray_quantiles = [0.15, 0.20, q_2cell, q_1cell]
            dedup_q = []
            for q in ray_quantiles:
                if all(abs(q - existing) > 1e-6 for existing in dedup_q):
                    dedup_q.append(float(q))
            sched = DarkSnailSchedule(
                num_steps=num_steps,
                lr=lr,
                anchor_weight=anchor_weight,
                triad_gain=triad_gain,
                ray_quantiles=dedup_q,
                turbulence_index=t_zeno,
                regime="high_turbulence_compositional",
                bypass_tta=False,
                asym_gain=1.75,
                cascade_gain=1.25,
            )
        else:
            # Balanced Deliberation Regime
            num_steps = int(round(20 + 14 * t_zeno))
            num_steps = max(self.min_steps, min(self.max_steps, num_steps))
            lr = float(np.clip(0.035 + 0.020 * t_zeno, self.min_lr, self.max_lr))
            anchor_weight = float(
                np.clip(0.10 + 0.08 * area_factor - 0.03 * t_zeno, self.min_anchor, self.max_anchor)
            )
            triad_gain = float(
                np.clip(0.65 + 0.65 * t_zeno, self.min_triad_gain, self.max_triad_gain)
            )
            if has_occlusion_tie:
                ray_quantiles = [q_1cell, q_2cell, 0.10, 0.18]
            else:
                ray_quantiles = [0.12, q_2cell, q_1cell, 0.20]
            dedup_q = []
            for q in ray_quantiles:
                if all(abs(q - existing) > 1e-6 for existing in dedup_q):
                    dedup_q.append(float(q))
            sched = DarkSnailSchedule(
                num_steps=num_steps,
                lr=lr,
                anchor_weight=anchor_weight,
                triad_gain=triad_gain,
                ray_quantiles=dedup_q,
                turbulence_index=t_zeno,
                regime="balanced_deliberation",
                bypass_tta=False,
                asym_gain=1.5,
                cascade_gain=1.0,
            )

        self.last_schedule = sched
        return sched

    @staticmethod
    def compute_ray_crossing_thresholds(
        l0: torch.Tensor,
        dl: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """
        Computes positive ray scaling thresholds `alpha > 0` where `l0 + alpha * dl`
        flips the argmax color at each active cell.
        """
        c0 = torch.argmax(l0, dim=-1)
        has_active = active_mask is not None and bool(active_mask.any().item())
        active_indices = (
            torch.where(active_mask)[0]
            if has_active
            else torch.arange(l0.size(0), device=l0.device)
        )
        c_start = 2 if (has_active and l0.size(-1) >= 12) else 0
        thresholds: List[float] = []

        for idx in active_indices.tolist():
            cur_c = int(c0[idx].item())
            m_row = l0[idx, cur_c] - l0[idx, c_start:]
            v_row = dl[idx, c_start:] - dl[idx, cur_c]
            pos = (v_row > 1e-6) & (m_row > 0.0)
            if pos.any():
                min_alpha = float((m_row[pos] / v_row[pos]).min().item())
                if min_alpha > 0.0:
                    thresholds.append(min_alpha)

        thresholds.sort()
        return thresholds


class DarkSnailSystem2Adaptor:
    """
    Integrated DarkSnail System 2 Latent Adaptor combining:
      1. `DarkSnailCanvasReadout` (Symmetric + Skew-Symmetric Triad + Directional Z-Order),
      2. `SlowFastTriadCascade` (`z_H -> z_L` terminal recurrent triad coupling),
      3. `ZenoAmygdalaController` (Tri-signal dynamic TTA and surgical ray quantile scheduling),
      4. `compute_grid_prompt_zeno_flux` (Live multi-prime prompt turbulence telemetry).
    """

    def __init__(
        self,
        n_steps: int = 30,
        lr: float = 0.05,
        lambda_anchor: float = 0.1,
        max_act_steps: int = 16,
        readout_scale: float = 6.0,
        use_zeno_schedule: bool = True,
        use_slow_fast_cascade: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        from brain_ai.models.callosum import System2LatentAdaptor

        self.base_adaptor = System2LatentAdaptor(
            n_steps=n_steps,
            lr=lr,
            lambda_anchor=lambda_anchor,
            max_act_steps=max_act_steps,
            readout_scale=readout_scale,
            device=device,
        )
        self.readout = DarkSnailCanvasReadout(readout_scale=readout_scale)
        self.cascade = SlowFastTriadCascade()
        self.controller = ZenoAmygdalaController()
        self.use_zeno_schedule = bool(use_zeno_schedule)
        self.use_slow_fast_cascade = bool(use_slow_fast_cascade)
        self._active_schedule: Optional[DarkSnailSchedule] = None

        # Monkey-patch `_apply_callosal_canvas_readout` on `self.base_adaptor` so all
        # forward/loss/prediction paths automatically route through `DarkSnailCanvasReadout`!
        self.base_adaptor._apply_callosal_canvas_readout = self._darksnail_canvas_readout_hook

    def _darksnail_canvas_readout_hook(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        delta_z: Optional[torch.Tensor],
        carry: Any = None,
        callosal_emb: Any = None,
    ) -> torch.Tensor:
        triad_gain = self._active_schedule.triad_gain if self._active_schedule is not None else None
        asym_gain = self._active_schedule.asym_gain if self._active_schedule is not None else None
        return self.readout(
            logits=logits,
            inputs=inputs,
            delta_z=delta_z,
            model=model,
            carry=carry,
            callosal_emb=callosal_emb,
            target_hw=self.base_adaptor._active_target_hw,
            color_preserving=self.base_adaptor._active_color_preserving,
            triad_gain_override=triad_gain,
            asym_gain_override=asym_gain,
        )

    def adapt_task(
        self,
        model: nn.Module,
        task_data: dict,
        puzzle_id: int = 0,
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_anchor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        from brain_ai.models.callosum import encode_grid_to_trm_seq, infer_arc_output_shape

        demos = task_data.get("train", [])
        test_pairs = task_data.get("test", [])
        flux_info = compute_grid_prompt_zeno_flux(demos, test_pairs)

        if self.use_zeno_schedule and n_steps is None and lr is None and lambda_anchor is None:
            # Estimate target canvas cells
            if test_pairs and "input" in test_pairs[0]:
                th, tw = infer_arc_output_shape(demos, test_pairs[0]["input"])
            elif demos and "output" in demos[0]:
                th, tw = len(demos[0]["output"]), len(demos[0]["output"][0])
            else:
                th, tw = 10, 10
            canvas_cells = th * tw

            # Initial schedule estimate before Step-0 pass
            sched = self.controller(
                zeno_flux=flux_info["zeno_flux"],
                lca_variance=flux_info["lca_variance"],
                step0_demo_loss=3.0,
                step0_grad_norm=1.0,
                s1_demo_fit=0.80,
                test_entropy=0.40,
                canvas_cells=canvas_cells,
            )
            self._active_schedule = sched
            n_steps = sched.num_steps
            lr = sched.lr
            lambda_anchor = sched.anchor_weight

        delta_z, diag = self.base_adaptor.adapt_task(
            model=model,
            task_data=task_data,
            puzzle_id=puzzle_id,
            n_steps=n_steps,
            lr=lr,
            lambda_anchor=lambda_anchor,
        )

        # Refine schedule using actual Step-0 loss and gradient norm for downstream ray extrapolation
        if test_pairs and "input" in test_pairs[0]:
            th, tw = infer_arc_output_shape(demos, test_pairs[0]["input"])
            canvas_cells = th * tw
        else:
            canvas_cells = 100

        refined_sched = self.controller(
            zeno_flux=flux_info["zeno_flux"],
            lca_variance=flux_info["lca_variance"],
            step0_demo_loss=diag.get("step0_loss", 3.0),
            step0_grad_norm=diag.get("grad_norm_step0", 1.0),
            s1_demo_fit=max(0.0, min(1.0, 1.0 - diag.get("step0_loss", 3.0) / 8.0)),
            test_entropy=0.35,
            canvas_cells=canvas_cells,
        )
        self._active_schedule = refined_sched
        diag["zeno_flux"] = flux_info["zeno_flux"]
        diag["lca_variance"] = flux_info["lca_variance"]
        diag["turbulence_index"] = refined_sched.turbulence_index
        diag["regime"] = refined_sched.regime
        diag["ray_quantiles"] = refined_sched.ray_quantiles
        return delta_z, diag

    def predict_s1_and_s2_pair(
        self,
        model: nn.Module,
        test_input_grid: List[List[int]],
        puzzle_id: int,
        delta_z: Optional[torch.Tensor],
        accepted: bool = True,
        demos: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates System 1 and System 2 with DarkSnail readout and adaptive surgical
        1-2 cell ray extrapolation quantiles from `ZenoAmygdalaController`.
        """
        res = self.base_adaptor.predict_s1_and_s2_pair(
            model=model,
            test_input_grid=test_input_grid,
            puzzle_id=puzzle_id,
            delta_z=delta_z,
            accepted=accepted,
            demos=demos,
        )
        return res

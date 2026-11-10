"""Base-Effect-Aware Forecast Head for an *external*, frozen TimesFM-3 baseline.

No claim is made that an exact CPI YoY identity applies to separately rounded,
first-release macroeconomic statistics. Historical rates and anchors are supplied
from a point-in-time data audit. Units: y/y and m/m rates in percent points.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

HORIZONS = (1, 3, 6, 12)


def base_effect_yoy(
    anchor_yoy_pct: torch.Tensor,
    past_12_mom_pct: torch.Tensor,
    future_12_mom_pct: torch.Tensor,
    horizons: Sequence[int] = HORIZONS,
) -> torch.Tensor:
    """Return CPI YoY at h=1..12 via the log-index identity.

    y_{t+h} = 100*expm1[log1p(y_t/100) + sum_{j=1}^h
                              (log1p(m_{t+j}/100)-log1p(m_{t+j-12}/100))].

    past_12_mom_pct: [B,12], chronological r_(t-11),...,r_t.
    future_12_mom_pct: [B,12], chronological r_(t+1),...,r_(t+12).
    """
    if past_12_mom_pct.ndim != 2 or past_12_mom_pct.shape[1] != 12:
        raise ValueError("past_12_mom_pct must have shape [batch,12]")
    if future_12_mom_pct.shape != past_12_mom_pct.shape:
        raise ValueError("future_12_mom_pct shape must match past_12_mom_pct")
    if anchor_yoy_pct.ndim != 1 or anchor_yoy_pct.shape[0] != past_12_mom_pct.shape[0]:
        raise ValueError("anchor_yoy_pct must have shape [batch]")
    if (past_12_mom_pct <= -100).any() or (future_12_mom_pct <= -100).any():
        raise ValueError("m/m growth <= -100% is invalid")
    if (anchor_yoy_pct <= -100).any():
        raise ValueError("anchor YoY <= -100% is invalid")
    if not torch.isfinite(past_12_mom_pct).all() or not torch.isfinite(future_12_mom_pct).all():
        raise ValueError("monthly observations must be finite; do not fill unavailable values")
    if not torch.isfinite(anchor_yoy_pct).all():
        raise ValueError("anchor YoY must be finite")

    y_log = torch.log1p(anchor_yoy_pct / 100.0)
    past_log = torch.log1p(past_12_mom_pct / 100.0)
    future_log = torch.log1p(future_12_mom_pct / 100.0)
    delta_log = torch.cumsum(future_log - past_log, dim=1)
    indexes = [h - 1 for h in horizons]
    if any(not 1 <= h <= 12 for h in horizons):
        raise ValueError("all horizons must be in [1,12]")
    return 100.0 * torch.expm1(y_log[:, None] + delta_log[:, indexes])


class BaseEffectAwareHead(nn.Module):
    """Low-parameter monthly-growth branch + horizon-wise gated fusion with B3.

    The TimesFM-3 model and its forecasts remain frozen. This head is trained
    on a separate, point-in-time panel of past CPI MoM/YoY and cached B3 outputs.
    The learned correction forecasts 12 future *monthly* CPI log changes, then
    transforms them into CPI YoY using the accounting identity. Its initial
    gate is 0, so the first forward pass equals B3 to floating-point accuracy.

    The gate is projected to [0,1], making forecasts a convex interpolation in
    YoY-log space. This prevents unbounded post-hoc residual extrapolation.
    """

    def __init__(self, width: int = 24, macro_dim: int = 0):
        super().__init__()
        if width < 1 or macro_dim < 0:
            raise ValueError("invalid width or macro_dim")
        self.macro_dim = macro_dim
        in_features = 12 + 1 + 2 + macro_dim * 2
        self.monthly = nn.Sequential(
            nn.Linear(in_features, width),
            nn.Tanh(),
            nn.Linear(width, 12),
        )
        # Zero-init of the residual predictor gives recent-three-months
        # mean monthly growth as a deterministic starting forecast.
        nn.init.zeros_(self.monthly[-1].weight)
        nn.init.zeros_(self.monthly[-1].bias)
        self.raw_gate = nn.Parameter(torch.zeros(len(HORIZONS)))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        anchor_yoy_pct: torch.Tensor,  # [B]
        past_12_mom_pct: torch.Tensor,  # [B,12]
        b3_yoy_pct: torch.Tensor,  # [B,4], horizon order 1/3/6/12
        anchor_month: torch.Tensor,  # [B] integer 1..12
        macro: torch.Tensor | None = None,  # [B,macro_dim] train-only scaled
        macro_mask: torch.Tensor | None = None,  # [B,macro_dim] True unavailable
    ) -> dict[str, torch.Tensor]:
        if b3_yoy_pct.ndim != 2 or b3_yoy_pct.shape[1] != 4:
            raise ValueError("b3_yoy_pct must be [batch,4] in horizon order 1,3,6,12")
        if not torch.isfinite(b3_yoy_pct).all() or (b3_yoy_pct <= -100).any():
            raise ValueError("B3 prediction must be finite and > -100%")
        if anchor_month.ndim != 1 or anchor_month.shape[0] != anchor_yoy_pct.shape[0]:
            raise ValueError("anchor_month must have shape [batch]")
        if ((anchor_month < 1) | (anchor_month > 12)).any():
            raise ValueError("anchor_month must be 1..12")
        # The helper provides full input checking and unit-preserving identity.
        observed_log = torch.log1p(past_12_mom_pct / 100)
        y_log = torch.log1p(anchor_yoy_pct / 100)
        month_phase = (anchor_month.float() - 1) * (2 * math.pi / 12)
        model_inputs = [
            100 * observed_log,
            100 * y_log[:, None],
            torch.sin(month_phase)[:, None],
            torch.cos(month_phase)[:, None],
        ]
        if self.macro_dim:
            if macro is None or macro_mask is None:
                raise ValueError("macro and macro_mask required when macro_dim>0")
            if macro.shape != (anchor_yoy_pct.shape[0], self.macro_dim):
                raise ValueError("wrong macro shape")
            if macro_mask.shape != macro.shape:
                raise ValueError("wrong macro_mask shape")
            # Missingness is not equivalent to a legitimate zero value.
            # Input macro features must be point-in-time audited and scaled
            # using data available at training cutoff only.
            model_inputs += [torch.where(macro_mask, 0.0, macro), macro_mask.float()]
        features = torch.cat(model_inputs, dim=1)
        if not torch.isfinite(features).all():
            raise ValueError("non-finite data in point-in-time input")

        # Recent 3-month mean is the fixed interpretable starting point.
        recent_mean_log = observed_log[:, -3:].mean(dim=1, keepdim=True)
        residual_future_log = 0.01 * torch.tanh(self.monthly(features))
        predicted_future_log = recent_mean_log + residual_future_log
        predicted_future_mom_pct = 100 * torch.expm1(predicted_future_log)
        structural_yoy_pct = base_effect_yoy(
            anchor_yoy_pct, past_12_mom_pct, predicted_future_mom_pct
        )

        gate = self.raw_gate.clamp(0.0, 1.0)
        base_log = torch.log1p(b3_yoy_pct / 100)
        struct_log = torch.log1p(structural_yoy_pct / 100)
        blended_log = base_log + gate[None, :] * (struct_log - base_log)
        final_yoy_pct = 100 * torch.expm1(blended_log)
        return {
            "yoy_pct": final_yoy_pct,
            "structural_yoy_pct": structural_yoy_pct,
            "future_mom_pct": predicted_future_mom_pct,
            "gates": gate,
        }


def masked_mae(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """MAE over labels that genuinely became public by the fit/score cutoff."""
    if pred.shape != target.shape or valid.shape != target.shape:
        raise ValueError("incompatible prediction, target and availability shapes")
    if not bool(valid.any()):
        raise ValueError("no released labels to compute loss")
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    return torch.abs(pred - safe_target).masked_select(valid).mean()

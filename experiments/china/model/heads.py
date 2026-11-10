"""Same-structure 688-parameter coordinate pair. No dataset/label readers."""
from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SOURCE = ROOT / 'stage2/imports/BaseEffect_TimesFM3_MVP_v1/BaseEffect_TimesFM3_MVP/beah.py'
reference = ModuleType('stage5_pinned_beah')
reference.__file__ = str(REFERENCE_SOURCE)
exec(compile(REFERENCE_SOURCE.read_bytes(), str(REFERENCE_SOURCE), 'exec'), reference.__dict__)
HORIZONS = reference.HORIZONS


def features(anchor, monthly, month):
    """Reference's 15 features. Frozen B3 enters the common skip/fusion only."""
    phase = (month.float() - 1) * (2 * math.pi / 12)
    return torch.cat((100 * torch.log1p(monthly / 100),
                      100 * torch.log1p(anchor[:, None] / 100),
                      phase.sin()[:, None], phase.cos()[:, None]), dim=1)


def fixed_candidate(anchor, monthly):
    recent = torch.log1p(monthly / 100)[:, -3:].mean(1, keepdim=True).expand(-1, 12)
    return reference.base_effect_yoy(anchor, monthly, 100 * torch.expm1(recent))


def blend(b3, candidate_log, raw_gate):
    """Algebraically log-convex fusion; exact B3 at zero gate, retaining gradient.

    The stable expression avoids log1p/expm1 round-trip error in the zero skip.
    Both heads use this same implementation, without a detached/zero-gradient
    branch at the gate boundary.
    """
    gate = raw_gate.clamp(0, 1)
    return b3 + (100 + b3) * torch.expm1(gate * (candidate_log - torch.log1p(b3 / 100)))


def check_inputs(anchor, monthly, b3, month):
    batch = anchor.shape[0] if anchor.ndim == 1 else -1
    if monthly.shape != (batch, 12) or b3.shape != (batch, 4) or month.shape != (batch,):
        raise ValueError('Expected anchor[B], monthly[B,12], B3[B,4], month[B]')
    if any(not bool(torch.isfinite(t).all()) for t in (anchor, monthly, b3, month)):
        raise ValueError('Unavailable observations must not be filled or passed as inputs')
    if any(bool((t <= -100).any()) for t in (anchor, monthly, b3)):
        raise ValueError('Rates must be greater than -100 percent')
    if bool(((month < 1) | (month > 12) | (month != month.round())).any()):
        raise ValueError('Calendar month must be an integer in 1..12')


def readout(kind):
    matrix = torch.zeros(4, 12)
    for row, horizon in enumerate(HORIZONS):
        start = 0 if kind == 'BEAH' or row == 0 else HORIZONS[row-1]
        matrix[row, start:horizon] = 1
    return matrix


class CoordinateHead(nn.Module):
    """Identical network, gate, activation and fusion; one fixed readout differs.

    BEAH: prefix sums of twelve bounded monthly log corrections.
    Direct: nonoverlapping sums over groups (1,2,3,6), interpreted directly as
    four YoY-log corrections, not as a future monthly trajectory. All twelve
    units contribute to scored horizons; no unused output-row parameters.

    If z is the common bounded output, P z = L (Q z), where L is the 4x4
    unit lower-triangular matrix. P and Q have the same row space and rank4.
    Bounds and optimizer geometry change under the mapping. This is a test
    of parameterization/inductive bias, not an additional-information test.
    """
    def __init__(self, kind):
        super().__init__()
        if kind not in ('BEAH', 'Direct'):
            raise ValueError(kind)
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(15, 24), nn.Tanh(), nn.Linear(24, 12))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_gate = nn.Parameter(torch.zeros(4))
        self.register_buffer('readout', readout(kind))

    def forward(self, anchor, monthly, b3, month):
        check_inputs(anchor, monthly, b3, month)
        fixed = fixed_candidate(anchor, monthly)
        residual = .01 * torch.tanh(self.net(features(anchor, monthly, month)))
        delta = residual @ self.readout.T
        candidate = fixed + (100 + fixed) * torch.expm1(delta)
        log = torch.log1p(candidate / 100)
        return dict(yoy_pct=blend(b3, log, self.raw_gate), candidate_yoy_pct=candidate,
                    candidate_log=log, correction_log=delta, latent_correction_log=residual,
                    gates=self.raw_gate.clamp(0, 1))


def create(kind, seed, dtype=torch.float32):
    torch.manual_seed(seed)
    return CoordinateHead(kind).to(device='cpu', dtype=dtype)

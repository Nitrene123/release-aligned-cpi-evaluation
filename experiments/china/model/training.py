"""One training/selection implementation for both heads; no file/data readers."""
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class TrainingSpec:
    learning_rate: float = .005
    weight_decay: float = .01
    betas: tuple = (.9, .999)
    eps: float = 1e-8
    clip_l2: float = 1.
    gate_penalty: float = .002
    max_epochs: int = 400
    patience: int = 35
    improvement_tolerance: float = 1e-8
    batch_policy: str = 'full batch; fixed input row order; one update per epoch'


SPEC = TrainingSpec()
SEEDS = (417, 42, 123, 2027, 3407)


def optimizer_for(model, spec=SPEC):
    return torch.optim.AdamW(model.parameters(), lr=spec.learning_rate,
        weight_decay=spec.weight_decay, betas=spec.betas, eps=spec.eps)


def mae(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('Invalid YoY target/mask shape or mask dtype')
    if not bool(valid.any()) or not bool(torch.isfinite(target[valid]).all()):
        raise ValueError('No finite eligible YoY labels')
    return (prediction - torch.where(valid, target, 0.)).abs()[valid].mean()


def step(model, optimizer, inputs, target, valid, spec=SPEC):
    model.train(); optimizer.zero_grad(set_to_none=True)
    prediction = model(*inputs)['yoy_pct']
    yoy_mae = mae(prediction, target, valid)
    loss = yoy_mae + spec.gate_penalty * model.raw_gate.square().sum()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.clip_l2, error_if_nonfinite=True)
    gradients = {n: dict(finite=bool(torch.isfinite(p.grad).all()), nonzero=bool((p.grad != 0).any()))
                 for n, p in model.named_parameters() if p.grad is not None}
    optimizer.step()
    with torch.no_grad():
        model.raw_gate.clamp_(0, 1)
    return dict(yoy_mae=float(yoy_mae.detach()), objective=float(loss.detach()),
                preclip_grad_l2=float(norm), gradients=gradients)


def select_epochs(model, train, validation, spec=SPEC):
    """Epoch count only. Final-fit data/release eligibility belongs to G2/G4."""
    ti, ty, tv = train; vi, vy, vv = validation
    optimizer = optimizer_for(model, spec)
    def score():
        model.eval()
        with torch.no_grad():
            return float(mae(model(*vi)['yoy_pct'][:, 1], vy[:, 1], vv[:, 1]))
    best = score(); best_epoch = 0; wait = 0
    trace = [dict(epoch=0, h3_mae=best)]
    for epoch in range(1, spec.max_epochs + 1):
        step(model, optimizer, ti, ty, tv, spec)
        value = score(); trace.append(dict(epoch=epoch, h3_mae=value))
        if value < best - spec.improvement_tolerance:
            best = value; best_epoch = epoch; wait = 0
        else:
            wait += 1
        if wait >= spec.patience:
            break
    return dict(selected_epoch=best_epoch, best_h3_mae=best, optimizer_steps=len(trace)-1,
                validation_calls=len(trace), trace=trace, spec=asdict(spec))

#!/usr/bin/env python3
"""Target-free checks for the packaged PIT mask counterfactual."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rows(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def close(a, b, tol=1e-10):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol

def main():
    manifest = json.loads((ROOT / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    failures = []
    for rel, expected in manifest.items():
        p = ROOT / rel
        if not p.is_file():
            failures.append(f'missing: {rel}')
        elif sha256(p) != expected:
            failures.append(f'hash mismatch: {rel}')
    if failures:
        raise SystemExit('\n'.join(failures))

    forecast = rows(ROOT / 'outputs' / 'forecast_predictions.csv')
    strict = [r for r in forecast if r['variant'] == 'strict_replay']
    backfill = [r for r in forecast if r['variant'] == 'post_origin_backfill']
    if len(forecast) != 3768 or len(strict) != 1884 or len(backfill) != 1884:
        raise SystemExit('forecast row-count mismatch')
    for variant_rows in (strict, backfill):
        keys = {(r['origin_id'], int(r['horizon_months'])) for r in variant_rows}
        if len(keys) != len(variant_rows):
            raise SystemExit('duplicate forecast key')
    if any(any(k in r for k in ('target_value', 'first_release_yoy_pct', 'y_true')) for r in forecast):
        raise SystemExit('forecast archive contains a target-value field')

    reference = rows(ROOT / 'inputs' / 'archived_b3_q50.csv')
    ref = {(r['origin_id'], int(r['horizon_months'])): float(r['archived_q50']) for r in reference}
    strict_map = {(r['origin_id'], int(r['horizon_months'])): float(r['point_forecast_q50']) for r in strict}
    if ref.keys() != strict_map.keys():
        raise SystemExit('strict replay keys do not match archived B3 q50')
    max_diff = max(abs(ref[k]-strict_map[k]) for k in ref)
    if max_diff > 0.002:
        raise SystemExit(f'strict replay exceeds q50 tolerance: {max_diff}')

    losses = rows(ROOT / 'derived_scores' / 'paired_origin_losses.csv')
    summaries = rows(ROOT / 'derived_scores' / 'mae_by_horizon.csv')
    if len(losses) != 396 or len(summaries) != 4:
        raise SystemExit('derived score row-count mismatch')
    for s in summaries:
        h = int(s['horizon_months'])
        g = [r for r in losses if int(r['horizon_months']) == h]
        if len(g) != 99:
            raise SystemExit(f'h{h}: expected 99 paired origins, got {len(g)}')
        strict_mae = sum(float(r['strict_absolute_error_pp']) for r in g)/len(g)
        backfill_mae = sum(float(r['post_origin_absolute_error_pp']) for r in g)/len(g)
        delta = backfill_mae - strict_mae
        if not close(strict_mae, float(s['strict_replay_mae_pp'])):
            raise SystemExit(f'h{h}: strict MAE mismatch')
        if not close(backfill_mae, float(s['post_origin_backfill_mae_pp'])):
            raise SystemExit(f'h{h}: backfill MAE mismatch')
        if not close(delta, float(s['mae_change_pp'])):
            raise SystemExit(f'h{h}: paired change mismatch')
        direction = [float(r['paired_absolute_error_change_pp']) for r in g]
        if any(not close(d, float(r['post_origin_absolute_error_pp'])-float(r['strict_absolute_error_pp']))
               for d, r in zip(direction, g)):
            raise SystemExit(f'h{h}: origin-level paired loss mismatch')
    print(json.dumps({
        'status': 'PASS_TARGET_FREE_ARCHIVE_AND_DERIVED_SCORE_CHECKS',
        'forecast_rows': len(forecast), 'paired_loss_rows': len(losses),
        'origins_per_horizon': 99, 'strict_replay_max_q50_difference_pp': max_diff,
        'target_values_loaded': False,
        'scope': 'hashes, key counts, q50 parity, and arithmetic from archived paired losses'
    }, indent=2))

if __name__ == '__main__':
    main()

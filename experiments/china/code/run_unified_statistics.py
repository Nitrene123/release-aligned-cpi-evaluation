#!/usr/bin/env python3
"""Generate four statistical baseline forecasts per registered key, without scoring."""
from __future__ import annotations

import csv, hashlib, json, math, os, sys, time, warnings, resource
from datetime import datetime, timedelta, timezone
from pathlib import Path
from portable_paths import project_root

import numpy as np
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = project_root(__file__)
EXP = ROOT / 'revision_experiments_v2/rolling_foundation_matched_v2'
DATA = EXP / 'source_data'
OUT = EXP / 'statistical_reference_v1'
TARGET_START, TARGET_END = '2018-01', '2026-09'
HORIZONS = (1, 3, 6, 12)
SARIMA_SPEC = dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), trend='c',
                   enforce_stationarity=True, enforce_invertibility=True,
                   simple_differencing=False, measurement_error=False,
                   time_varying_regression=False, mle_regression=True,
                   concentrate_scale=False, missing='none', initialization='stationary')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f'refusing empty output {path.name}')
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def stamp(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)


def add_month(month, n):
    y, m = map(int, month.split('-'))
    x = y * 12 + m - 1 + n
    return f'{x // 12:04d}-{x % 12 + 1:02d}'


def month_range(start, end):
    y, m = map(int, start.split('-')); ey, em = map(int, end.split('-'))
    out = []
    while (y, m) <= (ey, em):
        out.append(f'{y:04d}-{m:02d}')
        m += 1
        if m == 13: y, m = y + 1, 1
    return out


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    if statsmodels.__version__ != '0.14.4':
        raise RuntimeError(f'expected statsmodels 0.14.4, got {statsmodels.__version__}')
    if OUT.exists():
        raise RuntimeError('statistical_reference_v1 already exists; refusing to overwrite')
    cpi_rows = read_csv(DATA / 'unified_cpi_panel.csv')
    cpi = {}
    for r in cpi_rows:
        m = r['reference_month'][:7]
        av = stamp(r['available_at_conservative_local'])
        value = r['first_release_yoy_pct'].strip()
        cpi[m] = {'available_at': av, 'raw_value': value}
    expected_months = month_range('2012-01', '2026-09')
    if list(sorted(cpi)) != expected_months or cpi['2026-09']['raw_value']:
        raise RuntimeError('unified CPI panel must be contiguous and withhold only numeric 2026-09 target')

    origins = {}
    for r in read_csv(DATA / 'unified_origins.csv'):
        release = stamp(r['origin_timestamp_utc'])
        cutoff = release + timedelta(minutes=1)
        oid = r['origin_id']
        origins[oid] = {'origin_id': oid, 'origin_month': r['origin_month'],
                        'origin_release_timestamp_utc': release.isoformat(),
                        'origin_timestamp_utc': cutoff.isoformat()}
    planned = []
    for target in month_range(TARGET_START, TARGET_END):
        for h in HORIZONS:
            origin_month = add_month(target, -h)
            oid = 'CN_' + origin_month
            if oid not in origins or target not in cpi:
                raise RuntimeError(f'missing origin/target metadata: {oid} {target}')
            origin = origins[oid]
            cutoff = stamp(origin['origin_timestamp_utc'])
            if cpi[target]['available_at'] <= cutoff:
                raise RuntimeError(f'target release not after forecast origin: {oid}/h{h}/{target}')
            planned.append((oid, origin_month, cutoff, target, h))
    if len(planned) != 420 or len(set((x[0], x[4], x[3]) for x in planned)) != 420:
        raise RuntimeError('planned evaluation keys are not exactly 420 unique target-origin-horizon rows')
    by_origin = {}
    for x in planned: by_origin.setdefault(x[0], []).append(x)

    forecast_rows, fit_rows = [], []
    tic = time.perf_counter(); sarima_seconds = 0.0
    for index, (oid, reqs) in enumerate(sorted(by_origin.items(), key=lambda x: x[1][0][1])):
        origin_month = reqs[0][1]
        cutoff = reqs[0][2]
        observed = [m for m in expected_months if m <= origin_month and m in cpi
                    and cpi[m]['raw_value'] and cpi[m]['available_at'] < cutoff]
        if not observed or observed[-1] != origin_month:
            raise RuntimeError(f'point-in-time CPI context does not reach origin: {oid}')
        if observed != month_range(observed[0], observed[-1]):
            raise RuntimeError(f'CPI history contains a gap at {oid}')
        y = np.asarray([float(cpi[m]['raw_value']) for m in observed], dtype=np.float64)
        if not np.isfinite(y).all(): raise RuntimeError(f'nonfinite CPI history at {oid}')
        max_h = max(x[4] for x in reqs)

        # No-change and target-month seasonal-naive use only as-of first-release CPI.
        for _, _, _, target, h in reqs:
            pred = float(y[-1])
            forecast_rows.append({'model':'No-change','seed':'','origin_id':oid,'origin_month':origin_month,
                                  'origin_timestamp_utc':cutoff.isoformat(),'target_month':target,'horizon_months':h,
                                  'prediction':pred,'status':'FORECAST_COMPLETE','failure_reason':'',
                                  'training_start':observed[0],'training_end':observed[-1],
                                  'training_n_months':len(y),'adjacent_pairs':'',
                                  'sarima_optimizer_converged':'','sarima_iterations':'',
                                  'input_hash':hashlib.sha256((oid+'|'+observed[-1]+'|'+format(pred,'.17g')).encode()).hexdigest()})
            seasonal_month = add_month(target, -12)
            if seasonal_month not in observed or cpi[seasonal_month]['available_at'] >= cutoff:
                forecast_rows.append({'model':'Seasonal-naive','seed':'','origin_id':oid,'origin_month':origin_month,
                                      'origin_timestamp_utc':cutoff.isoformat(),'target_month':target,'horizon_months':h,
                                      'prediction':'','status':'FAILED_INPUT_INELIGIBLE','failure_reason':'seasonal lag unavailable before origin',
                                      'training_start':observed[0],'training_end':observed[-1],
                                      'training_n_months':len(y),'adjacent_pairs':'',
                                      'sarima_optimizer_converged':'','sarima_iterations':'','input_hash':''})
            else:
                sv=float(cpi[seasonal_month]['raw_value'])
                forecast_rows.append({'model':'Seasonal-naive','seed':'','origin_id':oid,'origin_month':origin_month,
                                      'origin_timestamp_utc':cutoff.isoformat(),'target_month':target,'horizon_months':h,
                                      'prediction':sv,'status':'FORECAST_COMPLETE','failure_reason':'',
                                      'training_start':observed[0],'training_end':observed[-1],
                                      'training_n_months':len(y),'adjacent_pairs':'',
                                      'sarima_optimizer_converged':'','sarima_iterations':'',
                                      'input_hash':hashlib.sha256((oid+'|'+seasonal_month+'|'+format(sv,'.17g')).encode()).hexdigest()})

        # AR(1): OLS with intercept on adjacent monthly pairs, then recursive forecast.
        if len(y) - 1 < 24:
            ar_forecasts = None; ar_reason = 'fewer than 24 adjacent monthly pairs'; alpha=phi=float('nan')
        else:
            X=np.column_stack([np.ones(len(y)-1), y[:-1]])
            beta=np.linalg.lstsq(X,y[1:],rcond=None)[0]
            alpha,phi=map(float,beta)
            ar_forecasts=[]; current=float(y[-1])
            for _ in range(max_h):
                current=alpha+phi*current; ar_forecasts.append(current)
            ar_reason=''
        for _, _, _, target, h in reqs:
            ok=ar_forecasts is not None and math.isfinite(ar_forecasts[h-1])
            forecast_rows.append({'model':'AR(1)','seed':'','origin_id':oid,'origin_month':origin_month,
                                  'origin_timestamp_utc':cutoff.isoformat(),'target_month':target,'horizon_months':h,
                                  'prediction':float(ar_forecasts[h-1]) if ok else '',
                                  'status':'FORECAST_COMPLETE' if ok else 'FAILED_FIT','failure_reason':ar_reason,
                                  'training_start':observed[0],'training_end':observed[-1],
                                  'training_n_months':len(y),'adjacent_pairs':max(0,len(y)-1),
                                  'sarima_optimizer_converged':'','sarima_iterations':'',
                                  'ar_intercept':alpha if math.isfinite(alpha) else '',
                                  'ar_coefficient':phi if math.isfinite(phi) else '',
                                  'input_hash':hashlib.sha256((oid+'|'+','.join(observed)+'|AR1').encode()).hexdigest() if ok else ''})
        fit_rows.append({'origin_id':oid,'origin_month':origin_month,'origin_timestamp_utc':cutoff.isoformat(),
                         'model':'AR(1)','status':'FIT_COMPLETE' if ar_forecasts is not None else 'FAILED_FIT',
                         'training_start':observed[0],'training_end':observed[-1],'training_n_months':len(y),
                         'adjacent_pairs':max(0,len(y)-1),'intercept':alpha if math.isfinite(alpha) else '',
                         'ar_coefficient':phi if math.isfinite(phi) else '', 'optimizer_converged':'',
                         'iterations':'','log_likelihood':'','parameter_vector':'','failure_reason':ar_reason})

        # Refit the frozen SARIMA specification once per origin. No alternate fit or retry.
        st=time.perf_counter(); sarima_ok=False; sarima_fc=None; sarima_reason=''
        convergence=''; iterations=''; ll=''; params=''
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res=SARIMAX(y,**SARIMA_SPEC).fit(method='lbfgs',maxiter=500,disp=False,
                                                   pgtol=1e-8,factr=1e7)
            sarima_fc=np.asarray(res.forecast(steps=max_h),dtype=np.float64)
            if not np.isfinite(sarima_fc).all(): raise RuntimeError('nonfinite forecast')
            sarima_ok=True
            ret=getattr(res,'mle_retvals',{}) or {}
            convergence=str(bool(ret.get('converged',False)))
            iterations=str(ret.get('iterations',''))
            ll=format(float(res.llf),'.17g')
            params=json.dumps({str(k):float(v) for k,v in zip(res.param_names,res.params)},sort_keys=True)
            reason='; '.join(str(w.message) for w in caught)
        except Exception as exc:
            sarima_reason=f'{type(exc).__name__}: {exc}'
        sarima_seconds += time.perf_counter()-st
        for _, _, _, target, h in reqs:
            ok=sarima_ok and math.isfinite(float(sarima_fc[h-1]))
            forecast_rows.append({'model':'SARIMA','seed':'','origin_id':oid,'origin_month':origin_month,
                                  'origin_timestamp_utc':cutoff.isoformat(),'target_month':target,'horizon_months':h,
                                  'prediction':float(sarima_fc[h-1]) if ok else '',
                                  'status':'FORECAST_COMPLETE' if ok else 'FAILED_FIT',
                                  'failure_reason':sarima_reason,'training_start':observed[0],
                                  'training_end':observed[-1],'training_n_months':len(y),'adjacent_pairs':'',
                                  'sarima_optimizer_converged':convergence,'sarima_iterations':iterations,
                                  'input_hash':hashlib.sha256((oid+'|'+','.join(observed)+'|SARIMA').encode()).hexdigest() if ok else ''})
        fit_rows.append({'origin_id':oid,'origin_month':origin_month,'origin_timestamp_utc':cutoff.isoformat(),
                         'model':'SARIMA','status':'FIT_COMPLETE' if sarima_ok else 'FAILED_FIT',
                         'training_start':observed[0],'training_end':observed[-1],'training_n_months':len(y),
                         'adjacent_pairs':'','intercept':'','ar_coefficient':'',
                         'optimizer_converged':convergence,'iterations':iterations,'log_likelihood':ll,
                         'parameter_vector':params,'failure_reason':sarima_reason})
        if (index+1)%12==0: print(f'statistical origins {index+1}/{len(by_origin)}',flush=True)

    if len(forecast_rows)!=420*4 or len({(r['model'],r['origin_id'],int(r['horizon_months']),r['target_month']) for r in forecast_rows})!=len(forecast_rows):
        raise RuntimeError(f'unexpected statistical forecast count/key uniqueness: {len(forecast_rows)}')
    OUT.mkdir(parents=True)
    write_csv(OUT/'forecast_predictions.csv',forecast_rows)
    write_csv(OUT/'origin_fit_audit.csv',fit_rows)
    manifest={
        'protocol_id':'CN_UNIFIED_EXPANDING_WINDOW_2018_2026_V1',
        'status':'STATISTICAL_FORECASTS_COMPLETE_NO_SCORING',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'target_month_range':[TARGET_START,TARGET_END], 'planned_pairs':420,
        'methods':['No-change','Seasonal-naive','AR(1)','SARIMA'],
        'origin_count':len(by_origin),'forecast_rows':len(forecast_rows),'fit_rows':len(fit_rows),
        'statsmodels_version':statsmodels.__version__,
        'sarima_spec':{**SARIMA_SPEC,'optimizer':'lbfgs','maxiter':500,'pgtol':1e-8,'factr':1e7},
        'sarima_origin_fit_wall_seconds':sarima_seconds,
        'total_wall_seconds':time.perf_counter()-tic,
        'process_peak_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        'cpu_threads':int(os.environ.get('OMP_NUM_THREADS','1')),
        'numeric_target_2026_09_loaded':False,'evaluation_target_errors_computed':False,
        'failed_sarima_origins':[r['origin_id'] for r in fit_rows if r['model']=='SARIMA' and r['status']!='FIT_COMPLETE'],
        'source_files':{n:sha(DATA/n) for n in ('unified_origins.csv','unified_cpi_panel.csv','source_bundle_manifest.json')},
        'outputs':{n:{'sha256':sha(OUT/n),'rows':len(read_csv(OUT/n))} for n in ('forecast_predictions.csv','origin_fit_audit.csv')},
        'limitations':['Retrospective expanding-window backtest; not blind or prospective.',
                       'SARIMA convergence warnings and nonconverged states are retained without retry.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('status','origin_count','forecast_rows','fit_rows','sarima_origin_fit_wall_seconds','failed_sarima_origins','outputs')},indent=2),flush=True)


if __name__=='__main__': main()

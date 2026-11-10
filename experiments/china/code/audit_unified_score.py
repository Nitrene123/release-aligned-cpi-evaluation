import csv,json,hashlib,math
from pathlib import Path
from portable_paths import project_root
from collections import defaultdict
base=project_root(__file__)/'revision_experiments_v2/rolling_foundation_matched_v2'
run=base/'runs/run_20261030_unified_full_v2'; sc=base/'score_v1'
sm=json.loads((sc/'score_manifest.json').read_text())
for n,h in sm['files'].items():
    assert hashlib.sha256((sc/n).read_bytes()).hexdigest()==h,n
sealed=json.loads((run/'sealed_manifest.json').read_text())
for n,h in sealed['files'].items():
    assert hashlib.sha256((run/n).read_bytes()).hexdigest()==h,n
methods=['No-change','Seasonal-naive','AR(1)','SARIMA','TimesFM-3 target-only','B3 multivariate','BEAH rolling','Direct rolling','Fixed candidate','Fixed candidate + gate']
seed_methods={'BEAH rolling','Direct rolling','Fixed candidate + gate'}
horizons=(1,3,6,12)
# Reconstruct canonical target values independently from the source panel and official extension.
targets={}
for r in csv.DictReader((base/'source_data/unified_cpi_panel.csv').open(newline='')):
    m=r['reference_month']
    if '2018-01'<=m<='2024-12' and r['first_release_yoy_pct']:
        targets[m]=round(float(r['first_release_yoy_pct']),1)
for r in csv.DictReader((base/'score_inputs/score_target_extension_2025_2026.csv').open(newline='')):
    targets[r['reference_month']]=round(float(r['first_release_yoy_pct']),1)
assert len(targets)==105
pred=defaultdict(dict)
for r in csv.DictReader((run/'model_forecasts_unscored.csv').open(newline='')):
    if r['status']!='FORECAST_COMPLETE': continue
    pred[(r['model'],int(r['horizon_months']))][(r['origin_id'],r['seed'])]=(float(r['prediction']),r['target_month'])
losses={}
for model in methods:
    for h in horizons:
        keys=pred[(model,h)]
        origins=sorted({o for o,s in keys})
        byorigin={}
        for o in origins:
            seedset=sorted(s for oo,s in keys if oo==o)
            actual_month=keys[(o,seedset[0])][1]
            assert actual_month in targets
            assert len({keys[(o,s)][1] for s in seedset})==1
            errs=[abs(keys[(o,s)][0]-targets[actual_month]) for s in seedset]
            byorigin[o]=sum(errs)/len(errs)
        losses[(model,h)]=byorigin
summary={(r['model'],int(r['horizon_months'])):r for r in csv.DictReader((sc/'score_summary_by_method_horizon.csv').open(newline=''))}
assert len(summary)==40
for k,byorigin in losses.items():
    calc=sum(byorigin.values())/len(byorigin)
    reported=float(summary[k]['native_MAE_pp'])
    assert len(byorigin)==int(summary[k]['native_origins'])
    assert abs(calc-reported)<6e-9,(k,calc,reported)
    common=set.intersection(*(set(losses[(m,k[1])]) for m in methods))
    assert len(common)==99
    common_mae=sum(byorigin[o] for o in common)/len(common)
    assert abs(common_mae-float(summary[k]['all10_common_MAE_pp']))<6e-9,(k,common_mae)
# Reconstruct all descriptive pairwise summaries.
pairs={(r['method_a'],r['method_b'],int(r['horizon_months'])):r for r in csv.DictReader((sc/'pairwise_method_comparisons.csv').open(newline=''))}
assert len(pairs)==180
for h in horizons:
    for i,a in enumerate(methods):
        for b in methods[i+1:]:
            common=sorted(set(losses[(a,h)])&set(losses[(b,h)]))
            diffs=[losses[(a,h)][o]-losses[(b,h)][o] for o in common]
            r=pairs[(a,b,h)]
            assert int(r['pairwise_common_origins'])==len(common)
            assert abs(sum(diffs)/len(diffs)-float(r['mean_absolute_loss_difference_a_minus_b_pp']))<6e-9
            assert int(r['a_lower_loss_origins'])==sum(d < -1e-12 for d in diffs)
            assert int(r['b_lower_loss_origins'])==sum(d > 1e-12 for d in diffs)
            assert int(r['ties'])==sum(abs(d)<=1e-12 for d in diffs)
# Check release audit has no numeric target field and contains 105 unique official target months.
targ=list(csv.DictReader((sc/'target_access_audit.csv').open(newline='')))
assert len(targ)==105 and len({r['target_month'] for r in targ})==105
assert all(r['raw_target_value_included']=='False' for r in targ)
assert not any(c in ('first_release_yoy_pct','raw_target_value','actual','target') for c in targ[0])
print('INDEPENDENT SCORE REBUILD PASS')
print('sealed forecast hashes verified:',len(sealed['files']))
print('score result hashes verified:',len(sm['files']))
print('targets:',len(targets),'loss cells:',sum(map(len,losses.values())),'pairwise rows:',len(pairs))
for h in horizons:
    common=set.intersection(*(set(losses[(m,h)]) for m in methods))
    ranked=sorted((sum(losses[(m,h)][o] for o in common)/len(common),m) for m in methods)
    print('h=',h,'all-method common n=',len(common),'lowest sample MAE=',ranked[0][1],f'{ranked[0][0]:.6f}')

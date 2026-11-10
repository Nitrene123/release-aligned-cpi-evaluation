from pathlib import Path
import csv,json,datetime,calendar,hashlib,importlib.util
import pandas as pd
root=Path(__file__).resolve().parent; bundle=root/'rae_p3_bundle';bundle.mkdir(exist_ok=False)
auditor=root.parent.parent/'supplementary/rae_injection_benchmark/audited_implementation/pit_audit.py'
spec=importlib.util.spec_from_file_location('pit_audit',auditor);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
contexts=[json.loads(x) for x in (root/'contexts_unscored.jsonl').read_text().splitlines() if json.loads(x)['policy']=='P2'];assert len(contexts)==180
sha=json.loads((root/'input_protocol.json').read_text())['source_sha256'];model='TimesFM3_P3'
orig=[];inputs=[];vintages=[];hashes={}
for c in contexts:
 oid='IPT_'+c['origin_month']; y,m=map(int,c['origin_month'].split('-'));end=calendar.monthrange(y,m)[1];av=f'{y:04d}-{m:02d}-{end:02d}T23:59:59+00:00'
 orig.append(dict(model=model,origin_id=oid,origin_month=c['origin_month'],origin_timestamp=c['origin_timestamp_utc'].replace(' ','T').replace('Z','+00:00')))
 rr=[]
 for mo,value in zip(c['months'],c['values']):
  value=format(value,'.17g');vintages.append(dict(source_id='PHILADELPHIA_FED',series_id='IPT_ANNUALIZED_MOM',observation_month=mo,vintage_id=c['vintage'],available_at=av,value=value,source_sha256=sha))
  rr.append(dict(model=model,origin_id=oid,feature='IPT_ANNUALIZED_MOM',observation_month=mo,source_observation_month=mo,masked='false',value=value,available_at=av,source_id='PHILADELPHIA_FED',series_id='IPT_ANNUALIZED_MOM',vintage_id=c['vintage'],source_sha256=sha,mask_reason=''))
 inputs.extend(rr);hashes[oid]=mod.canonical_input_hash(rr)
def write(name,rows):
 with (bundle/name).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('origins.csv',orig);write('inputs.csv',inputs);write('vintages.csv',vintages)
calendar_rows=pd.read_csv(root/'source/g17_calendar_2010_2026.csv',keep_default_na=False).set_index('release_month')['dates'].to_dict()
forecasts=pd.read_csv(root/'timesfm_forecasts_unscored.csv');forecasts=forecasts[forecasts.policy=='P2']; assert len(forecasts)==720
targets=[]
for target in sorted(set(forecasts.target_month)):
 if target=='2025-09': dt='03-December-2025'
 elif target in ('2025-10','2025-11'): dt='23-December-2025'
 else:
  release=str(pd.Period(target,freq='M')+1);dt=calendar_rows[release].split(' and ')[0]
 date=datetime.datetime.strptime(dt,'%d-%B-%Y').date();targets.append(dict(target_month=target,first_release_at=date.isoformat()+'T23:59:59+00:00'))
write('targets.csv',targets)
frows=[]
for r in forecasts.itertuples():
 oid='IPT_'+r.origin_month;frows.append(dict(model=model,origin_id=oid,horizon_months=r.horizon,target_month=r.target_month,prediction=r.prediction,input_hash=hashes[oid]))
write('forecasts.csv',frows)
config=dict(origins_file='origins.csv',inputs_file='inputs.csv',vintages_file='vintages.csv',targets_file='targets.csv',forecasts_file='forecasts.csv',expected_models=[model],expected_horizons=[1,3,6,12],required_features_by_model={model:['IPT_ANNUALIZED_MOM']},require_identical_model_origin_sets=True,vintage_registry_scope='Monthly IPT vintages 2010-01 through 2024-12, for input months used in the 180 forecast contexts.',scope_note='Vintage timestamps are conservative month-end bounds; target timestamps are end-of-release-day bounds from G.17. Transformed values and workbook hashes are supplied; this audit does not independently reconstruct level-to-growth transformations.')
(bundle/'config.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps({'inputs':len(inputs),'vintages':len(vintages),'forecasts':len(frows),'targets':len(targets)}))

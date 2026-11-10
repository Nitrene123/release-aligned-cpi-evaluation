#!/usr/bin/env python3
"""Native TimesFM-3 multivariate CPI baseline using release-aware PIT covariates."""
import csv,json,hashlib,math,time
from pathlib import Path
from portable_paths import project_root
import numpy as np
import torch
from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
R=project_root(__file__);H=12;BATCH=4;FEATURES=('PPI_YOY_PCT','M2_YOY_PCT','BRENT_LOG_YOY_PCT','USDCNY_LOG_YOY_PCT')
def read(rel):
 with (R/rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def month(s):return str(s)[:7]
def addmonth(m,n):y,x=map(int,m.split('-'));z=y*12+x-1+n;return f'{z//12:04d}-{z%12+1:02d}'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
cn={month(x['reference_month']):x for x in read('data/processed/china_cpi_first_release_panel_2012-01_2025-12.csv')}
us={month(x['reference_month']):x for x in read('data/processed/us_cpiaucns_yoy_candidate_first_release_2012_2025.csv')}
us_release={month(x['reference_month']):x for x in read('data/processed/us_cpiaucns_bls_release_calendar_crosswalk_2012_2025.csv')}
manifest=read('data/processed/rolling_origin_horizon_manifest_v1.csv'); covrows=read('data/processed/pit_covariate_inputs_v1.csv')
origin={}
for x in manifest:
 if x['origin_split'] in ('warmup_train','validation','bridge_update','test'):
  origin[x['origin_id']]={'country':x['country'],'month':x['origin_month'],'timestamp':x['origin_timestamp_utc'],'split':x['origin_split']}
cov={}
for x in covrows:
 oid=x['origin_id'];m=x['context_reference_month'];feat=x['feature_name']
 cov.setdefault(oid,{}).setdefault(m,{})[feat]=(float(x['feature_value']) if x['feature_value'].strip() else 0.0,x['masked'].lower()=='true')
examples=[]
for oid,o in sorted(origin.items(),key=lambda z:(z[1]['country'],z[1]['month'])):
 panel=cn if o['country']=='CN' else us; target=[]; cvals=[]; cmask=[]
 months=sorted(m for m in panel if m<=o['month'])
 for m in months:
  r=panel[m];v=r.get('first_release_yoy_pct','') if o['country']=='CN' else r.get('yoy_percent','')
  if not v or not math.isfinite(float(v)):raise AssertionError(f'invalid target context {oid} {m}')
  target.append(float(v)); row=cov[oid].get(m)
  if row is None:raise AssertionError(f'covariate row missing {oid} {m}')
  cvals.append([row[f][0] for f in FEATURES]);cmask.append([row[f][1] for f in FEATURES])
 examples.append({'id':oid,'meta':o,'months':months,'target':np.asarray(target,np.float32),'cov':np.asarray(cvals,np.float32).T,'covmask':np.asarray(cmask,np.bool_).T})

ckpt=R/'weights';weight=ckpt/'model.safetensors';expected=(ckpt/'model.safetensors.sha256.expected').read_text().strip();actual=sha(weight)
if expected and actual!=expected:raise RuntimeError('checkpoint SHA256 mismatch')
if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='NVIDIA A800-SXM4-80GB':raise RuntimeError('GPU0 not expected A800')
t0=time.time();model=TimesFM3Forecaster.from_pretrained(str(ckpt),device='cuda:0',per_core_batch_size=BATCH,local_files_only=True)
if model.config.quantiles[model.config.median_quantile_index]!=0.5:raise RuntimeError('median index is not q50')
outputs={}
for start in range(0,len(examples),BATCH):
 batch=examples[start:start+BATCH];L=max(len(x['target']) for x in batch)
 tgt=np.zeros((len(batch),1,L),dtype=np.float32);pc=np.zeros((len(batch),len(FEATURES),L),dtype=np.float32);pm=np.ones((len(batch),len(FEATURES),L),dtype=np.bool_);gm=np.ones((len(batch),L),dtype=np.bool_)
 for i,x in enumerate(batch):
  n=len(x['target']);sl=slice(L-n,L);tgt[i,0,sl]=x['target'];pc[i,:,sl]=x['cov'];pm[i,:,sl]=x['covmask'];gm[i,sl]=False
 with torch.inference_mode():
  logits=model.model.decode(target=torch.from_numpy(tgt).to(model.device),horizon=H,past_only_covariates=torch.from_numpy(pc).to(model.device),past_only_mask=torch.from_numpy(pm).to(model.device),mask=torch.from_numpy(gm).to(model.device))
 raw=logits[:,0,:H,:].detach().float().cpu().numpy();raw=np.sort(raw,axis=-1)
 if raw.shape!=(len(batch),H,9) or not np.isfinite(raw).all():raise AssertionError(f'bad output shape/data {raw.shape}')
 for x,q in zip(batch,raw):outputs[x['id']]=q

# Target release map for all forecast horizons.
labelmap={'CN':cn,'US':us};
rows=[]
for x in examples:
 oid=x['id'];o=x['meta'];q=outputs[oid]
 for h in range(1,H+1):
  tm=addmonth(o['month'],h);r=labelmap[o['country']].get(tm)
  if r is None:status,val,avail='outside_observed_target_panel','',''
  elif o['country']=='CN':val=r.get('first_release_yoy_pct','').strip();avail=r.get('available_at_conservative_local','');status='observed' if val else 'missing_label'
  else:val=r.get('yoy_percent','').strip();avail=us_release[tm].get('bls_release_timestamp_utc','');status='observed' if val else ('missing_current_index' if r.get('status')=='missing_current_index' else 'missing_label')
  row={'country':o['country'],'origin_id':oid,'origin_month':o['month'],'origin_timestamp_utc':o['timestamp'],'origin_split':o['split'],'horizon_months':h,'registered_eval_horizon':h in (1,3,6,12),'target_month':tm,'target_yoy_first_release_pct':val,'target_status':status,'target_available_at':avail,'valid_for_scoring':bool(status=='observed' and val),'context_length':len(x['target']),'num_unmasked_covariate_cells':int((~x['covmask']).sum()),'point_forecast_q50':float(q[h-1,4])}
  for j,qn in enumerate((10,20,30,40,50,60,70,80,90)):row[f'q{qn}']=float(q[h-1,j])
  rows.append(row)
# Verify the test output row validity against the frozen origin manifest.
by={(r['country'],r['origin_month'],int(r['horizon_months'])):r for r in manifest}
for r in rows:
 m=by.get((r['country'],r['origin_month'],int(r['horizon_months'])))
 if m and (m['valid_for_scoring'].lower()=='true')!=r['valid_for_scoring']:raise AssertionError('label validity differs from frozen manifest')
res=R/'results';res.mkdir(exist_ok=True);pred=res/'timesfm3_multivariate_predictions_v1.csv'
with pred.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=[]
for co in ('CN','US'):
 for h in (1,3,6,12):
  rr=[r for r in rows if r['country']==co and r['origin_split']=='test' and r['horizon_months']==h and r['valid_for_scoring']]
  e=[float(r['point_forecast_q50'])-float(r['target_yoy_first_release_pct']) for r in rr]
  metrics.append({'country':co,'horizon_months':h,'n':len(e),'mae':sum(abs(v) for v in e)/len(e) if e else '', 'rmse':math.sqrt(sum(v*v for v in e)/len(e)) if e else '','origin_start':min((r['origin_month'] for r in rr),default=''),'origin_end':max((r['origin_month'] for r in rr),default=''),'model':'TimesFM-3 native multivariate, 4 PIT covariates'})
with (res/'timesfm3_multivariate_test_metrics_v1.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
meta={'model':'TimesFM-3 official PyTorch checkpoint; direct decode with past_only_covariates and past_only_mask','checkpoint_sha256':actual,'device':'cuda:0 / NVIDIA A800-SXM4-80GB','cuda_visible_devices':'0','num_origins':len(examples),'num_month_ahead_predictions':len(rows),'target_context':'first-release CPI YoY through each actual origin','covariates':list(FEATURES),'covariate_input':'data/processed/pit_covariate_inputs_v1.csv','mask_semantics':'True means masked/unavailable; masked entries zeroed and supplied via official past_only_mask','horizon_months':H,'quantiles':model.config.quantiles,'point_forecast':'q50','elapsed_seconds':round(time.time()-t0,3),'origin_manifest_sha256':sha(R/'data/processed/rolling_origin_horizon_manifest_v1.csv'),'covariate_matrix_sha256':sha(R/'data/processed/pit_covariate_inputs_v1.csv'),'prediction_sha256':sha(pred)}
(res/'timesfm3_multivariate_run_metadata_v1.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'metadata':meta,'test_metrics':metrics},ensure_ascii=False,indent=2))

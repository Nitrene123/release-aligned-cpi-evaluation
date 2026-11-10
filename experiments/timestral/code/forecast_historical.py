"""Target-free historical Timestral forecasts from audited origin-specific CPI contexts."""
from __future__ import annotations
import csv, hashlib, json, math, platform, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from timestral import TimestralForecaster

ROOT=Path('/workspace/EconoFM_CPI')
HERE=ROOT/'revision_experiments_v2/timestral_cpi_baseline_20261109'
CN=ROOT/'revision_experiments_v2/rolling_foundation_matched_v2/runs/run_20261030_unified_full_v2'
US=ROOT/'stage6_us_cpi_crosscountry_v1/inputs_v1_attempt4'
MODEL=Path('/workspace/timestral/baseline/models')
SRC=Path('/workspace/timestral/baseline/timestral-main/src/timestral')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def dt(x):return datetime.fromisoformat(x.replace('Z','+00:00')).astimezone(timezone.utc)

def monthadd(x,n):
    y,m=map(int,x.split('-'));v=12*y+m-1+n
    return f'{v//12:04d}-{v%12+1:02d}'

def canonical(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def main():
    assert sha(MODEL/'model.safetensors')=='599fafa73cadbbc800b5a346ca900fa741ee3e5fd22e3b4a4430e151e74f4bc2'
    assert sha(MODEL/'config.json')=='7bb4c042fb2d2ed6469b6758ba380dfeda3ab32df47690a87aec1527bdae79bf'
    paths={'cn_plan':CN/'planned_origin_horizon_keys.csv','cn_context':CN/'cpi_input_snapshots.csv',
           'cn_index':CN/'evaluation_origin_input_index.csv','us_plan':US/'planned_target_origin_horizons.csv',
           'us_index':US/'origin_input_manifest.csv','us_access':US/'cpi_input_access_ledger.csv'}
    hashes={k:sha(v) for k,v in paths.items()}
    sealed=json.loads((CN/'sealed_manifest.json').read_text())['files']
    for key,file in [('cn_plan','planned_origin_horizon_keys.csv'),('cn_context','cpi_input_snapshots.csv'),('cn_index','evaluation_origin_input_index.csv')]:
        assert hashes[key]==sealed[file], (key,'archived hash mismatch')
    plans={};by_origin=defaultdict(list)
    for country,key in [('CN','cn_plan'),('US','us_plan')]:
        for r in read(paths[key]):
            origin=r['origin_id'];h=int(r['horizon_months']);target=r['target_month'];month=r['origin_month'];clock=r.get('origin_timestamp_utc') or r['origin_clock_utc']
            k=(country,origin,h,target)
            assert k not in plans and h in (1,3,6,12) and monthadd(month,h)==target
            plans[k]=(month,clock);by_origin[(country,origin)].append((h,target,month,clock))
    assert sum(1 for k in plans if k[0]=='CN')==420 and sum(1 for k in plans if k[0]=='US')==336
    assert len([k for k in by_origin if k[0]=='CN'])==116 and len([k for k in by_origin if k[0]=='US'])==95
    cn_index={r['origin_id']:r for r in read(paths['cn_index'])};contexts=defaultdict(list)
    for r in read(paths['cn_context']):
        if ('CN',r['origin_id']) in by_origin:
            assert r['feature']=='CPI_YOY_PCT' and r['masked'].lower()=='false'
            assert dt(r['available_at'])<dt(r['origin_timestamp_utc'])
            contexts[('CN',r['origin_id'])].append(r)
    prepared={}
    for (country,oid),req in sorted(by_origin.items()):
        assert len({(r[2],r[3]) for r in req})==1
        month,clock=req[0][2:]
        if country=='CN':
            rs=sorted(contexts[(country,oid)],key=lambda r:r['observation_month'])
            assert rs and rs[-1]['observation_month']==month and all(dt(r['origin_timestamp_utc'])==dt(clock) for r in rs)
            triples=[(r['observation_month'],float(r['value']),r['available_at']) for r in rs]
            assert len(triples)==int(cn_index[oid]['cpi_context_length']) and canonical(triples)==cn_index[oid]['cpi_context_sha256']
            assert all(monthadd(triples[0][0],i)==r[0] for i,r in enumerate(triples))
            x=np.asarray([r[1] for r in triples],dtype=np.float32)
            input_hash=cn_index[oid]['cpi_context_sha256']
        else:
            idx={r['origin_id']:r for r in read(paths['us_index'])}[oid]
            assert idx['origin_reference_month']==month and dt(idx['origin_clock_utc'])==dt(clock)
            f=US/idx['input_file'];assert sha(f)==idx['input_sha256']
            with np.load(f,allow_pickle=False) as z:x=z['target_yoy_context'].astype(np.float32).reshape(-1)
            assert len(x)==int(idx['context_length']) and idx['context_end']==month
            input_hash=idx['input_sha256']
        assert len(x)>=60 and np.isfinite(x).all()
        prepared[(country,oid)]=(x,input_hash,req)
    us_ledger=defaultdict(list)
    for r in read(paths['us_access']):
        oid='US_'+r['origin_reference_month']
        if ('US',oid) in prepared:us_ledger[oid].append(r)
    for (country,oid),(x,_,req) in prepared.items():
        if country!='US':continue
        rows=sorted(us_ledger[oid],key=lambda r:r['context_reference_month']);clock=req[0][3]
        assert len(rows)==len(x) and all(r['masked'].lower()=='false' and dt(r['available_at_utc'])<dt(clock) and dt(r['origin_clock_utc'])==dt(clock) for r in rows)
        assert all(monthadd(rows[0]['context_reference_month'],i)==r['context_reference_month'] for i,r in enumerate(rows))
        assert np.max(np.abs(np.asarray([float(r['value']) for r in rows])-x.astype('float64')))<1e-6
    assert hashes=={k:sha(v) for k,v in paths.items()}
    code_hashes={p.name:sha(p) for p in sorted(SRC.glob('*.py'))}
    out=HERE/'historical_forecasts';out.mkdir(exist_ok=False)
    started={'status':'INFERENCE_STARTED','at_utc':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
        'observation_mask_convention_sha256':sha(HERE/'INPUT_MASK_CONVENTION.md'),'model_sha256':sha(MODEL/'model.safetensors'),
        'config_sha256':sha(MODEL/'config.json'),'input_files_sha256':hashes,'model_source_sha256':code_hashes}
    (out/'run_started.json').write_text(json.dumps(started,indent=2)+'\n')
    torch.set_num_threads(4);torch.manual_seed(42);model=TimestralForecaster.from_pretrained(str(MODEL),device='cpu',local_files_only=True)
    forecasts=[];audit=[];t0=time.perf_counter()
    for i,((country,oid),(x,input_hash,req)) in enumerate(sorted(prepared.items()),1):
        try:
            # These archived histories are fully observed; explicitly set Timestral observed mask to True.
            obs=np.ones((1,1,len(x)),dtype=bool)
            result=model.forecast(x.reshape(1,1,-1),prediction_length=12,observation_mask=obs,symmetric_averaging=True,clip_nonnegative=True)
            values=result.median.detach().cpu().numpy().reshape(-1);assert len(values)==12 and np.isfinite(values).all()
            status='FORECAST_COMPLETE';reason=''
        except Exception as e:
            values=None;status='FORECAST_FAILED';reason=type(e).__name__+': '+str(e)
        audit.append({'country':country,'origin_id':oid,'context_length':len(x),'input_sha256':input_hash,'status':status,'reason':reason})
        for h,target,month,clock in req:
            forecasts.append({'country':country,'model':'Timestral-123M target-only','origin_id':oid,'origin_month':month,
                'origin_timestamp_utc':dt(clock).isoformat(),'target_month':target,'horizon_months':h,
                'prediction':format(float(values[h-1]),'.17g') if values is not None else '',
                'status':status,'input_sha256':input_hash})
        if i%25==0 or i==len(prepared):print(f'{i}/{len(prepared)} historical origins; {time.perf_counter()-t0:.1f}s',flush=True)
    assert len(forecasts)==756 and len(audit)==211
    for filename,records in [('forecasts.csv',forecasts),('origin_audit.csv',audit)]:
        with (out/filename).open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    assert sha(MODEL/'model.safetensors')==started['model_sha256'] and hashes=={k:sha(v) for k,v in paths.items()}
    assert code_hashes=={p.name:sha(p) for p in sorted(SRC.glob('*.py'))}
    manifest={**started,'status':'SEALED_HISTORICAL_FORECASTS_UNSCORED','completed_utc':datetime.now(timezone.utc).isoformat(),
        'origins':len(audit),'forecast_rows':len(forecasts),'failed_origins':sum(r['status']!='FORECAST_COMPLETE' for r in audit),
        'elapsed_seconds':round(time.perf_counter()-t0,3),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'cpu_threads':4,'future_targets_read':False,'p1_accessed':False,'model_updates':False,
        'files':{n:sha(out/n) for n in ('run_started.json','forecasts.csv','origin_audit.csv')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'rows':len(forecasts),'failed_origins':manifest['failed_origins'],'manifest_sha256':sha(out/'manifest.json')}))

if __name__=='__main__':main()

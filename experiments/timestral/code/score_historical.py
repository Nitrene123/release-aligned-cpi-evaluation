"""Score previously sealed Timestral historical forecasts on archived matched keys."""
from __future__ import annotations
import csv, hashlib, json, math, statistics
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CN=ROOT/'manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a'
US=ROOT/'stage6_us_cpi_crosscountry_v1'
USR=US/'runs/us_cpi_crosscountry_run_v1_attempt2'
ETS=ROOT/'revision_experiments_v2/ets_theta_rolling_20261109/output'
H=(1,3,6,12)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def norm_time(s,cn=False):
    d=datetime.fromisoformat(s.replace('Z','+00:00'))
    if d.tzinfo is None:
        assert cn;from datetime import timedelta;d=d.replace(tzinfo=timezone(timedelta(hours=8)))
    return d.astimezone(timezone.utc)
def key(r):return (r['origin_id'],r['origin_month'],norm_time(r['origin_timestamp_utc']).isoformat(),r['target_month'],int(r['horizon_months']))
def write(path,rows):
    assert rows
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    archive=HERE/'historical_forecasts';man=json.loads((archive/'manifest.json').read_text())
    assert man['status']=='SEALED_HISTORICAL_FORECASTS_UNSCORED' and man['failed_origins']==0
    assert all(sha(archive/n)==h for n,h in man['files'].items())
    assert sha(HERE/'PROTOCOL.md')==man['protocol_sha256'] and sha(HERE/'forecast_historical.py')==man['code_sha256']
    data={
        'CN':{'old':CN/'score_v2_clean_20261031/origin_method_losses.csv','panel':CN/'source_data/unified_cpi_panel.csv',
              'target':CN/'score_inputs/score_target_extension_2025_2026.csv',
              'old_forecasts':CN/'runs/run_20261030_unified_full_v2/source/statistical_forecast_predictions.csv',
              'expected':99,'planned':105},
        'US':{'old':USR/'score_us_2018_2024_once_v1/origin_method_losses.csv',
              'target':US/'data/us_cpi_first_release_targets_2018_2024.csv',
              'old_forecasts':USR/'statistics/statistical_forecasts_unscored.csv',
              'expected':84,'planned':84}}
    forecast=read(archive/'forecasts.csv');assert len(forecast)==756
    result=[];summary=[];comparisons=[];intervals=[];input_hashes={'forecast_manifest':sha(archive/'manifest.json')}
    for country in ('CN','US'):
        cfg=data[country];oldmeta=json.loads((ETS/country/'score/score_manifest.json').read_text())
        assert sha(cfg['old'])==oldmeta['old_origin_losses_sha256'] and sha(cfg['target'])==oldmeta['score_targets_sha256']
        additional=ETS/country/'score/new_origin_losses.csv'
        assert sha(additional)==oldmeta['outputs']['new_origin_losses.csv']
        if country=='CN':
            prior=json.loads((ETS/'CN/forecast/forecast_manifest.json').read_text())
            assert sha(cfg['panel'])==prior['source_sha256']['training']
            targets={r['reference_month']:(float(Decimal(r['first_release_yoy_pct']).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)),norm_time(r['available_at_conservative_local'],True))
                for r in read(cfg['panel']) if '2018-01'<=r['reference_month']<='2024-12'}
            targets.update({r['reference_month']:(float(Decimal(r['first_release_yoy_pct']).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)),norm_time(r['available_at_utc'])) for r in read(cfg['target'])})
            column='mean_seed_absolute_error_pp'
        else:
            targets={r['target_reference_month']:(float(r['target_first_release_yoy_pp']),norm_time(r['release_timestamp_utc'])) for r in read(cfg['target'])}
            column='origin_mean_absolute_error_pp'
        for name,path in cfg.items():
            if isinstance(path,Path):input_hashes[country+'_'+name]=sha(path)
        input_hashes[country+'_ets_theta_losses']=sha(additional)
        old={};new={}
        for r in read(cfg['old']):
            k=(r['model'],key(r));assert k not in old;old[k]=float(r[column]);assert math.isfinite(old[k]) and old[k]>=0
        assert len({m for m,k in old})==10
        for r in read(additional):
            k=(r['model'],key(r));assert k not in old;old[k]=float(r['absolute_error_pp'])
        assert len({m for m,k in old})==12
        max_reconstruction=0.0
        for r in read(cfg['old_forecasts']):
            if r['model']!='No-change':continue
            clock=r.get('origin_timestamp_utc') or r['origin_clock_utc']
            k=(r['origin_id'],r['origin_month'],norm_time(clock).isoformat(),r['target_month'],int(r['horizon_months']))
            y,available=targets[k[3]];assert available>norm_time(k[2])
            max_reconstruction=max(max_reconstruction,abs(abs(float(r['prediction'])-y)-old[('No-change',k)]))
        assert max_reconstruction<1e-9
        for r in forecast:
            if r['country']!=country:continue
            assert r['status']=='FORECAST_COMPLETE';k=key(r);assert k not in new
            y,available=targets[k[3]];assert available>norm_time(k[2])
            value=abs(float(r['prediction'])-y);assert math.isfinite(value)
            new[k]=value;result.append({'country':country,'model':r['model'],'origin_id':k[0],'origin_month':k[1],
                'origin_timestamp_utc':k[2],'target_month':k[3],'horizon_months':k[4],'absolute_error_pp':format(value,'.17g')})
        assert len(new)==cfg['planned']*4
        methods=sorted({m for m,k in old})
        for h in H:
            sets={m:{k for mm,k in old if mm==m and k[-1]==h} for m in methods}
            shared12=set.intersection(*(sets[m] for m in methods));shared13=shared12 & {k for k in new if k[-1]==h}
            assert len(shared12)==len(shared13)==cfg['expected'] and shared12==shared13
            pairs=sorted(shared13,key=lambda k:k[3])
            for model in methods+['Timestral-123M target-only']:
                losses=[new[k] if model.startswith('Timestral') else old[(model,k)] for k in pairs]
                summary.append({'country':country,'model':model,'horizon_months':h,'n_common13':len(pairs),'mae_common13_pp':format(statistics.mean(losses),'.12f')})
            for ref in methods:
                d=[new[k]-old[(ref,k)] for k in pairs]
                comparisons.append({'country':country,'model':'Timestral-123M target-only','reference':ref,'horizon_months':h,
                    'n_pairs':len(d),'mean_loss_difference_pp':format(statistics.mean(d),'.12f'),
                    'timestral_lower':sum(x < -1e-12 for x in d),'reference_lower':sum(x > 1e-12 for x in d),'ties':sum(abs(x)<=1e-12 for x in d)})
        print(country,'no-change target reconstruction max',max_reconstruction,'common',cfg['expected'])
    # Additional US B3 comparisons: same circular blocks/seed as the already reported U.S. exploration.
    usnew={(r['target_month'],int(r['horizon_months'])):float(r['absolute_error_pp']) for r in result if r['country']=='US'}
    usb3={(r['target_month'],int(r['horizon_months'])):float(r['origin_mean_absolute_error_pp']) for r in read(data['US']['old']) if r['model']=='B3 multivariate'}
    months=sorted({m for m,h in usb3});assert len(months)==84
    rng=np.random.default_rng(417)
    for h in H:
        d=np.array([usnew[(m,h)]-usb3[(m,h)] for m in months])
        for length in (3,6,12):
            starts=rng.integers(0,84,size=(30000,math.ceil(84/length)))
            ix=((starts[:,:,None]+np.arange(length))%84).reshape(30000,-1)[:,:84]
            lo,hi=np.quantile(d[ix].mean(axis=1),[.025,.975],method='linear')
            intervals.append({'country':'US','contrast':'Timestral-123M target-only minus B3 multivariate','horizon_months':h,'n_pairs':84,
                'mean_difference_pp':format(float(d.mean()),'.12f'),'block_months':length,'lower_pp':format(float(lo),'.12f'),'upper_pp':format(float(hi),'.12f')})
    assert len(result)==756 and len(summary)==104 and len(comparisons)==96 and len(intervals)==12
    out=HERE/'historical_scores';out.mkdir(exist_ok=False)
    for name,rows in [('origin_losses.csv',result),('common13_mae.csv',summary),('paired_comparisons.csv',comparisons),('us_timestral_vs_b3_intervals.csv',intervals)]:write(out/name,rows)
    meta={'status':'RETROSPECTIVE_EXPLORATORY_SCORE_COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User requested Timestral as a baseline','historical_forecasts_manifest_sha256':sha(archive/'manifest.json'),
        'score_code_sha256':sha(Path(__file__)),'input_sha256':input_hashes,'new_loss_rows':len(result),'common_summary_rows':len(summary),
        'pairwise_rows':len(comparisons),'us_interval_rows':len(intervals),'p1_accessed':False,'target_values_exported':False,
        'bootstrap':{'type':'circular moving-block','blocks_months':[3,6,12],'replicates':30000,'seed':417,'pointwise':True},
        'outputs':{n:sha(out/n) for n in ('origin_losses.csv','common13_mae.csv','paired_comparisons.csv','us_timestral_vs_b3_intervals.csv')}}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'status':meta['status'],'new_loss_rows':len(result),'outputs':meta['outputs']}))

if __name__=='__main__':main()

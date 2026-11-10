"""Post-score calendar-preserving paired block sensitivity, Timestral minus B3."""
import csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FILES={
 'CN':ROOT/'manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a/score_v2_clean_20261031/origin_method_losses.csv',
 'US':ROOT/'stage6_us_cpi_crosscountry_v1/runs/us_cpi_crosscountry_run_v1_attempt2/score_us_2018_2024_once_v1/origin_method_losses.csv'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def monthadd(m,n):
 y,x=map(int,m.split('-'));t=y*12+x-1+n;return f'{t//12:04d}-{t%12+1:02d}'

def main():
 score=HERE/'historical_scores';metadata=json.loads((score/'manifest.json').read_text())
 assert all(sha(score/n)==h for n,h in metadata['outputs'].items())
 new=read(score/'origin_losses.csv');outrows=[]
 for country,n in [('CN',105),('US',84)]:
  assert sha(FILES[country])==metadata['input_sha256'][country+'_old']
  old={(r['target_month'],int(r['horizon_months'])):r for r in read(FILES[country]) if r['model']=='B3 multivariate'}
  added={(r['target_month'],int(r['horizon_months'])):r for r in new if r['country']==country}
  months=[monthadd('2018-01',i) for i in range(n)];rng=np.random.default_rng(417)
  for h in (1,3,6,12):
   d=np.full(n,np.nan)
   for i,m in enumerate(months):
    if (m,h) not in old:continue
    a=added[(m,h)];b=old[(m,h)]
    assert all(a[k]==b[k] for k in ('origin_id','origin_month','target_month','horizon_months'))
    assert datetime.fromisoformat(a['origin_timestamp_utc'])==datetime.fromisoformat(b['origin_timestamp_utc'])
    d[i]=float(a['absolute_error_pp'])-float(b['mean_seed_absolute_error_pp' if country=='CN' else 'origin_mean_absolute_error_pp'])
   assert np.isfinite(d).sum()==(99 if country=='CN' else 84)
   for block in (3,6,12):
    starts=rng.integers(0,n,size=(30000,math.ceil(n/block)))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(30000,-1)[:,:n]
    drawn=d[ix];counts=np.isfinite(drawn).sum(axis=1);assert (counts>0).all()
    means=np.nansum(drawn,axis=1)/counts;lo,hi=np.quantile(means,[.025,.975],method='linear')
    outrows.append({'country':country,'contrast':'Timestral-123M target-only minus B3 multivariate',
      'horizon_months':h,'planned_months':n,'paired_origins':int(np.isfinite(d).sum()),
      'mean_difference_pp':format(float(np.nanmean(d)),'.12f'),'block_months':block,
      'lower_pp':format(float(lo),'.12f'),'upper_pp':format(float(hi),'.12f')})
 # Existing US scores were independently calculated in the scorer.
 prior={(int(r['horizon_months']),int(r['block_months'])):r for r in read(score/'us_timestral_vs_b3_intervals.csv')}
 for r in outrows:
  if r['country']=='US':
   p=prior[(r['horizon_months'],r['block_months'])]
   assert abs(float(r['lower_pp'])-float(p['lower_pp']))<1e-10 and abs(float(r['upper_pp'])-float(p['upper_pp']))<1e-10
 out=HERE/'paired_uncertainty';out.mkdir(exist_ok=False)
 with (out/'paired_block_intervals.csv').open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
 report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'analysis':'post-score exploratory; pointwise, unadjusted',
  'description':'Circular blocks over complete target-month calendar; absent matched months remain missing.',
  'draws':30000,'seed':417,'block_months':[3,6,12],'rows':len(outrows),
  'score_manifest_sha256':sha(score/'manifest.json'),'input_code_sha256':sha(Path(__file__)),
  'intervals_sha256':sha(out/'paired_block_intervals.csv'),'p1_accessed':False,'targets_read':False}
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()

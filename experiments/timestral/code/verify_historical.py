"""Independent check of the Timestral scores, common keys, and saved MAEs."""
import csv,hashlib,json,math,statistics
from datetime import datetime,timezone,timedelta
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CN=ROOT/'manuscript/ars_full_v2_4/tmp/cn_v2_27_repack_1031a'
US=ROOT/'stage6_us_cpi_crosscountry_v1'
U=US/'runs/us_cpi_crosscountry_run_v1_attempt2/score_us_2018_2024_once_v1/origin_method_losses.csv'
ETS=ROOT/'revision_experiments_v2/ets_theta_rolling_20261109/output'

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return (r['origin_id'],r['origin_month'],datetime.fromisoformat(r['origin_timestamp_utc'].replace('Z','+00:00')).astimezone(timezone.utc).isoformat(),r['target_month'],int(r['horizon_months']))
def one_decimal(s):return float(Decimal(s).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))

def main():
 archive=HERE/'historical_forecasts';scores=HERE/'historical_scores'
 am=json.loads((archive/'manifest.json').read_text());sm=json.loads((scores/'manifest.json').read_text())
 assert sm['historical_forecasts_manifest_sha256']==sha(archive/'manifest.json')
 assert all(sha(archive/n)==h for n,h in am['files'].items()) and all(sha(scores/n)==h for n,h in sm['outputs'].items())
 panel=CN/'source_data/unified_cpi_panel.csv';extension=CN/'score_inputs/score_target_extension_2025_2026.csv';us=US/'data/us_cpi_first_release_targets_2018_2024.csv'
 assert sha(panel)==sm['input_sha256']['CN_panel'] and sha(extension)==sm['input_sha256']['CN_target'] and sha(us)==sm['input_sha256']['US_target']
 targets={'CN':{r['reference_month']:one_decimal(r['first_release_yoy_pct']) for r in read(panel) if '2018-01'<=r['reference_month']<='2024-12'},
          'US':{r['target_reference_month']:float(r['target_first_release_yoy_pp']) for r in read(us)}}
 targets['CN'].update({r['reference_month']:one_decimal(r['first_release_yoy_pct']) for r in read(extension)})
 forecasts={(r['country'],key(r)):r for r in read(archive/'forecasts.csv')};losses={(r['country'],key(r)):float(r['absolute_error_pp']) for r in read(scores/'origin_losses.csv')}
 assert len(forecasts)==len(losses)==756 and forecasts.keys()==losses.keys()
 mx=max(abs(abs(float(r['prediction'])-targets[c][k[3]])-losses[(c,k)]) for (c,k),r in forecasts.items());assert mx<1e-12
 summary={(r['country'],r['model'],int(r['horizon_months'])):r for r in read(scores/'common13_mae.csv')};assert len(summary)==104
 paired={(r['country'],r['reference'],int(r['horizon_months'])):r for r in read(scores/'paired_comparisons.csv')};assert len(paired)==96
 max_mae=max_pair=0.0
 for country,oldpath,col in [('CN',CN/'score_v2_clean_20261031/origin_method_losses.csv','mean_seed_absolute_error_pp'),('US',U,'origin_mean_absolute_error_pp')]:
  assert sha(oldpath)==sm['input_sha256'][country+'_old']
  old={(r['model'],key(r)):float(r[col]) for r in read(oldpath)}
  extra=ETS/country/'score/new_origin_losses.csv';assert sha(extra)==sm['input_sha256'][country+'_ets_theta_losses']
  old.update({(r['model'],key(r)):float(r['absolute_error_pp']) for r in read(extra)})
  models={m for m,k in old};assert len(models)==12
  for h in (1,3,6,12):
   sets=[{k for m,k in old if m==model and k[-1]==h} for model in models]
   common=set.intersection(*sets)&{k for c,k in losses if c==country and k[-1]==h};assert len(common)==(99 if country=='CN' else 84)
   for model in models|{'Timestral-123M target-only'}:
    arr=[losses[(country,k)] if model.startswith('Timestral') else old[(model,k)] for k in common]
    row=summary[(country,model,h)];assert int(row['n_common13'])==len(common)
    max_mae=max(max_mae,abs(statistics.mean(arr)-float(row['mae_common13_pp'])))
    if not model.startswith('Timestral'):
     differences=[losses[(country,k)]-old[(model,k)] for k in common];q=paired[(country,model,h)]
     assert (int(q['timestral_lower']),int(q['reference_lower']),int(q['ties']))==(sum(x < -1e-12 for x in differences),sum(x > 1e-12 for x in differences),sum(abs(x)<=1e-12 for x in differences))
     max_pair=max(max_pair,abs(statistics.mean(differences)-float(q['mean_loss_difference_pp'])))
 assert max_mae<1e-10 and max_pair<1e-10
 out=HERE/'historical_verification.json';assert not out.exists()
 report={'status':'PASS','forecast_losses_recomputed':756,'common13_maes_recomputed':104,'paired_comparisons_recomputed':96,
  'max_origin_loss_difference_pp':mx,'max_mae_difference_pp':max_mae,'max_pair_difference_pp':max_pair,
  'score_manifest_sha256':sha(scores/'manifest.json'),'verifier_sha256':sha(Path(__file__)),
  'target_values_exported':False,'p1_accessed':False}
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

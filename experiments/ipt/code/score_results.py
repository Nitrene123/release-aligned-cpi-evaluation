from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
root=Path(__file__).resolve().parent
pred_files=[root/'statistical_forecasts_unscored.csv',root/'timesfm_forecasts_unscored.csv']
if not all(p.is_file() for p in pred_files):raise RuntimeError('prediction archives incomplete')
if (root/'score_summary.csv').exists():raise RuntimeError('refusing overwrite')
target_workbook=root/'source'/'ipt_first_second_third.xlsx'
if not target_workbook.is_file():raise RuntimeError(f'missing bundled target workbook: {target_workbook}')
lock={'prediction_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pred_files},'target_workbook':'source/ipt_first_second_third.xlsx','target_workbook_sha256':hashlib.sha256(target_workbook.read_bytes()).hexdigest(),'bootstrap':'circular blocks, length 6, 5000 replicates, seed 20261110; descriptive post hoc'}
(root/'pre_score_manifest.json').write_text(json.dumps(lock,indent=2)+'\n')
pred=pd.concat([pd.read_csv(p) for p in pred_files],ignore_index=True)
assert len(pred)==6480 and not pred.duplicated(['model','policy','origin_month','horizon']).any()
targets=pd.read_excel(target_workbook,sheet_name='DATA',header=4)
targets['target_month']=targets['Date'].astype(str).str.replace(':','-',regex=False)
targets=targets[['target_month','First']].rename(columns={'First':'target'})
assert not targets.duplicated('target_month').any()
d=pred.merge(targets,on='target_month',how='left',validate='many_to_one');d['prediction']=pd.to_numeric(d['prediction'],errors='coerce');d['target']=pd.to_numeric(d['target'],errors='coerce')
d['valid']=(d.status=='ok') & np.isfinite(d.prediction) & np.isfinite(d.target)
d['abs_error']=(d.prediction-d.target).abs(); d['squared_error']=(d.prediction-d.target)**2
models=['AR1','SARIMA','TimesFM3'];policies=['P0','P1','P2'];horizons=[1,3,6,12]
summ=[]; paired=[];rng=np.random.default_rng(20261110)
for h in horizons:
 x=d[d.horizon==h]; sets=[set(x[(x.model==m)&(x.policy==p)&x.valid].origin_month) for m in models for p in policies];common=sorted(set.intersection(*sets))
 for m in models:
  for p in policies:
   native=x[(x.model==m)&(x.policy==p)&x.valid]; z=native[native.origin_month.isin(common)].sort_values('origin_month')
   summ.append(dict(model=m,policy=p,horizon=h,n_native=len(native),n_common=len(z),mae=z.abs_error.mean(),rmse=math.sqrt(z.squared_error.mean())))
  base=x[(x.model==m)&(x.policy=='P2')&x.origin_month.isin(common)].sort_values('origin_month')
  for p in ('P0','P1'):
   alt=x[(x.model==m)&(x.policy==p)&x.origin_month.isin(common)].sort_values('origin_month')
   assert list(base.origin_month)==list(alt.origin_month)
   diff=alt.abs_error.to_numpy()-base.abs_error.to_numpy();n=len(diff);boot=[]
   for _ in range(5000):
    starts=rng.integers(0,n,size=math.ceil(n/6));ind=np.concatenate([(s+np.arange(6))%n for s in starts])[:n];boot.append(diff[ind].mean())
   paired.append(dict(model=m,horizon=h,contrast=p+'-P2',n=n,mean_abs_loss_difference=diff.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975)))
pd.DataFrame(summ).to_csv(root/'score_summary.csv',index=False,float_format='%.12g');pd.DataFrame(paired).to_csv(root/'paired_intervals.csv',index=False,float_format='%.12g')
d.drop(columns=['target']).to_csv(root/'derived_losses_target_free.csv',index=False,float_format='%.12g')
print(pd.DataFrame(summ).to_string(index=False));print(pd.DataFrame(paired).to_string(index=False))

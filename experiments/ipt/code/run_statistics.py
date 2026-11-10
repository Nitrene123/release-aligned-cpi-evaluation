import csv,hashlib,json,math,time,warnings
from pathlib import Path
import numpy as np
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

root=Path(__file__).resolve().parent
source=root/'contexts_unscored.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((root/'input_protocol.json').read_text())['contexts_sha256']
assert statsmodels.__version__=='0.14.4'
out=root/'statistical_forecasts_unscored.csv'
if out.exists(): raise RuntimeError('refusing overwrite')
spec=dict(order=(1,0,0),seasonal_order=(1,0,0,12),trend='c',enforce_stationarity=True,enforce_invertibility=True,simple_differencing=False,measurement_error=False,time_varying_regression=False,mle_regression=True,concentrate_scale=False,missing='none',initialization='stationary')
fields=['model','policy','origin_month','horizon','target_month','prediction','status','converged','context_sha256','error']
import pandas as pd
with source.open(encoding='utf8') as f, out.open('x',encoding='utf8',newline='') as g:
 w=csv.DictWriter(g,fieldnames=fields);w.writeheader()
 for i,line in enumerate(f,1):
  row=json.loads(line); y=np.asarray(row['values'],dtype=np.float64); n=13 if row['policy']!='P0' else 12
  x=np.column_stack((np.ones(len(y)-1),y[:-1])); a,b=np.linalg.lstsq(x,y[1:],rcond=None)[0]; ar=[]; v=y[-1]
  for j in range(n): v=a+b*v; ar.append(float(v))
  try:
   with warnings.catch_warnings(record=True):
    warnings.simplefilter('always'); fit=SARIMAX(y,**spec).fit(method='lbfgs',maxiter=500,disp=False,pgtol=1e-8,factr=1e7)
   sf=np.asarray(fit.forecast(steps=n)); conv=bool(fit.mle_retvals.get('converged',False)); err=''
   if not conv or not np.isfinite(sf).all(): raise RuntimeError('nonconvergence or nonfinite prediction')
  except Exception as e: sf=None;conv=False;err=str(e)
  for model,fc in [('AR1',ar),('SARIMA',sf)]:
   for h in (1,3,6,12):
    step=h+(row['policy']!='P0'); ok=fc is not None and math.isfinite(float(fc[step-1]))
    w.writerow(dict(model=model,policy=row['policy'],origin_month=row['origin_month'],horizon=h,target_month=str(pd.Period(row['origin_month'],freq='M')+h),prediction=float(fc[step-1]) if ok else '',status='ok' if ok else 'failed',converged=conv if model=='SARIMA' else '',context_sha256=row['context_sha256'],error=err if model=='SARIMA' else ''))
  g.flush()
  if i%10==0: print(f'{i}/540 contexts',flush=True)

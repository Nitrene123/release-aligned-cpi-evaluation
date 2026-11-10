from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('workbook',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
df=pd.read_excel(a.workbook,sheet_name='ipt');months=pd.PeriodIndex(df['DATE'].str.replace(':','-',regex=False),freq='M');latest=df.columns[-1]
rows=[]
for m in pd.period_range('2010-01','2024-12',freq='M'):
 for pol in ('P0','P1','P2'):
  selected=latest if pol in ('P0','P1') else f'IPT{m.year%100:02d}M{m.month}'
  vals=pd.to_numeric(df[selected],errors='coerce').to_numpy(dtype=float)
  with np.errstate(invalid='ignore',divide='ignore',over='ignore'): growth=100*((vals[1:]/vals[:-1])**12-1)
  end=m if pol=='P0' else m-1;ix=np.flatnonzero(np.isfinite(growth)&(months[1:]<=end));mm=months[1:][ix]
  assert len(ix)>=24 and mm[-1]==end and len(mm)==mm[-1].ordinal-mm[0].ordinal+1
  values=[float(x) for x in growth[ix]];payload=json.dumps({'months':[str(x) for x in mm],'values':values},sort_keys=True,separators=(',',':'))
  rows.append({'origin_month':str(m),'origin_timestamp_utc':str((m+1).start_time)+'Z','policy':pol,'vintage':selected,'context_start':str(mm[0]),'context_end':str(mm[-1]),'context_length':len(values),'months':[str(x) for x in mm],'values':values,'context_sha256':hashlib.sha256(payload.encode()).hexdigest()})
with a.output.open('x',encoding='utf8') as f:
 for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
print(hashlib.sha256(a.output.read_bytes()).hexdigest())

import argparse,csv,hashlib,json,math,os,time
from pathlib import Path
import numpy as np,torch
from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
root=Path(__file__).resolve().parent
source=root/'contexts_unscored.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((root/'input_protocol.json').read_text())['contexts_sha256']
parser=argparse.ArgumentParser(description='Run TimesFM-3 forecasts from an authorized local checkpoint.')
parser.add_argument('--checkpoint',type=Path,default=os.environ.get('TIMESFM3_CHECKPOINT'))
parser.add_argument('--expected-sha256',default=os.environ.get('TIMESFM3_MODEL_SHA256'))
args=parser.parse_args()
if args.checkpoint is None: parser.error('provide --checkpoint or set TIMESFM3_CHECKPOINT')
ckpt=Path(args.checkpoint).expanduser().resolve(); weight=ckpt/'model.safetensors'
h=hashlib.sha256()
with weight.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
actual_hash=h.hexdigest()
if args.expected_sha256 and actual_hash.lower()!=args.expected_sha256.strip().lower():
 raise RuntimeError('checkpoint SHA-256 does not match supplied expected value')
rows=[json.loads(x) for x in source.read_text().splitlines()]; assert len(rows)==540
model=TimesFM3Forecaster.from_pretrained(str(ckpt),device='cuda:0',per_core_batch_size=4,local_files_only=True)
out=root/'timesfm_forecasts_unscored.csv'
if out.exists():raise RuntimeError('refusing overwrite')
with out.open('x',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=['model','policy','origin_month','horizon','target_month','prediction','status','context_sha256']);w.writeheader()
 for start in range(0,len(rows),24):
  batch=rows[start:start+24]; ids=[f"{r['origin_month']}_{r['policy']}" for r in batch]
  outputs=list(model.predict_batch(contexts=[np.asarray(r['values'],dtype=np.float32) for r in batch],horizon=13,ts_ids=ids,return_quantiles=True,make_positive=False,sort_quantiles=True))
  assert len(outputs)==len(batch)
  for row,pred in zip(batch,outputs):
   for hzn in (1,3,6,12):
    idx=hzn+(row['policy']!='P0')-1; value=float(pred.forecast[idx]); assert math.isfinite(value)
    yy,mm=map(int,row['origin_month'].split('-')); n=yy*12+mm-1+hzn; target=f'{n//12:04d}-{n%12+1:02d}'
    w.writerow(dict(model='TimesFM3',policy=row['policy'],origin_month=row['origin_month'],horizon=hzn,target_month=target,prediction=value,status='ok',context_sha256=row['context_sha256']))
  f.flush(); print(f'{min(start+24,len(rows))}/540 contexts',flush=True)

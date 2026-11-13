"""Controlled feature ablations and simple baselines, on the revised data only."""
from pathlib import Path
import json, sys, shutil, importlib.metadata, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json')); args=ap.parse_args()
PKG=Path(__file__).parent
import model_core as m
import numpy as np
import pandas as pd

c=json.loads(args.config.read_text(encoding='utf-8-sig'))
for key in ['data_dir','db_dir','evaluator','test_input']:
 q=Path(c[key]); c[key]=str((args.config.resolve().parent/q).resolve()) if not q.is_absolute() else str(q)
t=m.utc(c['backtest_start'])
out=Path(c.get('output_dir',PKG/'study_runs'))/'feature_study'
out.mkdir(parents=True,exist_ok=True)
rows=[]
for condition,cols in [('full34',list(range(34))),('user_age4',list(range(4))),('activity90_7',[0,7,12,17,22,27,32])]:
    class StudyData(m.Data):
        def snapshot(self,t,labels=True):
            s=super().snapshot(t,labels)
            return dict(s,base=s['base'][:,cols])
    ds=StudyData(c)
    runner=m.Runner(c,out/condition,ds);runner.out.mkdir(exist_ok=True)
    for seed in [42,43]:
        r=runner.job(t,'control4',seed)
        met={
            'six':m.metric(r['y'],r['predictions']),
            'five':m.metric(r['y'][:,:5],r['predictions'][:,:5]),
        }
        rows.append({'condition':condition,'seed':seed,'features':len(cols),'users':len(r['ids']),'six_score':met['six']['score'],'five_score':met['five']['score']})
    if condition=='full34':
        snap=ds.snapshot(t);ix=np.flatnonzero(snap['cohort']);ids=snap['ids'][ix];y=snap['y'][ix]
        predictions={'all_zero':np.zeros_like(y),'previous_week':(snap['base'][ix][:,[5,10,15,20,25,30]]>0).astype(int)}
        # These counts cover the visible part of the prior seven days, ending t-4h.
        for name,p in predictions.items():
            rows.append({'condition':name,'seed':'deterministic','features':0,'users':len(ids),'six_score':m.metric(y,p)['score'],'five_score':m.metric(y[:,:5],p[:,:5])['score']})
result=pd.DataFrame(rows)
result.to_csv(PKG/'results'/'feature_study.csv',index=False)
versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','catboost','xgboost','scikit-learn','scipy']}
(PKG/'results'/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
print(result.to_string(index=False))

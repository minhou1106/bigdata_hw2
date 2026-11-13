"""Train a matched-lead forecast using the fixed, genuinely available snapshot."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import model_core as m

class EarlyData(m.Data):
    def __init__(self,c):
        super().__init__(c)
        self.lead=m.utc(c['prediction_start'])-m.utc(c['forecast_feature_time'])
        if self.lead<pd.Timedelta(0):raise ValueError('Feature reference must not follow target')
        if m.utc(c['forecast_feature_time'])>self.end:
            raise ValueError('Feature reference must not exceed actual observation cutoff')
        self.early_cache={}

    def snapshot(self,t,labels=True):
        key=(str(t),labels)
        if key in self.early_cache:return self.early_cache[key]
        feature_time=t-self.lead
        # Feature windows, recency, age, availability and cohort all use this anchor.
        # The unobserved gap is never silently interpreted as an observed zero window.
        s=dict(super().snapshot(feature_time,False))
        if labels:
            idx=self.users.set_index('github_user_id').loc[s['ids'],'user_id'].to_numpy()
            y=np.zeros((len(idx),6),dtype=np.int8)
            for j,d in enumerate(self.events):
                future=d.loc[d.event.ge(t)&d.event.lt(t+pd.Timedelta(days=7))]
                y[:,j]=np.isin(idx,future.actor)
            s['y']=y
        self.early_cache[key]=s
        return s

def validate_split(ds,t,c):
    asof=ds.fit_cutoff(t)
    out=[]
    for j,a in enumerate(m.ACT):
        weeks=c.get('training_weeks_by_activity',{}).get(a,m.MODELS[c['model']]['weeks'])
        tr,es,cal=ds.schedule(t,j,weeks)
        chunks=[ds.training_rows(d,j,asof,'base') for d in tr+[es,cal]]
        if any(len(np.unique(y))<2 for y in [np.concatenate([v[1] for v in chunks[:-2]]),chunks[-2][1],chunks[-1][1]]):
            raise ValueError(f'{a}: insufficient label classes; do not silently fill missing activities')
        if cal+pd.Timedelta(days=7)>asof:raise ValueError('Calibration labels unavailable at simulated fit')
        out.append({'activity':a,'train_label_starts':list(map(str,tr)),'early_stop_label_start':str(es),'calibration_label_start':str(cal),'fit_cutoff_utc':str(asof)})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.early.json'))
    ap.add_argument('--check',action='store_true');ap.add_argument('--backtest',action='store_true');a=ap.parse_args()
    c=json.loads(a.config.read_text(encoding='utf-8-sig'))
    for key in ['data_dir','db_dir','evaluator','test_input']:
        q=Path(c[key]);c[key]=str((a.config.resolve().parent/q).resolve()) if not q.is_absolute() else str(q)
    import catboost,xgboost
    ds=EarlyData(c);target=m.utc(c['backtest_start'] if a.backtest else c['prediction_start'])
    if a.backtest and target+pd.Timedelta(days=7)>ds.end:raise ValueError('Incomplete backtest target')
    if not a.backtest and ds.end>=target:raise ValueError('Forecast data must precede target')
    schedule=validate_split(ds,target,c)
    official=pd.read_csv(c['test_input'])
    if official.UserID.duplicated().any():raise ValueError('Duplicate official users')
    snap=ds.snapshot(target,labels=a.backtest)
    if not a.backtest and not official.UserID.isin(snap['ids']).all():raise ValueError('Official user absent at genuine feature cutoff')
    print('MODE', 'matched-lead historical backtest' if a.backtest else 'fixed-cutoff formal forecast',flush=True)
    print('LABEL TARGET',target,'FEATURE REFERENCE',target-ds.lead,'FIT CUTOFF',ds.fit_cutoff(target),'LEAD',ds.lead,flush=True)
    print('SCHEDULE',json.dumps(schedule),flush=True)
    if a.check:print('CHECK PASSED',len(official),'official users');return
    out=Path(c.get('output_dir',Path(__file__).parent/'early_runs'))/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    plan={'config':c,'mode':'matched_lead_backtest' if a.backtest else 'early_forecast','target_start_utc':str(target),'target_end_utc':str(target+pd.Timedelta(days=7)),
        'feature_reference_utc':str(target-ds.lead),'fit_cutoff_utc':str(ds.fit_cutoff(target)),'lead':str(ds.lead),'schedule':schedule,
        'input_sha256':ds.hashes,'official_sha256':{k:m.sha(Path(c[k])) for k in ['evaluator','test_input']},
        'code_sha256':{p.name:m.sha(p) for p in Path(__file__).parent.glob('*.py')},'catboost':catboost.__version__,'xgboost':xgboost.__version__}
    m.dump(out/'plan.json',plan)
    runner=m.Runner(c,out,ds);rows=[]
    for seed in c['seeds'] if a.backtest else [42]:
        r=runner.job(target,c['model'],seed,predict=not a.backtest)
        if a.backtest:
            metrics=json.loads((out/f'{m.stamp(target)}_{c["model"]}_seed{seed}'/'metrics.json').read_text())
            rows.append({'seed':seed,'historical_users':len(r['ids']),'six_observed_score':metrics['observed_six_NOT_COMPLETE']['score'],'five_activity_sensitivity':metrics['five_activity_sensitivity']['score']})
        else:
            runner.csv(out/'output_315552049.csv',r['ids'],r['predictions'])
            summary={'users':len(r['ids']),'columns':['UserID']+m.ACT,'positive_predictions':dict(zip(m.ACT,map(int,r['predictions'].sum(0)))),'future_score':None,'warning':'Early forecast with fixed available data, not a last-minute snapshot. Long-horizon generalization remains uncertain.'}
            m.dump(out/'forecast_summary.json',summary)
    if a.backtest:pd.DataFrame(rows).to_csv(out/'matched_lead_scores.csv',index=False)
    print('DONE',out,flush=True)

if __name__=='__main__':main()

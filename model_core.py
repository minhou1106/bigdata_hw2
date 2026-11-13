"""Local CPU experiment runner; no network, no synthetic event imputation."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
import pandas as pd
ACT=['open_pr','open_issue','issue_comment','pr_review','pr_review_comment','push']
SOURCES=[('PullRequests','author_id','created_at','first_seen_at'),('Issues','author_id','created_at','first_seen_at'),('IssueComments','author_id','created_at','created_at'),('PRReviews','reviewer_id','submitted_at','first_seen_at'),('PRReviewComments','author_id','created_at','created_at'),('Pushes','user_id','created_at','created_at')]
MODELS={
 'control4':dict(family='cb',weeks=4,features='base',depth=5,weight='ratio'),
 'cb8_plain':dict(family='cb',weeks=8,features='coverage',depth=4,weight='none'),
 'cb8_sqrt':dict(family='cb',weeks=8,features='coverage',depth=5,weight='sqrt'),
 'xgb8':dict(family='xgb',weeks=8,features='coverage',depth=3,weight='sqrt')}
MEMBERS=['cb8_plain','cb8_sqrt','xgb8']
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc(t):
 t=pd.Timestamp(t)
 return (t.tz_localize('Asia/Taipei') if t.tz is None else t).tz_convert('UTC').tz_localize(None)
def stamp(t):return pd.Timestamp(t).strftime('%Y%m%dT%H%M%S')
def metric(y,p,valid=None):
 y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
 if valid is not None:y=y[:,valid];p=p[:,valid]
 if y.shape[1]==0:return None
 tp=(y&p).sum(0);fp=(~y&p).sum(0);fn=(y&~p).sum(0);d=2*tp+fp+fn;f=np.divide(2*tp,d,out=np.ones(len(d)),where=d!=0)
 mi=float(2*tp.sum()/d.sum()) if d.sum() else 1.
 return {'macro_f1':float(f.mean()),'micro_f1':mi,'score':float((f.mean()+mi)/2),'per_activity_f1':f.tolist(),'tp':tp.tolist(),'fp':fp.tolist(),'fn':fn.tolist()}
def threshold(y,p):
 # Per-activity F1; calibration weeks can differ between activities.
 grid=np.r_[np.arange(.01,1,.01),1.01];scores=[metric(y[:,None],(p>=t)[:,None])['score'] for t in grid]
 best=max(scores);ties=np.flatnonzero(np.isclose(scores,best,atol=1e-12,rtol=0));return float(grid[ties[np.argmin(abs(grid[ties]-.5))]])
def quality(counts,ratio=.1):
 # Reference for day d uses only the preceding 28 UTC days.
 ref=counts.shift(1).rolling(28,min_periods=14).median()
 return pd.DataFrame({'count':counts,'reference':ref,'usable':ref.gt(0)&counts.ge(ref*ratio)&counts.gt(0)})
class Data:
 def __init__(self,c):
  self.c=c;self.root=Path(c['data_dir']);self.db=Path(c['db_dir']);self.end=utc(c['observed_through']);self.events=[]
  self.hashes={p.name:sha(p) for p in sorted(self.db.glob('*.parquet'))}
  if len(self.hashes)!=8:raise ValueError('Expected official root with eight Parquet tables')
  self.users=pd.read_parquet(self.db/'Users.parquet');self.users=self.users.sort_values('user_id')
  if self.users.github_user_id.duplicated().any():raise ValueError('Duplicate stable user ID')
  for table,actor,event,avail in SOURCES:
   cols=list(dict.fromkeys([actor,event,avail,'repo_id']+(['is_pull_request'] if table=='Issues' else [])))
   d=pd.read_parquet(self.db/(table+'.parquet'),columns=cols);ok=d[actor].notna()&d[event].notna()&d[avail].notna()&d[event].lt(self.end)&d[avail].lt(self.end)
   if table=='Issues':ok &= d.is_pull_request.eq(False)
   d=d.loc[ok].copy();self.events.append(pd.DataFrame({'actor':d[actor].astype('int64'),'event':d[event],'available':d[avail],'repo':d.repo_id}).reset_index(drop=True))
  latest=max(d.event.max() for d in self.events)
  if self.end-latest>pd.Timedelta(days=2):raise ValueError('Snapshot is too old for declared observed_through; do not advance date without new official data')
  self.qcache={};self.scache={};self.first=min(d.event.min() for d in self.events)
 def coverage(self,asof):
  asof=min(pd.Timestamp(asof),self.end);key=str(asof)
  if key not in self.qcache:
   days=pd.date_range(self.first.floor('d'),asof.floor('d')-pd.Timedelta(days=1),freq='D');out=[]
   for d in self.events:
    visible=d.loc[d.event.lt(asof)&d.available.lt(asof)];counts=visible.event.dt.floor('d').value_counts().reindex(days,fill_value=0).sort_index();out.append(quality(counts,self.c['coverage_ratio']))
   self.qcache[key]=out
  return self.qcache[key]
 def valid_week(self,start,j,asof):
  end=start+pd.Timedelta(days=7)
  if end>asof:return False
  days=pd.date_range(start.floor('d'),(end-pd.Timedelta(nanoseconds=1)).floor('d'))
  # Boundary UTC days must be complete too; conservative for partial-day overlap.
  return bool(self.coverage(asof)[j].usable.reindex(days,fill_value=False).all())
 def fit_cutoff(self,t):
  return min(t-getattr(self,"lead",pd.Timedelta(0))-pd.Timedelta(hours=4),self.end)
 def schedule(self,t,j,n):
  asof=self.fit_cutoff(t)
  good=[t-pd.Timedelta(weeks=i) for i in range(self.c['search_weeks'],0,-1) if self.valid_week(t-pd.Timedelta(weeks=i),j,asof)]
  if len(good)<n+2:raise ValueError(f'{ACT[j]} at {t}: only {len(good)} usable weeks; need {n+2}. Do not fill unknown labels with zero.')
  chosen=good[-(n+2):];return chosen[:-2],chosen[-2],chosen[-1]
 def snapshot(self,t,labels=True):
  key=(str(t),labels)
  if key in self.scache:return self.scache[key]
  end=min(t-pd.Timedelta(hours=4),self.end);u=self.users.loc[self.users.first_seen_at.lt(end)].copy();idx=u.user_id.to_numpy();n=len(u)
  age=(t-u.first_seen_at).dt.total_seconds().to_numpy()/86400;base=[np.ones(n),np.log1p(age)/7,np.exp(-age/7),np.exp(-age/28)];extra=[];y=np.zeros((n,6),dtype=np.int8);history=np.zeros(n);q=self.coverage(end)
  for j,d in enumerate(self.events):
   past=d.loc[d.event.lt(end)&d.available.lt(end)&d.event.ge(t-pd.Timedelta(days=90))].copy();counts=[]
   for days in [1,7,28,90]:
    z=past.loc[past.event.ge(t-pd.Timedelta(days=days))];cnt=z.groupby('actor').size().reindex(idx,fill_value=0).to_numpy();counts.append(cnt);base.append(np.log1p(cnt)/5)
    if days==90:history+=cnt
   # Recency may use all visible history, capped at 91 days.
   last=past.groupby('actor').event.max().reindex(idx);rec=(t-last).dt.total_seconds().div(86400).fillna(91).clip(0,91).to_numpy();base.append(rec/91)
   past['day']=past.event.dt.floor('d')
   for days in [7,28]:
    a=t-pd.Timedelta(days=days);z=past.loc[past.event.ge(a)];dates=pd.date_range(a.floor('d'),(end-pd.Timedelta(nanoseconds=1)).floor('d'));usable=q[j].usable.reindex(dates,fill_value=False);known=set(usable.index[usable]);good=z.loc[z.day.isin(known)]
    exposure=sum(max(0,(min(end,day+pd.Timedelta(days=1))-max(a,day)).total_seconds()/86400) for day in known);total=(end-a).total_seconds()/86400
    gc=good.groupby('actor').size().reindex(idx,fill_value=0).to_numpy();extra.extend([np.full(n,exposure/max(total,1e-6)),np.log1p(gc/max(exposure,1))/5,np.full(n,exposure==0,dtype=float),np.log1p(good.groupby('actor').day.nunique().reindex(idx,fill_value=0).to_numpy())/5,np.log1p(good.groupby('actor').repo.nunique().reindex(idx,fill_value=0).to_numpy())/5])
   extra.append(np.log1p(counts[1])-np.log1p(np.maximum(counts[2]-counts[1],0)/3))
   if labels:
    future=d.loc[d.event.ge(t)&d.event.lt(t+pd.Timedelta(days=7))];y[:,j]=np.isin(idx,future.actor)
  x=np.column_stack(base).astype('float32');extended=np.column_stack([x,*extra]).astype('float32');assert x.shape[1]==34 and np.isfinite(extended).all()
  result={'base':x,'coverage':extended,'y':y,'ids':u.github_user_id.to_numpy(dtype='int64'),'cohort':u.is_cohort.fillna(False).to_numpy(dtype=bool)&(history>=5)};self.scache[key]=result;return result
 def training_rows(self,d,j,asof,kind):
  # Labels must also have been available by the simulated model-fit cutoff.
  s=self.snapshot(d);keep=s['cohort'];visible=self.events[j];f=visible.loc[visible.event.ge(d)&visible.event.lt(d+pd.Timedelta(days=7))&visible.available.lt(asof)]
  userids=self.users.set_index('github_user_id').loc[s['ids'],'user_id'].to_numpy();y=np.isin(userids,f.actor).astype('int8')
  return s[kind][keep],y[keep]
class Runner:
 def __init__(self,c,out,ds):
  self.c,self.out,self.ds=c,out,ds;self.verifications=[]
  spec=importlib.util.spec_from_file_location('official_evaluate',Path(c['evaluator']));self.ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.ev)
 def csv(self,path,ids,p):
  f=pd.DataFrame(p.astype(int),columns=ACT);f.insert(0,'UserID',ids);f.to_csv(path,index=False);self.ev.read_predictions(path,ids);return f
 def job(self,t,name,seed,predict=False):
  job=self.out/f'{stamp(t)}_{name}_seed{seed}';done=job/'done.json'
  if done.exists():
   manifest=json.loads(done.read_text());assert all(sha(job/k)==v for k,v in manifest.items()),'Changed checkpoint';print('REUSE',job.name,flush=True)
   with np.load(job/'output.npz') as a:return {k:a[k] for k in a.files}
  job.mkdir(exist_ok=True);target=self.ds.snapshot(t,labels=not predict)
  if predict:
   official=pd.read_csv(Path(self.c['test_input']));ids=official.UserID.to_numpy(dtype='int64')
   if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate official users')
   ix=pd.Index(target['ids']).get_indexer(ids)
   if (ix<0).any():raise ValueError('Official users unavailable at history cutoff')
  else:ix=np.flatnonzero(target['cohort']);ids=target['ids'][ix]
  if name=='ensemble':
   results=[self.job(t,k,seed,predict) for k in MEMBERS]
   for r in results:np.testing.assert_array_equal(r['ids'],ids)
   p=np.mean([r['probabilities'] for r in results],axis=0);th=[]
   for j in range(6):
    vals=[];truth=None
    for member in MEMBERS:
     with np.load(self.out/f'{stamp(t)}_{member}_seed{seed}'/f'calibration_{j}.npz') as a:
      if truth is None:truth=a['y'];calids=a['ids']
      else:np.testing.assert_array_equal(truth,a['y']);np.testing.assert_array_equal(calids,a['ids'])
      vals.append(a['p'])
    th.append(threshold(truth,np.mean(vals,axis=0)))
   th=np.array(th)
  else:
   from catboost import CatBoostClassifier
   from xgboost import XGBClassifier
   s=MODELS[name];kind=s['features'];p=np.empty((len(ids),6));th=[];details=[];asof=self.ds.fit_cutoff(t)
   for j,a in enumerate(ACT):
    tr,es,cal=self.ds.schedule(t,j,self.c.get('training_weeks_by_activity',{}).get(a,s['weeks']));chunks=[self.ds.training_rows(d,j,asof,kind) for d in tr];X=np.concatenate([z[0] for z in chunks]);y=np.concatenate([z[1] for z in chunks]);V,z=self.ds.training_rows(es,j,asof,kind);C,w=self.ds.training_rows(cal,j,asof,kind);T=target[kind][ix]
    if any(len(np.unique(v))<2 for v in [y,z,w]):raise ValueError(f'{a}: insufficient class diversity in training/early-stop/calibration; refusing zero fallback')
    ratio=min(15,max(1,(len(y)-y.sum())/y.sum()));positive={'ratio':ratio,'sqrt':np.sqrt(ratio),'none':1}[s['weight']]
    if s['family']=='cb':
     model=CatBoostClassifier(iterations=self.c['iterations'],depth=s['depth'],learning_rate=.05,l2_leaf_reg=5,class_weights=[1,float(positive)],loss_function='Logloss',random_seed=seed,thread_count=self.c['threads'],allow_writing_files=False,verbose=False)
     model.fit(X,y,eval_set=(V,z),early_stopping_rounds=self.c['patience'],verbose=False);loaded=CatBoostClassifier();ext='cbm'
    else:
     model=XGBClassifier(n_estimators=self.c['iterations'],max_depth=3,learning_rate=.05,min_child_weight=20,reg_lambda=10,subsample=.8,colsample_bytree=.9,tree_method='hist',n_jobs=self.c['threads'],random_state=seed,eval_metric='logloss',early_stopping_rounds=self.c['patience'])
     model.fit(X,y,sample_weight=np.where(y,positive,1),eval_set=[(V,z)],sample_weight_eval_set=[np.where(z,positive,1)],verbose=False);loaded=XGBClassifier();ext='json'
    p[:,j]=model.predict_proba(T)[:,1];cp=model.predict_proba(C)[:,1];th.append(threshold(w,cp));mp=job/f'{a}.{ext}';model.save_model(str(mp));loaded.load_model(str(mp));err=float(np.max(abs(loaded.predict_proba(T)[:,1]-p[:,j])));assert err<1e-7
    cs=self.ds.snapshot(cal);np.savez_compressed(job/f'calibration_{j}.npz',y=w,p=cp,ids=cs['ids'][cs['cohort']]);details.append({'activity':a,'train':[str(d) for d in tr],'early_stop':str(es),'calibration':str(cal),'fit_asof':str(asof),'positive_weight':float(positive),'reload_error':err,'training_rows':len(y),'feature_count':X.shape[1]});print('FIT',job.name,a,flush=True)
   th=np.array(th);dump(job/'training.json',details)
  pred=(p>=th).astype(int);self.csv(job/('output_315552049.csv' if predict else 'BACKTEST_ONLY.csv'),ids,pred)
  result={'ids':ids,'probabilities':p,'thresholds':th,'predictions':pred}
  if not predict:
   pd.DataFrame({'UserID':ids}).to_csv(job/'backtest_input.csv',index=False)
   y=target['y'][ix];valid=np.array([self.ds.valid_week(t,j,self.ds.end) for j in range(6)]);result.update(y=y,valid=valid);truth=self.csv(job/'observed_truth_NOT_COMPLETE.csv',ids,y);checked=self.ev.scores(self.ev.read_predictions(job/'BACKTEST_ONLY.csv',ids),truth);ours=metric(y,pred);assert np.isclose(checked['score'],ours['score'])
   dump(job/'metrics.json',{'observed_six_NOT_COMPLETE':ours,'five_activity_sensitivity':metric(y[:,:5],pred[:,:5]),'heuristic_coverage_score':metric(y,pred,valid),'heuristic_usable_activities':[a for a,k in zip(ACT,valid) if k],'warning':'Heuristic coverage is not an official scoring mask or guarantee of complete truth.'})
  np.savez_compressed(job/'output.npz',**result);dump(done,{f.name:sha(f) for f in job.iterdir() if f.is_file() and f.name!='done.json'});return result
 def run(self):
  rows=[]
  for date in self.c['development']:
   t=utc(date)
   for name in list(MODELS)+['ensemble']:
    r=self.job(t,name,self.c['seed']);s=metric(r['y'],r['predictions'],r['valid'])
    if s is None:raise ValueError('No usable activity in development fold; cannot select model')
    rows.append({'date':date,'model':name,'score':s['score'],'five_activity_sensitivity':metric(r['y'][:,:5],r['predictions'][:,:5])['score']})
  df=pd.DataFrame(rows);df.to_csv(self.out/'development_scores.csv',index=False);ranking=df.groupby('model').score.agg(['mean',lambda a:np.std(a,ddof=0)]).rename(columns={'<lambda_0>':'std'});ranking['selection_score']=ranking['mean']-.25*ranking['std'];ranking=ranking.sort_values(['selection_score','model'],ascending=[False,True]);ranking.to_csv(self.out/'ranking.csv');winner=ranking.index[0];selection={'winner':winner,'method_settings':MODELS,'rule':'mean heuristic-coverage Score - 0.25 population std','warning':'Previously inspected historical development, not unseen final test.'};f=self.out/'selection_frozen.json'
  if f.exists():assert json.loads(f.read_text())==selection
  else:dump(f,selection)
  frozen=sha(f);print('SELECTED',winner,flush=True);later=[]
  for seed in self.c['later_seeds']:
   for name in sorted(set([winner,'control4'])):
    r=self.job(utc(self.c['later_check']),name,seed);s=metric(r['y'],r['predictions'],r['valid']);later.append({'model':name,'seed':seed,'coverage_score':None if s is None else s['score'],'five_activity_sensitivity':metric(r['y'][:,:5],r['predictions'][:,:5])['score'],'observed_six_NOT_COMPLETE':metric(r['y'],r['predictions'])['score']})
  assert sha(f)==frozen;pd.DataFrame(later).to_csv(self.out/'later_scores.csv',index=False)
  (self.out/'SUMMARY.txt').write_text('Completed historical experiments. NOT formal November prediction.\nSelected: '+winner+'\n\nDevelopment:\n'+ranking.to_string()+'\n\nLater check:\n'+pd.DataFrame(later).to_string(index=False)+'\nCoverage score is heuristic only; compare with control and prior runs before adoption.\n',encoding='utf-8');print('DONE',self.out,flush=True)

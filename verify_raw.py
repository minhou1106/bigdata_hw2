"""Check every expected gzip to EOF; file presence alone does not imply integrity."""
import argparse, datetime as dt, gzip, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',required=True);ap.add_argument('--end',required=True);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    day=dt.date.fromisoformat(a.start);end=dt.date.fromisoformat(a.end);records=[];errors=[]
    if day>end:raise ValueError('start must be <= end')
    while day<=end:
        for hour in range(24):
            name=f'{day:%Y-%m-%d}-{hour}.json.gz';p=a.raw/name
            record={'name':name,'status':'ok'}
            try:
                record['bytes']=p.stat().st_size
                if not record['bytes']:raise IOError('Empty file')
                with gzip.open(p,'rb') as f:
                    while f.read(1024*1024):pass
            except (OSError,EOFError) as e:record['status']=str(e);errors.append(name)
            records.append(record)
        print(day,'checked',flush=True);day+=dt.timedelta(days=1)
    a.out.write_text(json.dumps({'start_utc':a.start,'end_utc_inclusive':a.end,'files':records,'errors':errors},indent=2),encoding='utf-8')
    print('FILES',len(records),'ERRORS',len(errors))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()

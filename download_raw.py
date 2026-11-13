"""Sequential, restartable hourly downloads from the required source."""
import argparse, datetime as dt, time, urllib.request, urllib.error
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',required=True);ap.add_argument('--end',required=True);ap.add_argument('--raw',type=Path,required=True);a=ap.parse_args()
    day=dt.date.fromisoformat(a.start);end=dt.date.fromisoformat(a.end)
    if day>end:raise ValueError('start must be <= end')
    a.raw.mkdir(parents=True,exist_ok=True);failed=[]
    while day<=end:
        for hour in range(24):
            name=f'{day:%Y-%m-%d}-{hour}.json.gz';dest=a.raw/name
            if dest.is_file() and dest.stat().st_size>0:continue
            part=dest.with_name(name+'.part')
            for attempt in range(4):
                try:
                    print('DOWNLOAD',name,flush=True)
                    with urllib.request.urlopen('https://data.gharchive.org/'+name,timeout=90) as response,part.open('wb') as output:
                        size=0
                        while chunk:=response.read(1024*1024):output.write(chunk);size+=len(chunk)
                        length=response.headers.get('Content-Length')
                    if length and size!=int(length):raise IOError('Incomplete HTTP transfer')
                    with part.open('rb') as f:
                        if f.read(2)!=b'\x1f\x8b':raise IOError('Response is not gzip')
                    part.replace(dest);break
                except (OSError,urllib.error.URLError) as e:
                    print('WARNING',name,str(e),flush=True)
                    if attempt==3:failed.append(name)
                    else:time.sleep(2)
        day+=dt.timedelta(days=1)
    print('FAILED',failed)
    if failed:raise SystemExit(1)

if __name__=='__main__':main()

"""Use the previously selected CatBoost strategy on official revised tables."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import model_core as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--predict', action='store_true')
    args = ap.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8-sig'))
    for key in ['data_dir', 'db_dir', 'evaluator', 'test_input']:
        p = Path(c[key])
        c[key] = str((args.config.resolve().parent / p).resolve()) if not p.is_absolute() else str(p)
    import catboost
    import xgboost
    for key in ['evaluator', 'test_input']:
        if not Path(c[key]).is_file():
            raise FileNotFoundError(c[key])
    ds = m.Data(c)
    official = pd.read_csv(c['test_input'])
    if official.UserID.duplicated().any():
        raise ValueError('Duplicate official UserIDs')
    missing = set(official.UserID) - set(ds.users.github_user_id)
    if missing:
        raise ValueError(f'{len(missing)} official users absent from tables')
    t = m.utc(c['prediction_start'] if args.predict else c['backtest_start'])
    if args.predict:
        if not t - pd.Timedelta(hours=4) <= ds.end < t:
            raise ValueError('Formal forecast needs actual data ending within 4h before target. Current data is too early; do not change the cutoff without new data.')
    elif t + pd.Timedelta(days=7) > ds.end:
        raise ValueError('Backtest week is not complete')
    strategy = m.MODELS[c['model']]
    asof = min(t - pd.Timedelta(hours=4), ds.end)
    schedule = []
    for j, activity in enumerate(m.ACT):
        weeks = c.get('training_weeks_by_activity', {}).get(activity, strategy['weeks'])
        tr, es, cal = ds.schedule(t, j, weeks)
        parts = [ds.training_rows(d, j, asof, strategy['features']) for d in tr + [es, cal]]
        train_labels = np.concatenate([part[1] for part in parts[:-2]])
        if any(len(np.unique(y)) < 2 for y in [train_labels, parts[-2][1], parts[-1][1]]):
            raise ValueError(f'{activity}: need both classes in train, early-stop and calibration')
        item = {'activity': activity, 'train': list(map(str, tr)), 'early_stop': str(es), 'calibration': str(cal)}
        schedule.append(item)
        print(activity, 'TRAIN', item['train'], 'EARLY_STOP', es, 'CALIBRATION', cal, flush=True)
    print('PREFLIGHT OK:', len(official), 'official users; model', c['model'], flush=True)
    if args.check:
        return
    out = Path(c.get('output_dir', Path(__file__).parent / 'runs')) / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    m.dump(out/'plan.json', {'config': c, 'input_sha256': ds.hashes,
        'evaluator_sha256': m.sha(Path(c['evaluator'])), 'test_input_sha256': m.sha(Path(c['test_input'])),
        'code_sha256': {p.name: m.sha(p) for p in Path(__file__).parent.glob('*.py')},
        'strategy': strategy, 'schedule': schedule, 'catboost': catboost.__version__,
        'xgboost': xgboost.__version__, 'mode': 'forecast' if args.predict else 'backtest'})
    print('OUTPUT', out, flush=True)
    runner = m.Runner(c, out, ds)
    for j, coverage in enumerate(ds.coverage(ds.end)):
        coverage.to_csv(out / f'coverage_{m.ACT[j]}_UTC.csv')
    if args.predict:
        r = runner.job(t, c['model'], c['seeds'][0], True)
        runner.csv(out/'output_315552049.csv', r['ids'], r['predictions'])
        (out/'SUMMARY.txt').write_text('Formal prediction generated and official IDs checked. Future Score unavailable.', encoding='utf-8')
    else:
        rows = []
        for seed in c['seeds']:
            r = runner.job(t, c['model'], seed)
            metrics = json.loads((out/f'{m.stamp(t)}_{c["model"]}_seed{seed}'/'metrics.json').read_text())
            rows.append({'model': c['model'] + '_push2', 'seed': seed,
                'observed_six_NOT_COMPLETE': metrics['observed_six_NOT_COMPLETE']['score'],
                'five_activity_sensitivity': metrics['five_activity_sensitivity']['score'],
                'heuristic_coverage_score': (metrics['heuristic_coverage_score'] or {}).get('score')})
        scores = pd.DataFrame(rows)
        scores.to_csv(out/'scores.csv', index=False)
        summary = 'Historical backtest ONLY, not formal submission.\n' + scores.to_string(index=False) + '\nTwo seeds on one week are not two independent weeks. Missingness masks are heuristic.\n'
        (out/'SUMMARY.txt').write_text(summary, encoding='utf-8')
        print(summary, flush=True)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()

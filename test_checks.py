import unittest
import numpy as np
import pandas as pd
import model_core as m

class Checks(unittest.TestCase):
    def test_coverage_reference_has_no_future_counts(self):
        days=pd.date_range('2026-01-01',periods=50)
        counts=pd.Series(100,index=days)
        a=m.quality(counts)
        counts.iloc[40:]=10000
        pd.testing.assert_frame_equal(a.iloc[:40],m.quality(counts).iloc[:40])

    def test_missing_day_excludes_label_week(self):
        ds=m.Data.__new__(m.Data); ds.end=pd.Timestamp('2026-03-01')
        q=pd.DataFrame({'usable':True},index=pd.date_range('2026-01-01','2026-02-28'))
        q.loc['2026-02-05','usable']=False
        ds.coverage=lambda t:[q]*6
        self.assertFalse(ds.valid_week(pd.Timestamp('2026-02-01'),0,ds.end))
        self.assertTrue(ds.valid_week(pd.Timestamp('2026-02-08'),0,ds.end))

    def test_snapshot_excludes_late_available_rows(self):
        ds=m.Data.__new__(m.Data)
        ds.end=pd.Timestamp('2026-02-10');ds.scache={}
        ds.users=pd.DataFrame({'user_id':[0], 'github_user_id':[100], 'first_seen_at':[pd.Timestamp('2026-01-01')], 'is_cohort':[True]})
        d=pd.DataFrame({'actor':[0,0], 'event':pd.to_datetime(['2026-02-01','2026-02-01']), 'available':pd.to_datetime(['2026-02-01','2026-02-09']), 'repo':[0,0]})
        ds.events=[d.copy() for _ in range(6)]
        q=pd.DataFrame({'usable':True},index=pd.date_range('2026-01-01','2026-02-10'))
        ds.coverage=lambda t:[q]*6
        s=ds.snapshot(pd.Timestamp('2026-02-08'))
        self.assertAlmostEqual(float(s['base'][0,7]),np.log1p(1)/5,places=6)

    def test_utc_conversion(self):
        self.assertEqual(m.utc('2026-10-30'),pd.Timestamp('2026-10-29 16:00:00'))

    def test_empty_class_official_convention(self):
        z=np.zeros((2,1),dtype=int)
        self.assertEqual(m.metric(z,z)['score'],1.0)

    def test_threshold_uses_calibration(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.2,.8,.9])
        threshold=m.threshold(y,p)
        self.assertEqual(m.metric(y[:,None],(p>=threshold)[:,None])['score'],1)

if __name__=='__main__':unittest.main()

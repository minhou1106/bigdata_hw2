import unittest
import numpy as np
import pandas as pd
from early_forecast import EarlyData

class EarlyChecks(unittest.TestCase):
    def fixture(self):
        ds=EarlyData.__new__(EarlyData)
        ds.end=pd.Timestamp('2026-03-01');ds.lead=pd.Timedelta(days=14,hours=16)
        ds.scache={};ds.early_cache={}
        ds.users=pd.DataFrame({'user_id':[0],'github_user_id':[100],'first_seen_at':[pd.Timestamp('2026-01-01')],'is_cohort':[True]})
        dates=pd.to_datetime(['2026-02-01','2026-02-23'])
        d=pd.DataFrame({'actor':[0,0],'event':dates,'available':dates,'repo':[0,0]})
        ds.events=[d.copy() for _ in range(6)]
        q=pd.DataFrame({'usable':True},index=pd.date_range('2026-01-01','2026-02-28'))
        ds.coverage=lambda t:[q]*6
        return ds

    def test_features_use_earlier_reference(self):
        ds=self.fixture();target=pd.Timestamp('2026-02-22 16:00:00')
        s=ds.snapshot(target,False)
        # Reference is Feb 8, so the observed Feb 1 event is in the prior 7 days.
        self.assertAlmostEqual(float(s['base'][0,5]),np.log1p(1)/5,places=6)

    def test_label_week_stays_at_future_target(self):
        s=self.fixture().snapshot(pd.Timestamp('2026-02-22 16:00:00'),True)
        np.testing.assert_array_equal(s['y'],np.ones((1,6),dtype=np.int8))

    def test_fit_cutoff_precedes_unobserved_gap(self):
        ds=self.fixture()
        self.assertEqual(ds.fit_cutoff(pd.Timestamp('2026-02-22 16:00:00')),pd.Timestamp('2026-02-07 20:00:00'))

    def test_late_label_does_not_enter_features(self):
        ds=self.fixture();target=pd.Timestamp('2026-02-22 16:00:00')
        s=ds.snapshot(target,True)
        self.assertAlmostEqual(float(s['base'][0,7]),np.log1p(1)/5,places=6)

if __name__=='__main__':unittest.main()

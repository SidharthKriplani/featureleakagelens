import unittest
import pandas as pd

from featureleakagelens import LeakageAuditConfig, audit_dataframe

class FeatureLeakageLensTests(unittest.TestCase):
    def make_df(self):
        return pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(20)],
            "x": list(range(20)),
            "target_proxy_paid": [0]*10 + [1]*10,
            "segment": ["a"]*10 + ["b"]*10,
            "outcome_ts": pd.to_datetime(["2025-02-01"]*20),
            "proxy_ts": pd.to_datetime(["2025-02-05"]*20),
            "split": ["train"]*10 + ["test"]*10,
            "target": [0]*10 + [1]*10,
        })

    def test_missing_target_fails(self):
        report = audit_dataframe(pd.DataFrame({"x": [1, 2]}), LeakageAuditConfig(target_col="y"))
        self.assertEqual(report.status, "FAIL")

    def test_name_heuristic_warns(self):
        df = self.make_df()
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        self.assertTrue(any(f.check_name == "post_outcome_name_heuristic" for f in report.findings))

    def test_temporal_future_leakage_fails(self):
        df = self.make_df()
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            outcome_time_col="outcome_ts",
            feature_time_cols={"target_proxy_paid": "proxy_ts"},
        ))
        self.assertEqual(report.status, "FAIL")
        self.assertTrue(any(f.check_name == "temporal_availability" and f.status == "FAIL" for f in report.findings))

    def test_id_proxy_warns(self):
        df = self.make_df()
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        self.assertTrue(any(f.check_name == "id_proxy_scan" for f in report.findings))

    def test_outputs_render(self):
        report = audit_dataframe(self.make_df(), LeakageAuditConfig(target_col="target"))
        self.assertIn("FeatureLeakageLens", report.to_markdown())
        self.assertIn("<html", report.to_html())
        self.assertIn("findings", report.to_json())

    def test_cleanish_data_pass_or_insufficient_only(self):
        df = pd.DataFrame({
            "x1": [1,2,3,4,5,6,7,8,9,10,11,12],
            "x2": [1,1,2,2,3,3,1,2,3,1,2,3],
            "target": [0,1,0,1,0,1,1,0,1,0,1,0],
        })
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        self.assertIn(report.status, ["PASS", "INSUFFICIENT_INPUT"])

class TargetCorrelationTests(unittest.TestCase):
    def test_high_correlation_warns(self):
        import numpy as np
        n = 50
        target = np.array([0]*25 + [1]*25, dtype=float)
        # Feature that is essentially a noisy proxy of the target
        proxy = target + np.random.default_rng(42).normal(0, 0.05, n)
        df = pd.DataFrame({"proxy_feature": proxy, "target": target})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target", high_corr_threshold=0.85))
        self.assertTrue(any(f.check_name == "target_correlation_scan" for f in report.findings))

    def test_low_correlation_does_not_warn(self):
        import numpy as np
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "target": rng.integers(0, 2, 100).astype(float)})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target", high_corr_threshold=0.85))
        corr_findings = [f for f in report.findings if f.check_name == "target_correlation_scan"]
        self.assertEqual(len(corr_findings), 0)


class SplitDistributionTests(unittest.TestCase):
    def make_shifted_df(self):
        import numpy as np
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            "x": list(rng.normal(0, 1, 40)) + list(rng.normal(10, 1, 40)),
            "target": [0, 1] * 40,
            "split": ["train"] * 40 + ["test"] * 40,
        })

    def test_large_distribution_shift_warns(self):
        report = audit_dataframe(
            self.make_shifted_df(),
            LeakageAuditConfig(target_col="target", split_col="split", distribution_shift_threshold=0.35),
        )
        self.assertTrue(any(f.check_name == "split_distribution_scan" and f.status == "WARN" for f in report.findings))

    def test_no_split_col_gives_insufficient_input(self):
        df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        self.assertTrue(any(f.check_name == "split_distribution_scan" and f.status == "INSUFFICIENT_INPUT" for f in report.findings))


class EvidenceTests(unittest.TestCase):
    def test_temporal_finding_has_evidence(self):
        df = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(10)],
            "outcome_ts": pd.to_datetime(["2025-02-01"] * 10),
            "proxy_ts":   pd.to_datetime(["2025-02-05"] * 10),
            "feature":    [1.0] * 10,
            "target":     [0, 1] * 5,
        })
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            outcome_time_col="outcome_ts",
            feature_time_cols={"feature": "proxy_ts"},
        ))
        temporal = next(f for f in report.findings if f.check_name == "temporal_availability" and f.status == "FAIL")
        self.assertIn("future_rows", temporal.evidence)
        self.assertGreater(temporal.evidence["future_rows"], 0)

    def test_report_summary_has_expected_keys(self):
        df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        for key in ("rows", "columns", "finding_count", "warn_count", "fail_count"):
            self.assertIn(key, report.summary)

    def test_save_writes_three_files(self):
        import tempfile, os
        df = pd.DataFrame({"x": list(range(10)), "target": [0, 1] * 5})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        with tempfile.TemporaryDirectory() as tmp:
            report.save(tmp, stem="test_report")
            files = os.listdir(tmp)
            self.assertIn("test_report.json", files)
            self.assertIn("test_report.md", files)
            self.assertIn("test_report.html", files)


if __name__ == "__main__":
    unittest.main()

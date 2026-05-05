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


class HighCardinalityTests(unittest.TestCase):
    """Tests for the high-cardinality ID-proxy scan."""

    def test_high_cardinality_feature_warns(self):
        import numpy as np
        # Feature with near-unique values relative to row count → likely an ID proxy
        df = pd.DataFrame({
            "transaction_id": list(range(100)),       # 100% unique
            "target": np.random.default_rng(7).integers(0, 2, 100).astype(float),
        })
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            high_cardinality_ratio_threshold=0.90,
        ))
        self.assertTrue(any(f.check_name == "id_proxy_scan" for f in report.findings))

    def test_normal_cardinality_does_not_trigger_id_proxy(self):
        df = pd.DataFrame({
            "region": ["north", "south", "east", "west"] * 25,
            "target": [0, 1] * 50,
        })
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        id_proxy_findings = [f for f in report.findings if f.check_name == "id_proxy_scan" and f.status in ("WARN", "FAIL")]
        # region has low cardinality — should not trigger
        region_id_proxy = [f for f in id_proxy_findings if f.feature == "region"]
        self.assertEqual(len(region_id_proxy), 0)


class TargetRateTests(unittest.TestCase):
    """Tests that target rate gap between train and test is flagged."""

    def test_extreme_target_rate_gap_warns(self):
        # train: 90% positive, test: 10% positive — extreme label shift
        df = pd.DataFrame({
            "x": list(range(100)),
            "target": [1] * 45 + [0] * 5 + [0] * 45 + [1] * 5,
            "split":  ["train"] * 50 + ["test"] * 50,
        })
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            split_col="split",
            categorical_target_rate_gap_threshold=0.30,
        ))
        # Should flag split_distribution_scan or target correlation issues
        self.assertIn(report.status, ["WARN", "FAIL"])

    def test_balanced_target_rate_does_not_flag(self):
        df = pd.DataFrame({
            "x": list(range(200)),
            "target": [0, 1] * 100,
            "split": ["train"] * 100 + ["test"] * 100,
        })
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            split_col="split",
        ))
        # Balanced split → no major flags expected beyond INFO-level
        split_fails = [f for f in report.findings if f.check_name == "split_distribution_scan" and f.status == "FAIL"]
        self.assertEqual(len(split_fails), 0)


class WeightedRiskScoreTests(unittest.TestCase):
    """Tests for weighted_leakage_risk_score."""

    def test_score_increases_with_severity(self):
        from featureleakagelens import LeakageFinding, weighted_leakage_risk_score
        findings_high = [LeakageFinding("x", "FAIL", "high", "f", "d", "r")]
        findings_low  = [LeakageFinding("x", "WARN", "low",  "f", "d", "r")]
        self.assertGreater(
            weighted_leakage_risk_score(findings_high),
            weighted_leakage_risk_score(findings_low),
        )

    def test_zero_score_for_pass_findings(self):
        from featureleakagelens import LeakageFinding, weighted_leakage_risk_score
        findings = [LeakageFinding("x", "PASS", "none", None, "d", "r")]
        self.assertEqual(weighted_leakage_risk_score(findings), 0.0)

    def test_score_in_summary_dict(self):
        df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        self.assertIn("weighted_risk_score", report.summary)
        self.assertIsInstance(report.summary["weighted_risk_score"], float)


class TrainingFutureDateScanTests(unittest.TestCase):
    """Tests for the auto-detected temporal boundary scan."""

    def _make_boundary_df(self, future: bool):
        import numpy as np
        # outcome_ts is the "training cutoff" — 2025-01-15
        # feature_ts: if future=True, training rows have feature dates AFTER the cutoff
        n_train, n_test = 20, 20
        cutoff = pd.Timestamp("2025-01-15")
        feature_dates_train = (
            pd.date_range("2025-01-20", periods=n_train, freq="D")  # after cutoff
            if future
            else pd.date_range("2025-01-01", periods=n_train, freq="D")  # before
        )
        return pd.DataFrame({
            "feature_date": list(feature_dates_train) + pd.date_range("2025-02-01", periods=n_test, freq="D").tolist(),
            "x": list(range(n_train + n_test)),
            "outcome_ts": [cutoff] * n_train + [pd.Timestamp("2025-02-28")] * n_test,
            "split": ["train"] * n_train + ["test"] * n_test,
            "target": [0, 1] * 20,
        })

    def test_future_feature_dates_in_training_fails(self):
        df = self._make_boundary_df(future=True)
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            split_col="split",
            outcome_time_col="outcome_ts",
        ))
        scan_findings = [f for f in report.findings if f.check_name == "training_future_date_scan"]
        fail_findings = [f for f in scan_findings if f.status == "FAIL"]
        self.assertGreater(len(fail_findings), 0)
        self.assertIn("future_rows", fail_findings[0].evidence)

    def test_clean_feature_dates_do_not_fail(self):
        df = self._make_boundary_df(future=False)
        report = audit_dataframe(df, LeakageAuditConfig(
            target_col="target",
            split_col="split",
            outcome_time_col="outcome_ts",
        ))
        fail_findings = [
            f for f in report.findings
            if f.check_name == "training_future_date_scan" and f.status == "FAIL"
        ]
        self.assertEqual(len(fail_findings), 0)

    def test_missing_config_gives_insufficient_input(self):
        df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
        report = audit_dataframe(df, LeakageAuditConfig(target_col="target"))
        self.assertTrue(any(
            f.check_name == "training_future_date_scan" and f.status == "INSUFFICIENT_INPUT"
            for f in report.findings
        ))


if __name__ == "__main__":
    unittest.main()

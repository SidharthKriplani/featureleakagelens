from __future__ import annotations

from dataclasses import dataclass, field, asdict
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional
import json
import math

import numpy as np
import pandas as pd

Status = Literal["PASS", "WARN", "FAIL", "INSUFFICIENT_INPUT"]
Severity = Literal["none", "low", "medium", "high"]

POST_OUTCOME_TERMS = (
    "default", "defaulted", "paid", "payment_received", "settled", "recovered",
    "approved", "approval", "denied", "closed", "chargeback", "refund", "outcome",
    "target", "label", "bad_flag", "delinquent", "fraud_confirmed"
)

@dataclass
class LeakageAuditConfig:
    target_col: str
    split_col: Optional[str] = None
    train_value: str = "train"
    test_value: str = "test"
    outcome_time_col: Optional[str] = None
    feature_time_cols: Dict[str, str] = field(default_factory=dict)
    ignore_cols: List[str] = field(default_factory=list)
    high_corr_threshold: float = 0.85
    categorical_target_rate_gap_threshold: float = 0.65
    max_unique_ratio_for_categorical_scan: float = 0.25
    high_cardinality_ratio_threshold: float = 0.90
    distribution_shift_threshold: float = 0.35

@dataclass
class LeakageFinding:
    check_name: str
    status: Status
    severity: Severity
    feature: Optional[str]
    detail: str
    recommendation: str
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

@dataclass
class LeakageReport:
    status: Status
    findings: List[LeakageFinding]
    summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# FeatureLeakageLens Audit Report",
            "",
            f"**Overall status:** `{self.status}`",
            "",
            "## Summary",
            "",
        ]
        for k, v in self.summary.items():
            lines.append(f"- **{k}:** {v}")
        lines += ["", "## Findings", ""]
        if not self.findings:
            lines.append("No findings generated.")
        for i, f in enumerate(self.findings, 1):
            feature = f.feature or "dataset"
            lines += [
                f"### {i}. {f.check_name} — `{f.status}` / {f.severity}",
                f"- **Feature:** `{feature}`",
                f"- **Detail:** {f.detail}",
                f"- **Recommendation:** {f.recommendation}",
            ]
            if f.evidence:
                ev = ", ".join(f"{k}={v}" for k, v in f.evidence.items())
                lines.append(f"- **Evidence:** {ev}")
            lines.append("")
        lines += [
            "## Interpretation note",
            "",
            "FeatureLeakageLens is an auditor. WARN and FAIL findings should be reviewed against feature availability rules before removing any column.",
        ]
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = []
        for f in self.findings:
            rows.append(
                "<tr>"
                f"<td>{escape(f.check_name)}</td>"
                f"<td>{escape(f.status)}</td>"
                f"<td>{escape(f.severity)}</td>"
                f"<td>{escape(f.feature or 'dataset')}</td>"
                f"<td>{escape(f.detail)}</td>"
                f"<td>{escape(f.recommendation)}</td>"
                "</tr>"
            )
        summary_items = "".join(f"<li><b>{escape(str(k))}:</b> {escape(str(v))}</li>" for k, v in self.summary.items())
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>FeatureLeakageLens Report</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:40px;line-height:1.45;color:#1f2937}}
.badge{{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{border:1px solid #e5e7eb;padding:10px;text-align:left;vertical-align:top}}th{{background:#f9fafb}}
.note{{background:#fff7ed;border:1px solid #fed7aa;padding:14px;border-radius:12px;margin-top:24px}}
</style></head><body>
<h1>FeatureLeakageLens Audit Report</h1>
<p class='badge'>Overall status: {escape(self.status)}</p>
<h2>Summary</h2><ul>{summary_items}</ul>
<h2>Findings</h2>
<table><thead><tr><th>Check</th><th>Status</th><th>Severity</th><th>Feature</th><th>Detail</th><th>Recommendation</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class='note'><b>Truth boundary:</b> This report flags suspicious leakage patterns. It does not prove leakage without business-time availability review.</div>
</body></html>"""

    def save(self, output_dir: str | Path, stem: str = "featureleakagelens_report") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{stem}.json").write_text(self.to_json(), encoding="utf-8")
        (out / f"{stem}.md").write_text(self.to_markdown(), encoding="utf-8")
        (out / f"{stem}.html").write_text(self.to_html(), encoding="utf-8")


def _overall_status(findings: List[LeakageFinding]) -> Status:
    if any(f.status == "FAIL" for f in findings):
        return "FAIL"
    if any(f.status == "WARN" for f in findings):
        return "WARN"
    if findings and all(f.status == "INSUFFICIENT_INPUT" for f in findings):
        return "INSUFFICIENT_INPUT"
    return "PASS"


def _safe_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    frame = pd.concat([x, y], axis=1).dropna()
    if frame.shape[0] < 3:
        return None
    a = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    b = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return None
    if a[valid].nunique() <= 1 or b[valid].nunique() <= 1:
        return None
    corr = float(np.corrcoef(a[valid], b[valid])[0, 1])
    if math.isnan(corr):
        return None
    return corr


def _target_rate_gap(feature: pd.Series, target: pd.Series) -> Optional[float]:
    frame = pd.concat([feature, target], axis=1).dropna()
    if frame.shape[0] < 10:
        return None
    y = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    valid = y.notna()
    frame = frame.loc[valid]
    y = y.loc[valid]
    if y.nunique() != 2:
        return None
    rates = frame.groupby(frame.iloc[:, 0])[frame.columns[1]].mean()
    counts = frame.groupby(frame.iloc[:, 0]).size()
    rates = rates[counts >= 5]
    if len(rates) < 2:
        return None
    return float(rates.max() - rates.min())


def _numeric_shift(train: pd.Series, test: pd.Series) -> Optional[float]:
    tr = pd.to_numeric(train, errors="coerce").dropna()
    te = pd.to_numeric(test, errors="coerce").dropna()
    if len(tr) < 10 or len(te) < 10:
        return None
    denom = float(tr.std(ddof=0) or 0) + 1e-9
    return float(abs(te.mean() - tr.mean()) / denom)


def _categorical_tvd(train: pd.Series, test: pd.Series) -> Optional[float]:
    tr = train.dropna().astype(str)
    te = test.dropna().astype(str)
    if len(tr) < 10 or len(te) < 10:
        return None
    cats = sorted(set(tr.unique()).union(set(te.unique())))
    p = tr.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    q = te.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return float(0.5 * np.abs(p - q).sum())


def weighted_leakage_risk_score(findings: List[LeakageFinding]) -> float:
    """Compute a weighted severity risk score across all findings.

    FAIL/high=4.5, FAIL/medium=3.0, FAIL/low=1.5,
    WARN/high=1.5, WARN/medium=1.0, WARN/low=0.3
    Returns a non-negative float; higher = riskier.
    """
    weights = {
        ("FAIL", "high"): 4.5,
        ("FAIL", "medium"): 3.0,
        ("FAIL", "low"): 1.5,
        ("WARN", "high"): 1.5,
        ("WARN", "medium"): 1.0,
        ("WARN", "low"): 0.3,
    }
    return round(sum(weights.get((f.status, f.severity), 0.0) for f in findings), 2)


def _auto_detect_datetime_cols(df: pd.DataFrame, ignore: set) -> List[str]:
    """Return column names that appear to be datetime-valued (dtype or parseable strings)."""
    detected = []
    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            detected.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(30)
            if len(sample) < 3:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                if parsed.notna().sum() >= len(sample) * 0.8:
                    detected.append(col)
            except Exception:
                pass
    return detected


def audit_dataframe(df: pd.DataFrame, config: LeakageAuditConfig) -> LeakageReport:
    findings: List[LeakageFinding] = []
    if config.target_col not in df.columns:
        findings.append(LeakageFinding(
            "input_schema", "FAIL", "high", config.target_col,
            f"Target column '{config.target_col}' is missing.",
            "Provide a valid target column before running the leakage audit.",
        ))
        return LeakageReport("FAIL", findings, {"rows": len(df), "columns": len(df.columns)})

    ignored = set(config.ignore_cols + [config.target_col])
    if config.split_col:
        ignored.add(config.split_col)
    if config.outcome_time_col:
        ignored.add(config.outcome_time_col)
    ignored.update(config.feature_time_cols.values())

    target = df[config.target_col]
    candidate_cols = [c for c in df.columns if c not in ignored]

    # Naming heuristic
    for col in candidate_cols:
        lower = col.lower()
        matched = [term for term in POST_OUTCOME_TERMS if term in lower]
        if matched:
            findings.append(LeakageFinding(
                "post_outcome_name_heuristic", "WARN", "medium", col,
                f"Column name contains post-outcome/proxy term(s): {matched}.",
                "Verify this feature is available at prediction time and not computed after the label.",
                {"matched_terms": matched},
            ))

    # Numeric correlation
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target):
            corr = _safe_corr(df[col], target)
            if corr is not None and abs(corr) >= config.high_corr_threshold:
                findings.append(LeakageFinding(
                    "target_correlation_scan", "WARN", "high", col,
                    f"Feature has very high absolute correlation with target ({corr:.3f}).",
                    "Review whether this feature is a target proxy or post-outcome measurement.",
                    {"correlation": round(corr, 4), "threshold": config.high_corr_threshold},
                ))

    # Categorical target-rate gap
    for col in candidate_cols:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        should_scan = (not pd.api.types.is_numeric_dtype(df[col])) or unique_ratio <= config.max_unique_ratio_for_categorical_scan
        if should_scan:
            gap = _target_rate_gap(df[col], target)
            if gap is not None and gap >= config.categorical_target_rate_gap_threshold:
                findings.append(LeakageFinding(
                    "categorical_proxy_scan", "WARN", "medium", col,
                    f"Feature values have a large target-rate gap ({gap:.3f}).",
                    "Check if this categorical feature encodes post-outcome state or business-process leakage.",
                    {"target_rate_gap": round(gap, 4), "threshold": config.categorical_target_rate_gap_threshold},
                ))

    # Temporal availability
    if config.outcome_time_col and config.feature_time_cols:
        if config.outcome_time_col not in df.columns:
            findings.append(LeakageFinding(
                "temporal_availability", "INSUFFICIENT_INPUT", "medium", config.outcome_time_col,
                "Outcome timestamp column was configured but is missing.",
                "Provide a valid outcome timestamp column.",
            ))
        else:
            outcome_ts = pd.to_datetime(df[config.outcome_time_col], errors="coerce")
            for feature, ts_col in config.feature_time_cols.items():
                if ts_col not in df.columns:
                    findings.append(LeakageFinding(
                        "temporal_availability", "INSUFFICIENT_INPUT", "medium", feature,
                        f"Feature timestamp column '{ts_col}' is missing.",
                        "Provide the feature-generation timestamp or remove this feature from temporal audit.",
                    ))
                    continue
                feature_ts = pd.to_datetime(df[ts_col], errors="coerce")
                valid = outcome_ts.notna() & feature_ts.notna()
                future_count = int((feature_ts[valid] > outcome_ts[valid]).sum())
                if valid.sum() > 0 and future_count > 0:
                    findings.append(LeakageFinding(
                        "temporal_availability", "FAIL", "high", feature,
                        f"Feature timestamp occurs after outcome timestamp for {future_count} rows.",
                        "Do not use this feature unless the prediction-time workflow can prove availability before outcome.",
                        {"future_rows": future_count, "checked_rows": int(valid.sum())},
                    ))
    else:
        findings.append(LeakageFinding(
            "temporal_availability", "INSUFFICIENT_INPUT", "low", None,
            "No outcome timestamp and feature timestamp mapping provided.",
            "Provide outcome_time_col and feature_time_cols to audit future leakage.",
        ))

    # ID/proxy leakage
    for col in candidate_cols:
        lower = col.lower()
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if (lower.endswith("id") or lower.endswith("_id") or "identifier" in lower) and unique_ratio >= config.high_cardinality_ratio_threshold:
            findings.append(LeakageFinding(
                "id_proxy_scan", "WARN", "medium", col,
                f"ID-like high-cardinality feature has unique ratio {unique_ratio:.3f}.",
                "Avoid training directly on identifiers unless there is a documented, stable business reason.",
                {"unique_ratio": round(unique_ratio, 4)},
            ))

    # Training future date scan (auto-detected datetime columns vs. training boundary)
    if (config.split_col and config.outcome_time_col
            and config.split_col in df.columns
            and config.outcome_time_col in df.columns):
        train_subset = df[df[config.split_col].astype(str) == str(config.train_value)]
        outcome_ts = pd.to_datetime(train_subset[config.outcome_time_col], errors="coerce")
        training_cutoff = outcome_ts.max()
        if pd.notna(training_cutoff):
            date_cols = _auto_detect_datetime_cols(
                train_subset, ignored | {config.outcome_time_col}
            )
            for col in date_cols:
                feature_ts = pd.to_datetime(train_subset[col], errors="coerce")
                valid = feature_ts.notna()
                future_count = int((feature_ts[valid] > training_cutoff).sum())
                if future_count > 0:
                    findings.append(LeakageFinding(
                        "training_future_date_scan", "FAIL", "high", col,
                        f"Feature '{col}' has {future_count} training row(s) with dates after "
                        f"the inferred training cutoff ({training_cutoff.date()}).",
                        "Do not use features derived from data beyond the training boundary. "
                        "Verify this date column is not computed from post-outcome events.",
                        {
                            "future_rows": future_count,
                            "training_cutoff": str(training_cutoff.date()),
                            "checked_rows": int(valid.sum()),
                        },
                    ))
    else:
        findings.append(LeakageFinding(
            "training_future_date_scan", "INSUFFICIENT_INPUT", "low", None,
            "Temporal boundary scan requires both split_col and outcome_time_col.",
            "Provide split_col and outcome_time_col to enable auto-detected date-boundary scanning.",
        ))

    # Split distribution warnings
    if config.split_col:
        if config.split_col not in df.columns:
            findings.append(LeakageFinding(
                "split_distribution_scan", "INSUFFICIENT_INPUT", "low", config.split_col,
                "Split column was configured but missing.",
                "Provide a valid split column or omit split_col.",
            ))
        else:
            train_df = df[df[config.split_col].astype(str) == str(config.train_value)]
            test_df = df[df[config.split_col].astype(str) == str(config.test_value)]
            if len(train_df) < 10 or len(test_df) < 10:
                findings.append(LeakageFinding(
                    "split_distribution_scan", "INSUFFICIENT_INPUT", "low", config.split_col,
                    "Train/test split has too few rows for distribution checks.",
                    "Use larger split samples for stable warnings.",
                ))
            else:
                for col in candidate_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        score = _numeric_shift(train_df[col], test_df[col])
                        if score is not None and score >= config.distribution_shift_threshold * 3:
                            findings.append(LeakageFinding(
                                "split_distribution_scan", "WARN", "low", col,
                                f"Train/test numeric distribution differs materially (standardized mean shift {score:.3f}).",
                                "Review split logic and whether validation reflects deployment conditions.",
                                {"standardized_mean_shift": round(score, 4)},
                            ))
                    else:
                        score = _categorical_tvd(train_df[col], test_df[col])
                        if score is not None and score >= config.distribution_shift_threshold:
                            findings.append(LeakageFinding(
                                "split_distribution_scan", "WARN", "low", col,
                                f"Train/test categorical distribution differs materially (TVD {score:.3f}).",
                                "Review split logic and segment coverage before trusting validation metrics.",
                                {"total_variation_distance": round(score, 4)},
                            ))
    else:
        findings.append(LeakageFinding(
            "split_distribution_scan", "INSUFFICIENT_INPUT", "low", None,
            "No split column provided.",
            "Provide split_col to audit train/test distribution warnings.",
        ))

    status = _overall_status(findings)
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "candidate_features_checked": int(len(candidate_cols)),
        "finding_count": int(len(findings)),
        "warn_count": int(sum(f.status == "WARN" for f in findings)),
        "fail_count": int(sum(f.status == "FAIL" for f in findings)),
        "insufficient_input_count": int(sum(f.status == "INSUFFICIENT_INPUT" for f in findings)),
        "weighted_risk_score": weighted_leakage_risk_score(findings),
    }
    if not findings:
        findings.append(LeakageFinding(
            "audit_complete", "PASS", "none", None,
            "No suspicious leakage patterns were detected by v0 checks.",
            "Continue with normal feature review; this is not a guarantee that no leakage exists.",
        ))
        status = "PASS"
        summary["finding_count"] = 1
    return LeakageReport(status, findings, summary)

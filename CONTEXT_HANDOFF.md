# FeatureLeakageLens v0.1.0 — Context Handoff

FeatureLeakageLens was built as a focused Python package for Sidharth's ML decision-quality / risk / MLOps-lite portfolio lane.

## Why it was built

Sidharth already has large portfolio systems. FeatureLeakageLens is intentionally a smaller reusable library that proves practical ML review maturity: before training a model, audit whether the training data contains features that may leak target or future information.

The library supports roles across:

- AI/ML Engineer
- Risk / Decisioning ML
- ML Platform / MLOps-lite
- Senior Data Scientist roles where validation quality matters

## Positioning

FeatureLeakageLens is a **pre-training leakage auditor** for tabular ML datasets.

It is not a drift monitoring platform, not a full feature store validator, and not a model-risk governance suite. It flags suspicious features for human review.

Correct claim:

> FeatureLeakageLens audits tabular ML datasets for possible leakage patterns such as target proxies, post-outcome feature names, future-generated timestamps, high-cardinality identifiers, and suspicious train/test distribution differences.

## Current v0 features

- Target correlation scan
- Categorical target-rate proxy scan
- Post-outcome naming heuristic
- Temporal availability audit with outcome and feature timestamps
- ID/proxy leakage heuristic
- Train/test distribution warnings
- JSON / Markdown / HTML report export
- Demo dataset with planted leakage
- Unit tests

## Truth boundary

The tool does not prove leakage. It returns warnings and review recommendations. Leakage is often semantic and depends on business-time availability, so human review is mandatory.

## Improvements Claude can make

- Strengthen API schemas and type hints
- Add a `FeatureMetadata` object for availability-time semantics
- Add explicit `PASS/WARN/FAIL/INSUFFICIENT_INPUT` aggregation tests
- Add richer synthetic scenarios: target leakage, future leakage, ID leakage, train/test contamination
- Add methodology examples showing what each warning means
- Add GitHub-ready README, badges, CI, release, and static HTML demo report
- Consider adding optional sklearn mutual information when dependencies are available
- Keep scope narrow; do not turn this into Evidently or Great Expectations

## Resume-safe bullet

Built FeatureLeakageLens, a Python library that audits tabular ML datasets before training for potential target leakage, future leakage, post-outcome feature names, ID/proxy leakage, and split distribution issues, generating structured JSON/Markdown/HTML review reports.

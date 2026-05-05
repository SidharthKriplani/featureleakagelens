# Methodology

FeatureLeakageLens v0 focuses on practical leakage review for tabular datasets.

## Status vocabulary

- `PASS`: no obvious issue found for that check.
- `WARN`: suspicious pattern found; human review required.
- `FAIL`: strong structural issue found, such as feature timestamps occurring after outcome timestamps.
- `INSUFFICIENT_INPUT`: the check could not run because required inputs were missing.

## Checks

### 1. Target correlation scan

Numeric features are correlated with the target. Extremely high absolute correlation is suspicious because it may indicate that a feature encodes the outcome directly or indirectly. This does not prove leakage.

### 2. Categorical proxy scan

Categorical and boolean columns are checked for strong target-rate separation across values. This is a lightweight mutual-information-style proxy suitable for v0 without external dependencies.

### 3. Naming heuristic

Column names containing post-outcome terms such as `defaulted`, `paid`, `settled`, `recovered`, `approved`, `chargeback`, or `refund` are flagged. Naming is noisy but useful as a first-pass review layer.

### 4. Temporal availability audit

If the user provides an outcome timestamp and feature timestamp columns, FeatureLeakageLens checks whether feature values were generated after the outcome. Such features are usually invalid at prediction time.

### 5. ID/proxy leakage heuristic

High-cardinality ID-like fields are flagged when they have a suspicious relationship with the target. This catches cases where identifiers are proxies for outcome, campaign, or post-processing logic.

### 6. Split distribution warnings

When a split column is provided, the tool compares train/test distributions. This is not leakage by itself, but large differences can indicate split logic issues or validation design problems.

## Limitations

- The tool cannot know business availability rules unless the user provides timestamps or metadata.
- High correlation is not always leakage; legitimate predictive features can be highly correlated.
- Some leakage is semantic and cannot be detected from a DataFrame alone.
- v0 does not train a model and does not inspect pipeline code.

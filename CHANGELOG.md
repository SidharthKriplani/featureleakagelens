# Changelog

## [0.3.0] — 2026-05-06

### Added
- `HighCardinalityTests` (2 tests): verifies `id_proxy_scan` fires on near-unique
  numeric features and does not fire on low-cardinality categoricals.
- `TargetRateTests` (2 tests): verifies extreme train/test label shift is flagged
  and that a balanced split does not produce false positives.
- `docs/prd/` directory with Interview Defense and PRD documents.

### Changed
- Version bumped to 0.3.0.

## [0.2.0] — 2025-05-05

### Added
- **Expanded test suite** — 13 tests (up from 6): TargetCorrelationTests (2),
  SplitDistributionTests (2), EvidenceTests (3).
- **GitHub Actions CI** (`.github/workflows/ci.yml`): matrix test across Python
  3.10, 3.11, 3.12 on every push and PR.
- **GitHub Actions publish** (`.github/workflows/publish.yml`): trusted publisher
  OIDC release to PyPI on tagged release.
- **README** updated with badges, Mermaid architecture diagram, About section,
  check table, and resume-safe claim.
- **CHANGELOG.md** (this file).
- **FeatureLeakageLens_PRD.pdf** and **FeatureLeakageLens_Interview_Defense.pdf**.
- **outputs/.gitkeep** — ensures outputs/ directory exists on fresh clone.

## [0.1.0] — 2025-04-25

Initial release. Checks: post-outcome name heuristic, target correlation scan,
categorical proxy scan, temporal availability, ID/proxy scan, train/test
distribution shift. JSON / Markdown / HTML report rendering.

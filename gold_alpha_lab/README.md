# Gold Alpha Lab

First-phase structural-alpha research for gold. It deliberately researches **whether an edge exists** before attempting leverage or live trading.

## Scope

Implemented: timezone-aware session features, Asian-range breakout and fake-breakout labels, causal trend state, cost scenarios, alpha statistics, robustness classification, and session scans.

Not implemented: live orders, brokerage connections, API keys, macro/event execution, portfolio leverage, or claims of future performance.

## Quick start

```powershell
cd gold_alpha_lab
python scripts/download_data.py
python scripts/scan_session_alpha.py
pytest -q
```

The downloader uses only an existing local, real XAUUSD dataset. It reports its actual coverage and refuses to manufacture data. The default source is the repository's Dukascopy hourly bid file (2003-05-05 to 2024-09-30); it is resampled to 15-minute only when native finer bars are available.

## Outputs

- `output/reports/data_quality.json` — actual coverage and gaps
- `output/reports/session_alpha_report.md` — session and breakout conditional results
- `output/alpha_scan/*.csv` — observations and all cost scenarios

## Research guardrails

- All calculations use only information available at the signal timestamp.
- Session membership is derived from IANA time zones, not hard-coded UTC hours, so DST is handled correctly.
- P-values are descriptive; a result is never promoted solely because it is statistically significant.
- `DECAYED` means the recent period has lost its historical behavior.

"""
Run from project root: python src/classify_sample.py
Creates data/results/sample_classification.csv with tier labels for every observation.
"""
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
results_path = PROJECT_ROOT / "data" / "results" / "dcf_results.csv"
output_path = PROJECT_ROOT / "data" / "results" / "sample_classification.csv"

df = pd.read_csv(results_path, low_memory=False)
print(f"Loaded {len(df)} observations from dcf_results.csv")

# Classification flags
shell = df['industry'].str.contains('Shell', case=False, na=False)
neg_ev = df['ev'] < 0
zero_ev = df['ev'] == 0
wacc_fb = (df['wacc_used'] >= 0.0899) & (df['wacc_used'] <= 0.0901)

# Assign tier
tiers = []
reasons = []
for i, row in df.iterrows():
    if shell.iloc[i]:
        tiers.append("EXCLUDED")
        reasons.append("Shell/SPAC company")
    elif zero_ev.iloc[i]:
        tiers.append("EXCLUDED")
        reasons.append("Zero EV (no operating data)")
    elif neg_ev.iloc[i]:
        tiers.append("EXCLUDED")
        reasons.append("Negative EV")
    elif row['dq_errors'] == 0:
        tiers.append("TIER 1")
        reasons.append("Clean — zero data quality errors")
    elif row['dq_errors'] > 0 and not wacc_fb.iloc[i]:
        tiers.append("TIER 2")
        reasons.append("DQ issues but firm-specific WACC computed")
    elif row['dq_errors'] > 0 and wacc_fb.iloc[i]:
        tiers.append("TIER 3")
        reasons.append("DQ issues + WACC fallback (9%)")
    else:
        tiers.append("UNCLASSIFIED")
        reasons.append("Unknown")

df['sample_tier'] = tiers
df['tier_reason'] = reasons

# Summary
print("\nSample classification:")
print(df['sample_tier'].value_counts())

# Save with key columns
out_cols = [
    'ticker', 'asof_year', 'sector', 'industry', 'gic_sector', 'gic_industry',
    'ev', 'equity_value', 'wacc_used', 'ebit_margin', 'revenue_cagr_start',
    'terminal_growth', 'beta_used', 'beta_source', 'mcap_source',
    'dq_errors', 'dq_warnings', 'sample_tier', 'tier_reason'
]
# Only include columns that exist
out_cols = [c for c in out_cols if c in df.columns]
out = df[out_cols].sort_values(['sample_tier', 'ticker'])

out.to_csv(output_path, index=False)
print(f"\nSaved -> {output_path}")

# Also print tier breakdowns
for tier in ['TIER 1', 'TIER 2', 'TIER 3', 'EXCLUDED']:
    subset = df[df['sample_tier'] == tier]
    print(f"\n{tier}: {len(subset)} firms")
    if tier != 'EXCLUDED' and len(subset) > 0:
        print(f"  EV range: ${subset['ev'].min()/1e6:.0f}M - ${subset['ev'].max()/1e6:.0f}M")
        print(f"  Sectors: {dict(subset['sector'].value_counts().head(5))}")

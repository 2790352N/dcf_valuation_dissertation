"""
Sensitivity Analysis: WACC +/-1% and Terminal g +/-1%
Run: python src\sensitivity_analysis.py
"""

import pandas as pd
import numpy as np
import os

RESULTS_FILE = r"C:\Users\Max\Desktop\dcf_project\data\results\dcf_results_tier1_final.csv"
OUTPUT_FILE = r"C:\Users\Max\Desktop\dcf_project\data\results\sensitivity_results.csv"

TCJA_EFFECTIVE_YEAR = 2018
MARGINAL_TAX_POST_TCJA = 0.21
MARGINAL_TAX_PRE_TCJA = 0.35


def recompute_ev(forecast_df, wacc, terminal_g, ebit_margin, da_pct, marginal_tax):
    rows = forecast_df.copy()
    years = rows['year'].values
    ufcf = rows['ufcf'].values
    revenue = rows['revenue'].values

    disc = 1.0 / (1.0 + wacc) ** (years - 0.5)
    pv_ufcf = (ufcf * disc).sum()

    terminal_revenue = revenue[-1] * (1.0 + terminal_g)
    terminal_ebit = terminal_revenue * ebit_margin
    terminal_nopat = terminal_ebit * (1.0 - marginal_tax)
    terminal_da = terminal_revenue * da_pct
    terminal_capex = terminal_revenue * da_pct
    terminal_dnwc = 0.0
    terminal_ufcf = terminal_nopat + terminal_da - terminal_capex - terminal_dnwc

    if wacc <= terminal_g:
        return np.nan

    tv = terminal_ufcf / (wacc - terminal_g)
    pv_tv = tv * disc[-1]

    return pv_ufcf + pv_tv


def print_summary(label, subset):
    print(f"\n{'='*70}")
    print(f"{label} (N={len(subset)})")
    print(f"{'='*70}")
    print(f"{'Specification':<20s} {'Med Signed':>12s} {'Med Absolute':>14s}")
    print(f"{'-'*50}")
    for spec in ['baseline', 'wacc_plus_1', 'wacc_minus_1', 'g_plus_1', 'g_minus_1']:
        se_col = f'se_{spec}'
        ae_col = f'ae_{spec}'
        v = subset[subset[se_col].notna()]
        lbl = {
            'baseline': 'Baseline',
            'wacc_plus_1': 'WACC + 1%',
            'wacc_minus_1': 'WACC - 1%',
            'g_plus_1': 'Terminal g + 1%',
            'g_minus_1': 'Terminal g - 1%',
        }[spec]
        print(f"{lbl:<20s} {v[se_col].median():>+11.1%} {v[ae_col].median():>13.1%}")


def main():
    results = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(results)} observations")

    output_rows = []

    for idx, row in results.iterrows():
        ticker = row['ticker']
        forecast_path = row['forecast_file']

        if not os.path.exists(forecast_path):
            print(f"  SKIP {ticker}: forecast file not found at {forecast_path}")
            continue

        try:
            forecast = pd.read_csv(forecast_path)
        except Exception as e:
            print(f"  SKIP {ticker}: error reading forecast: {e}")
            continue

        wacc = row['wacc_used']
        g = row['terminal_growth']
        deal_ev = row['deal_ev']
        baseline_ev = row['ev']
        ebit_margin = row['ebit_margin']
        asof_year = row['asof_year']

        da_pct = (forecast['da'] / forecast['revenue']).iloc[0]

        if asof_year < TCJA_EFFECTIVE_YEAR:
            marginal_tax = MARGINAL_TAX_PRE_TCJA
        else:
            marginal_tax = MARGINAL_TAX_POST_TCJA

        specs = {
            'baseline': (wacc, g),
            'wacc_plus_1': (wacc + 0.01, g),
            'wacc_minus_1': (max(wacc - 0.01, g + 0.005), g),
            'g_plus_1': (wacc, min(g + 0.01, wacc - 0.005)),
            'g_minus_1': (wacc, max(g - 0.01, 0.005)),
        }

        result_row = {
            'ticker': ticker,
            'asof_year': asof_year,
            'ebit_margin': ebit_margin,
            'wacc_used': wacc,
            'terminal_growth': g,
            'deal_ev': deal_ev,
            'ev_original': baseline_ev,
        }

        for spec_name, (w, tg) in specs.items():
            ev = recompute_ev(forecast, w, tg, ebit_margin, da_pct, marginal_tax)
            if ev is not None and not np.isnan(ev) and deal_ev > 0:
                se = (ev - deal_ev) / deal_ev
                ae = abs(se)
            else:
                se = np.nan
                ae = np.nan

            result_row[f'ev_{spec_name}'] = ev
            result_row[f'se_{spec_name}'] = se
            result_row[f'ae_{spec_name}'] = ae

        output_rows.append(result_row)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(results)}")

    out = pd.DataFrame(output_rows)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Processed: {len(out)} firms")

    # Baseline check
    print(f"\n{'='*70}")
    print("BASELINE VERIFICATION")
    print(f"{'='*70}")
    valid = out[out['ev_original'].notna() & out['ev_baseline'].notna()]
    diff_pct = ((valid['ev_baseline'] - valid['ev_original']) / valid['ev_original'] * 100)
    print(f"Median EV difference (recomputed vs original): {diff_pct.median():.2f}%")
    print(f"Max absolute difference: {diff_pct.abs().max():.2f}%")
    print(f"Firms with >1% difference: {(diff_pct.abs() > 1).sum()}")

    # Summaries
    print_summary("FULL SAMPLE", out)

    strong = out[(out['ebit_margin'] >= 0.10) & (out['ebit_margin'] <= 0.20)]
    print_summary("STRONG MARGIN (EBIT 10-20%)", strong)

    floor = out[out['ebit_margin'] <= 0.02]
    print_summary("FLOOR FIRMS (EBIT <= 2%)", floor)

    profitable = out[out['ebit_margin'] > 0.02]
    print_summary("PROFITABLE (EBIT > 2%)", profitable)


if __name__ == '__main__':
    main()

# DCF Valuation Model — Undergraduate Dissertation

Automated FCFF/WACC DCF model applied to US public company acquisitions. All inputs are derived mechanically from trailing financial data via the EODHD API. Requires an active EODHD API key.

---

## File Descriptions

| File | Description |
|------|-------------|
| `main.py` | Entry point — runs the full valuation pipeline across all sample observations |
| `financials.py` | Parses EODHD fundamentals JSON into structured income statement, cash flow, and balance sheet objects |
| `data_loader.py` | Fetches and caches raw fundamentals from the EODHD API |
| `price_loader.py` | Fetches and caches daily EOD price data from EODHD, used for beta estimation and market cap computation |
| `dcf_inputs.py` | Builds historical driver tables (revenue, EBIT, D&A, capex, ΔNWC, tax rate) from parsed financials |
| `dcf.py` | Core DCF model — projects free cash flows over a ten-year horizon, computes terminal value via Gordon growth model, and returns enterprise value |
| `wacc.py` | Computes WACC from cost of equity (CAPM), cost of debt (synthetic rating approach), and market-value capital structure weights |
| `beta.py` | Estimates regression beta from five-year rolling monthly OLS regressions against the S&P 500, with Bloomberg/Vasicek adjustment |
| `historical_mcap.py` | Retrieves historical market capitalisation from EODHD; falls back to price × shares outstanding for earlier periods |
| `balance_sheet.py` | Extracts cash, total debt, and net debt from balance sheet items |
| `macro_loader.py` | Loads time-varying risk-free rate and implied ERP from `data/macro_series.csv`, matched to each valuation date to avoid look-ahead bias |
| `data_quality.py` | Runs data quality checks across drivers, WACC inputs, and valuation outputs; assigns error, warning, and info flags |
| `classify_sample.py` | Classifies observations into Tier 1, 2, and 3 based on data completeness and assigns EBIT margin groups |
| `cross_sectional.py` | Computes cross-sectional variables for hypothesis testing including earnings volatility (EBIT CV), leverage, and terminal value share of EV |
| `sensitivity_analysis.py` | Reruns valuations under four alternative specifications: WACC ±1% and terminal growth rate ±1% |

---

## Requirements

```
pip install pandas numpy requests scipy statsmodels matplotlib openpyxl python-dotenv
```

---

## Usage

**1. Set your API key**

Create a `.env` file in the project root:
```
EODHD_API_KEY=your_api_key_here
```

**2. Populate `tickers.txt`**

Create a `tickers.txt` file in the project root. Each line should contain a ticker and base acquisition year, comma-separated. The base year should be the fiscal year-end on or before deal completion:
```
TICKER.US,YEAR
```
For example:
```
AAPL.US,2022
MSFT.US,2021
```

**3. Run the model**

```
python main.py
```

Results are saved to `data/results/dcf_results.csv`. Individual forecast, sensitivity, and summary files are also saved per observation. Sensitivity analysis across WACC and terminal growth rate specifications can be run separately:

```
python sensitivity_analysis.py
```

---

*This repository is provided for inspection purposes. The EODHD API key and transaction dataset are not included.*

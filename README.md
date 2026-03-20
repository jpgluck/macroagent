# MacroAgent Forecaster

> **A fully open-source, free, educational Streamlit web app that lets any user
> upload their historical demand data and run "what-if" forecasts powered by
> real GDP growth and inflation data from the Federal Reserve FRED API.**

---

## Live Demo

*Coming soon — deploy your own for free on [Streamlit Cloud](https://streamlit.io/cloud)*

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## Zero Cost – Free FRED API Key

This app connects to the Federal Reserve's free FRED (Federal Reserve Economic Data)
API to pull official GDP and inflation data.

**Get your free key here (takes 30 seconds):**
👉 https://fredaccount.stlouisfed.org/apikeys

No credit card required. Personal and educational use only.

---

## What It Does

| Feature | Description |
|---------|-------------|
| **Data Upload** | Upload any CSV with `date` + `demand` columns |
| **FRED Integration** | Fetches 10+ years of Real GDP growth (A191RL1Q225SBEA) and CPI inflation (CPIAUCSL) |
| **Correlation Research** | Computes Pearson correlations and OLS regression weights |
| **What-If Scenarios** | Slider-driven GDP and inflation scenario builder |
| **AI Forecast** | Prophet model with macro regressors — 12-month forward projection |
| **Interactive Charts** | Plotly charts: forecast bands, decomposition, correlation heatmap |
| **CSV Download** | Download the forecast as a CSV for further analysis |

---

## Folder Structure

```
macroagent-forecaster-rutgers/
├── app.py                  # Main Streamlit app — the only file you run
├── requirements.txt        # Exact pinned dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Brand colours and theme
├── data/
│   └── sample_demand.csv   # Realistic 5-year monthly demand dataset
└── utils/
    └── fred_helper.py      # FRED API helper class (GDP, CPI, alignment, regression)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 (recommended)
- pip
- A free [FRED API key](https://fredaccount.stlouisfed.org/apikeys)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/jpgluck/macroagent-forecaster-rutgers.git
cd macroagent-forecaster-rutgers

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Sample Data

If you don't have your own CSV yet, use the included sample:
`data/sample_demand.csv` — 60 months of realistic monthly demand data (2021–2025).

---

## How to Use

1. **Upload** your demand CSV (sidebar → Step 1). Column names must contain `date` and `demand`.
2. **Connect** your free FRED API key (sidebar → Step 2) and click *Connect to Federal Reserve*.
3. **Research** historical correlations — click *Research Historical Correlations*.
4. **Set your scenario** with the GDP and inflation sliders (sidebar → Step 3).
5. **Generate** your 12-month macro-adjusted forecast.
6. **Download** the forecast CSV for reporting.

---

## Sample Screenshots

*(Add screenshots after first deploy)*

| Data Upload | Correlation Research | Forecast Output |
|-------------|---------------------|-----------------|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## ⚠️ Legal Disclaimer — FRED Terms of Use

This application accesses the Federal Reserve Bank of St. Louis FRED® API
**for educational and personal use only**, in accordance with their Terms of Use.

> **From the FRED Terms of Use (https://fred.stlouisfed.org/about/terms):**
>
> FRED data is made available to the public free of charge. Users may access,
> download, and use FRED data for personal and educational purposes. Commercial
> use, redistribution as a primary product, and use for AI/ML training require
> additional permissions from the Federal Reserve Bank of St. Louis.

**This app:**
- Requires users to supply their **own** free FRED API key.
- Does **not** store, cache, or redistribute FRED data.
- Is intended solely for **educational and personal use**.
- Is **not** investment advice.
- Displays mandatory compliance disclaimers throughout the UI.

If you use this app for any purpose, you are responsible for complying with
[FRED's Terms of Use](https://fred.stlouisfed.org/about/terms).

---

## About the Project

Built by a first-year **Rutgers University** student studying
**Supply Chain Management** and **Business Analytics & Information Technology (BAIT)**
as an educational exploration of how macroeconomic indicators influence
real-world demand patterns.

### Tech Stack
| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [Prophet](https://facebook.github.io/prophet/) | Time-series forecasting with additive regressors |
| [fredapi](https://github.com/mortada/fredapi) | Python wrapper for the FRED API |
| [pandas](https://pandas.pydata.org) | Data manipulation |
| [scikit-learn](https://scikit-learn.org) | OLS linear regression for correlation weights |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [NumPy](https://numpy.org) | Numerical operations |

---

## Contributing

Contributions are welcome! This is an open educational project.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-idea`)
3. Commit your changes (`git commit -m 'Add my idea'`)
4. Push to the branch (`git push origin feature/my-idea`)
5. Open a Pull Request

Please open an Issue first to discuss major changes.

If this project helped you, please ⭐ **star the repo** — it helps other students find it!

---

## License

MIT License — free to use, share, and modify with attribution.
See [LICENSE](LICENSE) for details.

---

*MacroAgent Forecaster · Rutgers BAIT / SCM Student Project · Educational Use Only*

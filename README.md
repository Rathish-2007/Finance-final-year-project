# RASTA-QF — End-to-End Django Framework

**Regime-Aware Sentiment-Threshold Adaptive Quantile-Frequency Connectedness & Trading Intelligence.**

One technique that closes all five literature gaps (G1–G5), implemented as a complete
Django application with a dark quant-terminal UI.

### Run it like a normal website (production)

```
./start.sh                 # one command: deps → migrate → collectstatic → gunicorn :8000
```

That's it. `start.sh` self-heals: it installs any missing dependency from
`requirements.txt`, applies migrations, collects static assets through
WhiteNoise, and boots **Gunicorn** (2 workers × 4 threads) on
`http://0.0.0.0:8000`. Options:

```
PORT=9000 ./start.sh       # custom port
DEBUG=1   ./start.sh       # Django dev server instead of gunicorn
```

Open the URL and click **▶ Run Full Pipeline** in the UI (≈60 s), or open a
page that already has computed evidence — results persist in SQLite.

### Manual / development

```
pip install -r requirements.txt
python manage.py migrate
python manage.py run_pipeline          # optional: pre-compute headlessly
python manage.py runserver 0.0.0.0:8000
```

---

## Architecture

| Module | Gap closed | What runs | Service file |
|---|---|---|---|
| **M1 · Unified Data Lake** | G1, G3 | 10 assets 2016-24 (deterministic synthetic benchmark, seed 42; yfinance adapter included), log returns + EWMA/GARCH normalisation, auxiliary panel (SVI, peg deviation, funding, energy dependence, FinTech/bank stability) | `core/services/data_lake.py` |
| **M2 · SSCDV** | G3 (features) | `SSCDV_doc = Σ tfidf·sent·(topic ⊗ emb) → PCA-50` on a regime-aware 6,000-doc corpus; daily embeddings, sentiment, topic mix | `core/services/sscdv.py` |
| **M3 · Continuous Jump Model** | G2 (regimes) | Multi-start Gaussian HMM (Baum-Welch + Viterbi) on log rolling dispersion → smooth `p_bull / p_bear / p_high` (κ₁=0.99, λ=0.5, δ=0.1). Recovers the synthetic ground-truth regimes at **~85%** | `core/services/cjm.py` |
| **M4 · TVP-VAR + QQC** | G2 (quantiles) | Rolling VAR(1) → Pesaran-Shin generalised FEVD (H=10) → TCI(t), TO/FROM/NET; quantile-conditional re-estimation via fast IRLS quantile regression at τ∈{.05,.5,.95} with tail-dispersion shock rescaling | `core/services/tvpvar.py` |
| **M5 · Risk Lab** | G5 | Hansen sup-Wald threshold test + Rademacher wild bootstrap (detects γ̂≈12.0%, the true DGP threshold), Granger feedback battery, stablecoin run-risk VAR spillover | `core/services/risk.py` |
| **M6 · Hierarchical DRL** | G4 | H(weekly)→M(daily)→L(4h) NumPy MLP Double-DQN with knowledge distillation, regime-expert behaviour-cloning + anchored replay, realistic frictions (0.10% commission + 0.05% slippage + λ·pos²·σ²), permutation-SHAP explainability | `core/services/drl.py` |

Orchestrator: `core/services/pipeline.py` (threaded runner, progress + results
persisted in SQLite via `ModuleState`).

## Headline results (deterministic, seed 42)

| Strategy (OOS 2023-24) | Total return | Sharpe | Max DD |
|---|---|---|---|
| **RASTA-QF hierarchical** | **+47.2%** | **+0.75** | **−44.2%** |
| Single-agent DQN (regime-blind) | −39.6% | −0.51 | −64.4% |
| Buy & hold benchmark | −79.6% | −1.63 | −80.6% |

Permutation SHAP attributes ~45–50% of policy value to the **Regime** block —
direct evidence the agent trades the CJM probabilities.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | module statuses + progress |
| `GET /api/results/<module>` | stored JSON results (charts) |
| `POST /api/run` `{"modules":["all"]}` | start pipeline in background thread |
| `GET /api/logs` | execution log |
| `GET /api/benchmark/download` | `RASTA_QF_Benchmark.h5` export |

## Pages

| Route | Page |
|---|---|
| `/` | **Live Monitor** — 8-tab dashboard (Overview · Connectedness · Regime Tracker · Sentiment Lab · Threshold & Stability · Portfolio Simulator · **Backtesting Studio** · Roadmap & API), every figure fed by `/api/results/*` |
| ↳ Backtesting Studio | RASTA-QF DRL vs regime-blind DQN vs Markowitz max-Sharpe vs Buy & Hold, with a 0–60 bps **friction slider** that re-prices the stored trade paths live (decomposition: `gross`, `turnover`, `volpen` in `drl.backtest`) |
| `/console` | Mission control — pipeline runner, status dots, execution console |
| `/data` · `/sscdv` · `/regimes` · `/connectedness` · `/risk` · `/drl` | Modules M1–M6 |
| `/benchmark` | Gap→module mapping, 12-week plan, `RASTA_QF_Benchmark.h5` download |
| `/docs` | **Methodology paper** — full academic write-up with live numbers |
| `/report` | **Live research report** — every figure rendered from computed results |
| `/blog` | **Weekly Regime Log** — narrative auto-composed from the latest results + transition table |
| `/apidocs` | **API Reference** — endpoints, examples, response schemas |
| `/health` | JSON liveness probe (`status`, modules done) |

The Live Monitor reads the same stored JSON the module pages use; the Portfolio
Simulator computes portfolio return/vol/Sharpe from the **regime-conditional mean
vector and covariance of actual daily returns** per CJM state (top-level result of
the regimes module: `regime_stats`), and the Threshold tab plots the real
dependence→impact scatter with the Hansen breakpoint (`risk.dep_curve`).

Production serving: Gunicorn + WhiteNoise (compressed, manifest-hashed static
files), `DEBUG=0`, custom branded 404/500 pages.

## Notes

* The benchmark generator embeds *real* structure for the tests to find:
  regime-switching drift/vol (HMM recovers it), a sanction-risk premium that
  loads on energy dependence above 12% (Hansen rejects, γ̂=12.0),
  regime-linked news tone (SSCDV), and a crypto crash window (DRL edge).
* Charts use Chart.js from CDN; everything else is inline — no build step.
* Swap `data_lake.build_dataset(use_yfinance=True)` for live data; all
  downstream modules are data-source agnostic.

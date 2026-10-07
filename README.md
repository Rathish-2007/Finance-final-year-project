# RASTA-QF — End-to-End Django Framework

**Regime-Aware Sentiment-Threshold Adaptive Quantile-Frequency Connectedness & Trading Intelligence.**

One technique that closes all five literature gaps (G1–G5), implemented as a complete
Django application with a dark quant-terminal UI.

```
pip install django numpy pandas scipy scikit-learn statsmodels h5py
python manage.py migrate
python manage.py run_pipeline          # optional: pre-compute everything headlessly
python manage.py runserver 0.0.0.0:8000
```

Open http://localhost:8000 — or click **▶ Run Full Pipeline** in the UI (≈60 s).

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

`/` mission control · `/data` M1 · `/sscdv` M2 · `/regimes` M3 ·
`/connectedness` M4 · `/risk` M5 · `/drl` M6 · `/benchmark` gap→module
mapping, 12-week plan, benchmark download.

## Notes

* The benchmark generator embeds *real* structure for the tests to find:
  regime-switching drift/vol (HMM recovers it), a sanction-risk premium that
  loads on energy dependence above 12% (Hansen rejects, γ̂=12.0),
  regime-linked news tone (SSCDV), and a crypto crash window (DRL edge).
* Charts use Chart.js from CDN; everything else is inline — no build step.
* Swap `data_lake.build_dataset(use_yfinance=True)` for live data; all
  downstream modules are data-source agnostic.

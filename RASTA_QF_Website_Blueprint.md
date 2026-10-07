# RASTA-QF — Website Blueprint: from Research Framework → Live Web Platform

*Companion to the running site. Everything below is organised so nothing is missed;
the shipped columns reflect what this repository already implements.*

---

## 1 · Positioning (what the website can be)

| Identity | Audience | What they get | Return |
|---|---|---|---|
| **A. Research showcase** | PhD committees, journals, conferences | Interactive paper: live charts, replication hub, benchmark download | Citations, credibility |
| **B. Analytics platform (SaaS)** | Quant/ESG funds, fintechs, banks | Regime tracker, connectedness API, alerts, portfolio simulator | Subscriptions |
| **C. Teaching / community hub** | Students, universities, CFA candidates | Courses, tutorials, leaderboard, notebooks | Courses, sponsorships |

**Recommended path:** start as A (done — this site), add B features progressively, seed C through the benchmark release.

---

## 2 · Sitemap (12 pages) — ✅ shipped / ⏳ planned

| # | Page | Route | Status |
|---|---|---|---|
| 1 | **Live Monitor** — TCI, regime nowcast, sentiment, top transmitter, alerts feed | `/` | ✅ |
| 2 | **Connectedness Explorer** — TCI + event shading, quantile nets, pairwise drill-down | (tab) | ✅ |
| 3 | **Regime Tracker** — probability bands, calendar, A⁵ transition matrix, nowcast | (tab) | ✅ |
| 4 | **Sentiment Lab (SSCDV)** — topic × sentiment view, sentiment series, driver table | (tab) | ✅ |
| 5 | **Threshold & Stability** — dependence→impact scatter, stablecoin monitor, Granger battery | (tab) | ✅ |
| 6 | **Portfolio Simulator** — sliders → return/vol/Sharpe/DD from regime-conditional moments | (tab) | ✅ |
| 7 | **Backtesting Studio** — upload strategy vs RASTA-QF DRL vs buy&hold; cost slider | ⏳ | — |
| 8 | **Benchmark Download** — RASTA_QF_Benchmark.h5, data card, citation block (Gap 3!) | `/benchmark` + `/api/benchmark/download` | ✅ |
| 9 | Paper Replications — one-click notebooks of source papers | ⏳ | — |
| 10 | **API Docs** — endpoints, examples, response schemas | `/apidocs` | ✅ |
| 11 | **Blog / Research Log** — auto-generated regime report + transition table | `/blog` | ✅ |
| 12 | **Methodology Paper + Research Report + Mission Control** | `/docs` `/report` `/console` | ✅ |

---

## 3 · Feature catalogue — "what more can be done"

### 3.1 Data layer
- Live connectors: Yahoo Finance *(adapter present: `USE_YFINANCE=True`)*, Binance, CoinGecko, Trading Economics, Google Trends, GPR index
- Alternative data v2: Reddit, Telegram, RBI/SEBI releases, climate policy uncertainty
- Frequency upgrade: 4-h crypto + daily TradFi + weekly macro alignment engine
- **India module:** NSE Nifty, sovereign green bonds, MCX Gold, BTC-INR, USDINR — unique local angle
- Data-quality dashboard; benchmark versioning v1/v2/v3 each with a Zenodo DOI

### 3.2 Analytics & model extensions
- **Nowcasting** regime probabilities intraday (the daily CJM is already the engine; intraday = update step)
- Regime **transition forecast** — done in the Regime tab (A⁵); extend to k-day horizons
- Tail module: CoVaR, MES, ΔCoVaR for green bond & stablecoin tail risk
- Multi-scale wavelet × quantile × frequency cube (downloadable Plotly figure)
- Animated network evolution (nodes = assets, edges = spillovers, time scrubber)
- Explainable-AI force plots per day — *feature-group SHAP already shipped in Sentiment tab*
- LLM narrative layer over the numbers — *currently deterministic template narrative (`/blog`)*

### 3.3 Product features
- Alerts: email / Telegram / Slack ("TCI crossed 70", "regime switched")
- Watchlists & custom portfolios; strategy leaderboard (paper trading)
- PDF report generator; screener query builder; white-label mode; PWA/mobile

### 3.4 Community & academic
- Replication hub (Binder/Docker) for the source literature
- Crowd regime labelling → inter-rater statistics
- Open method leaderboard on the benchmark; course mode; data donation portal

### 3.5 Business
- API tiers (Free 100/day → Pro → Enterprise + websocket)
- Saved backtests, team workspaces, SSO
- **Regime-as-a-Service** daily feed; consulting page; scholar program

---

## 4 · Technical architecture

### As built (this repo — Track 1 complete)
```
Frontend   Django templates + vanilla-JS canvas charts (no CDN dependency)
Server     Gunicorn (2×4 gthread) + WhiteNoise, DEBUG off, one-command ./start.sh
Pipeline   threaded orchestrator; progress + results persisted in SQLite
Modules    data → SSCDV → CJM → TVP-VAR/QQC → risk lab → hierarchical DRL
Benchmark  RASTA_QF_Benchmark.h5 (HDF5, self-describing)
Determinism seed 42 everywhere — every number reproducible
```

### Scale path (Track 2)
```
Frontend  Next.js + Tailwind + ECharts        API    FastAPI (REST + WS)
Workers   Prefect/Celery nightly pipeline     DB     PostgreSQL + TimescaleDB
Files     S3/R2 for .h5/.parquet              ML     ONNX Runtime (CJM, DRL)
Auth      Supabase  ·  Billing Razorpay+Stripe
Host      Fly.io / Render / Hetzner           CI/CD  GitHub Actions → Docker
```

### Nightly pipeline schedule
```
18:00 IST  fetch prices + news + SVI        18:45  CJM regime inference
18:15      clean → log returns → GARCH      19:00  TVP-VAR + QQC → TCI by regime/quantile
18:30      SSCDV embedding → PCA-50         19:15  threshold + stablecoin + SHAP
                                            19:30  publish JSON → alerts → regenerate report
```

### API (v1 live now, v1.1 planned)
```
GET  /api/status · /api/results/<module> · /api/logs · /health
POST /api/run {"modules":["all"]}
GET  /api/benchmark/download
—— planned ——
GET  /v1/tci?start=&end=&quantile=  ·  /v1/regime/current  ·  /v1/connectedness/net
GET  /v1/sentiment/sscdv?date=  ·  POST /v1/portfolio/stats  ·  WS /v1/stream
```

---

## 5 · Deployment & costs

| Stage | Stack | ₹/month |
|---|---|---|
| Demo (now) | this repo, any VPS / HF Spaces | 0–500 |
| Growth | Hetzner CX22 + R2 + Razorpay | 800–1,500 |
| Scale | 2× VPS + managed Postgres + CDN | 5,000–15,000 |
| Domain | rastaqf.com / .in / .ai | ~1,000/yr |

**Compliance (India):** market-data licensing (yfinance OK for research, not commercial redistribution — document it);
"analytics & research tool, not investment advice" wording; DPDP consent if accounts; MIT code + CC-BY benchmark.

---

## 6 · How the website strengthens the paper

- **Gap 3 solved visibly** — the benchmark download page *is* the contribution: citable, DOI-ready (Zenodo).
- **Design-science framing** — SoftwareX / Expert Systems with Applications, website as the artefact.
- **Altmetric boost** — live dashboard + replication hub get clicked and cited.
- **Reproducibility** — seed 42 + Docker + Binder: reviewers rerun everything.

---

## 7 · 12-week roadmap

| Weeks | Deliverable | Status |
|---|---|---|
| 1–2 | Data lake + dashboard skeleton | ✅ done |
| 3–4 | SSCDV embedding + Sentiment page | ✅ done |
| 5–6 | CJM + TVP-VAR/QQC + Connectedness/Regime pages | ✅ done |
| 7 | Threshold & Stability page; benchmark v1 + DOI | ✅ page · ⏳ DOI |
| 8–10 | DRL + Backtesting studio + Portfolio simulator | ✅ done |
| 11 | Alerts, API v1, auto regime log | ✅ API + log · ⏳ alerts |
| 12 | Landing polish, SEO, paper draft, launch | ⏳ |

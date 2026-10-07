"""
Module 2 — SSCDV: Supervised Sentiment-Contextualised Document Vectors.

Implements the joint topic⊗sentiment embedding of Ueda et al.:

    SSCDV_doc = Σ_w tfidf(d,w) · sent(w) · [ topic(w) ⊗ emb(w) ]  → PCA(50)

For the offline benchmark we generate a realistic synthetic financial news
corpus (6,000 documents, 5 topics, regime-linked tone) and derive:
  * a T×50 daily document-embedding panel,
  * daily market sentiment,
  * topic-mix time series,
all consumed by the DRL state space and the paper's explainability analysis.
Swap `generate_corpus` for NewsAPI output in production — the formula is
identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

TOPICS = ["Markets", "Crypto", "Green/ESG", "Macro", "Regulation"]

TOPIC_WORDS = {
    "Markets": ["equities", "rally", "earnings", "index", "stocks", "dividend", "valuation",
                "futures", "ipo", "buyback", "profit", "guidance", "volume", "blue-chip"],
    "Crypto": ["bitcoin", "ethereum", "token", "defi", "blockchain", "stablecoin", "exchange",
               "wallet", "mining", "altcoin", "protocol", "onchain", "halving", "web3"],
    "Green/ESG": ["green-bond", "climate", "renewable", "solar", "wind", "esg", "carbon",
                  "emission", "sustainable", "transition", "taxonomy", "net-zero", "impact", "grid"],
    "Macro": ["inflation", "rates", "fed", "gdp", "recession", "employment", "cpi", "yields",
              "liquidity", "fiscal", "supply-chain", "growth", "monetary", "dollar"],
    "Regulation": ["sec", "rules", "compliance", "sanctions", "policy", "oversight", "basel",
                   "directive", "audit", "licensing", "enforcement", "miifid", "watchdog", "law"],
}

POS_WORDS = ["surge", "gain", "record", "growth", "outperform", "upgrade", "boom", "soar",
             "rally", "beat", "strong", "optimism", "breakthrough", "adoption", "milestone"]
NEG_WORDS = ["crash", "plunge", "loss", "default", "downgrade", "fear", "collapse", "selloff",
             "fraud", "hack", "bankruptcy", "recession", "tumble", "warning", "liquidation"]

ASSET_TAGS = {
    "Markets": "SPX", "Crypto": "BTC", "Green/ESG": "GRNB", "Macro": "SPX", "Regulation": "SPX",
}


def generate_corpus(index: pd.DatetimeIndex, states: np.ndarray,
                    n_docs: int = 6000, seed: int = 11):
    """Regime-aware synthetic news corpus (documents with date, topic, asset, text)."""
    rng = np.random.default_rng(seed)
    T = len(index)
    docs = []
    dates = index[rng.integers(0, T, n_docs)]
    state_of = {d: states[np.searchsorted(index, d)] for d in set(dates)}

    for i in range(n_docs):
        d = dates[i]
        s = state_of[d]
        # topic mix shifts with regime: crisis → more Regulation/Macro coverage
        w = np.array([0.30, 0.28, 0.18, 0.14, 0.10])
        if s == 2:
            w = np.array([0.18, 0.30, 0.10, 0.22, 0.20])
        elif s == 1:
            w = np.array([0.24, 0.24, 0.12, 0.24, 0.16])
        k = rng.choice(5, p=w / w.sum())
        topic = TOPICS[k]
        words = list(rng.choice(TOPIC_WORDS[topic], size=rng.integers(12, 26)))
        # tone: bullish regimes positive, crisis negative
        n_sent = rng.integers(2, 6)
        p_pos = {0: 0.72, 1: 0.42, 2: 0.22}[s]
        for _ in range(n_sent):
            words.append(rng.choice(POS_WORDS) if rng.random() < p_pos else rng.choice(NEG_WORDS))
        rng.shuffle(words)
        docs.append({"date": d, "topic": topic, "asset": ASSET_TAGS[topic],
                     "text": " ".join(words)})
    return pd.DataFrame(docs)


def _lexicon_scores():
    sent, toki = {}, {}
    for w in POS_WORDS:
        sent[w] = 1.0
    for w in NEG_WORDS:
        sent[w] = -1.0
    vocab = sorted(set(w for ws in TOPIC_WORDS.values() for w in ws) | set(sent))
    for j, w in enumerate(vocab):
        toki[w] = j
    return vocab, toki, sent


def build_sscdv(corpus: pd.DataFrame, n_components: int = 50, seed: int = 5):
    """Compute the full SSCDV pipeline; returns daily embeddings + analytics."""
    vocab, toki, sent_lex = _lexicon_scores()
    V = len(vocab)
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((V, 32))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    docs = corpus.copy()
    docs["tokens"] = docs["text"].str.split()

    # --- tf-idf -------------------------------------------------------
    dfreq = np.zeros(V)
    bow = []
    for toks in docs["tokens"]:
        seen = set()
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            seen.add(t)
        for t in seen:
            if t in toki:
                dfreq[toki[t]] += 1
        bow.append(counts)
    idf = np.log((len(docs) + 1) / (dfreq + 1)) + 1.0

    K = len(TOPICS)
    topic_ix = {t: k for k, t in enumerate(TOPICS)}
    dim = K * 32
    M = np.zeros((len(docs), dim))
    doc_sent = np.zeros(len(docs))
    doc_topic = np.zeros((len(docs), K))

    for i, counts in enumerate(bow):
        toks_total = sum(counts.values()) or 1
        for w, c in counts.items():
            j = toki.get(w)
            if j is None:
                continue
            tfidf = (c / toks_total) * idf[j]
            s = sent_lex.get(w, 0.0)
            # soft topic attribution: dominant topic of the document
            k = topic_ix.get(docs["topic"].iloc[i], 0)
            M[i, k * 32:(k + 1) * 32] += tfidf * max(abs(s), 0.15) * np.sign(s or 1.0) * emb[j]
            doc_sent[i] += tfidf * s
        doc_topic[i, topic_ix[docs["topic"].iloc[i]]] = 1.0

    norm = np.linalg.norm(M, axis=1, keepdims=True)
    M /= np.where(norm == 0, 1, norm)
    doc_sent /= np.clip(np.array([sum(1 for t in toks if t in sent_lex)
                                  for toks in docs["tokens"]]), 1, None)

    # --- PCA to 50 dims -----------------------------------------------
    pca = PCA(n_components=n_components, random_state=seed)
    sscdv_full = pca.fit_transform(M)

    # --- daily aggregation ----------------------------------------------
    docs["sscdv"] = list(sscdv_full)
    docs["sent"] = doc_sent
    daily = docs.groupby("date")
    daily_emb = np.array([np.mean(g["sscdv"].tolist(), axis=0) for _, g in daily])
    daily_sent = daily["sent"].mean()
    daily_topic = pd.DataFrame(
        np.array([g["topic"].value_counts(normalize=True).reindex(TOPICS, fill_value=0)
                  for _, g in daily]),
        index=daily_sent.index, columns=TOPICS)
    daily_news = daily.size()

    idx = pd.DatetimeIndex(daily_sent.index)
    return {
        "daily_emb": pd.DataFrame(daily_emb, index=idx,
                                  columns=[f"SSC{i+1}" for i in range(n_components)]),
        "daily_sent": daily_sent,
        "daily_topic": daily_topic,
        "daily_news": daily_news,
        "pca_var": float(pca.explained_variance_ratio_.sum()),
        "n_docs": len(docs),
        "vocab_size": V,
        "topic_totals": docs["topic"].value_counts().reindex(TOPICS).to_dict(),
        "docs": docs.drop(columns=["sscdv"]),
    }

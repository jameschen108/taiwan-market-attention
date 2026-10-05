# Long-Tail Attention in Taiwan Stocks (v2)

[中文](README.md) | English

Does investor attention that builds up while the market is closed show up in next week's stock prices?

Li, Liu, Ye, Zhao & Zhao, *"It Depends on When You Search"* (MIS Quarterly,
[SSRN 4370525](https://ssrn.com/abstract=4370525)), used Google search volume for the
S&P 500 (2004–2019) and found that abnormal search volume on weekends predicts
next-week returns, while weekday search volume does not. This project applies the same
research design to long-tail stocks in Taiwan, measuring attention by the number of posts on
the PTT Stock board instead of search volume. Every deliberate departure from the original
paper is listed in [`PROJECT.md`](PROJECT.md) §0.2.

v2 redoes `taiwan-attention-long-tail` (v1, not public) with a new corpus: posts crawled
directly from ptt.cc, which include commenter accounts and timestamps, over a shorter period
of 2020–2024. All stages are complete and all 327 tests pass.

> **Read before using**
> - All results are **diagnostic**: licensed data such as news volume, analyst coverage and
>   four-factor returns were not obtained, so the results are not formal conclusions.
>   Switching corpora does not lift this label.
> - The population of inference is "the 267 stocks covered by this sample", not "Taiwan
>   stocks" (see [Sample](#sample) below).
> - Any citation of a result must also cite [`LIMITATIONS.md`](LIMITATIONS.md).

All linked documents other than this README are written in Chinese.

---

## Main findings

Full details are in [`FINDINGS.md`](FINDINGS.md); section numbers are given in parentheses.
Terms are explained under [Glossary](#glossary) below.

1. **Attention accumulated while the market is closed predicts next week's opening gap (§4).**
   Abnormal attention outside trading hours positively predicts the opening gap on the first
   trading day of the next week; attention accumulated during trading hours shows no
   detectable relationship. This contrast is significant at the 1% level in all twelve
   combinations of three specifications × two inference standards × with or without
   excluding limit-hit weeks. The effect is stable but small: one standard deviation more
   attention raises the gap by 5.2 basis points, about one third of the mean gap and 3.8%
   of its standard deviation.
   - This is a **predictive relationship, not a causal one**: the event study designed for
     causal identification cannot identify the effect in this design (§6.4).
   - This is **not a tradable signal**: a strategy can trade at the open at the earliest,
     which misses the gap exactly; portfolios built on the same signal are significantly
     negative after costs in all six versions (§8.2).
   - The original paper did not separate out the opening gap, so this is new evidence from
     this study, **not a replication** (§3).

2. **Whether the original paper's weekend effect is significant in this sample depends on
   the estimation method (§7).** Following the original paper (firm fixed effects,
   clustering by firm only), the weekend coefficient is significant at the 1% level in all
   three specifications; under this study's main specification (firm and week fixed
   effects, two-way clustering by firm and week), it is not significant in any of the three.

3. **The original paper's weekday/weekend split is confounded in Taiwan (§5).** 77% of
   attention in the weekday window occurs after the market closes, while weekends account
   for only 20.2% of the attention accumulated while the market is closed. Split by calendar,
   both windows carry the same signal and cannot be separated; split by whether the market is
   open, the comparison is clean.

4. **The mechanisms cannot be separated (§6).** The project designed several tests in
   advance to distinguish an "information processing" explanation from a "price pressure"
   explanation, and none of them discriminates: some results are predicted by both
   explanations, some have a shape neither predicts, and some do not survive a change of
   attention measure.

5. **The new corpus is 14% smaller than the old archive, and the loss is not random
   ([`LIMITATIONS.md`](LIMITATIONS.md) §2).** The 2020–2024 ptt.cc crawl (143,637 posts)
   is entirely contained in the older pttweb.cc archive (167,142 posts). The missing
   23,505 posts were deleted on the official site and are concentrated in two categories,
   `請益` (asking for advice) and `心得` (personal experience). See
   [`audit/P0_corpus_integrity.md`](audit/P0_corpus_integrity.md).

## Sample

| | |
|---|---|
| Stocks | 267 TWSE-listed common stocks (260 usable) across 28 TWSE industry categories |
| Selection rule | The first 10 stocks in each industry, by ascending security code |
| Period | 2020-01-12 to 2024-12-29, 260 weeks; 2019 serves only as a warm-up period for look-back windows |
| Corpus | Posts crawled directly from the ptt.cc Stock board: 143,637 posts and 15.63 million comments (including 2019) |
| Panel | 64,980 rows (stock × week) |

Taiwanese security codes were assigned roughly in order of listing, so this rule picks the
10 longest-listed stocks in each industry: older firms of small to mid size. Financial
holding companies and the main index heavyweights (MediaTek, Quanta, Chunghwa Telecom and
others) are not in the sample. This is why the population of inference can only be "the 267
stocks covered by this sample".

## Glossary

| Term | Meaning |
|---|---|
| Abnormal attention | A stock's number of PTT posts this week, minus its average over the previous 8 weeks (both after `log1p`). It counts posts, not searches, so it is not called ASVI |
| Trading hours / outside trading hours | 09:00–13:30 on trading days is trading hours; everything else (weekday pre- and post-market, weekends, holidays) is outside trading hours |
| Weekday / weekend | The original paper's calendar split: Monday to Friday, Saturday and Sunday |
| Opening gap | The return from the last trading day's close of the previous week to the first trading day's open of this week |
| Two inference standards | This study's main specification (firm and week fixed effects, two-way clustering) and the original paper's approach (firm fixed effects only, clustering by firm only). The two also differ in fixed effects. Every main table reports both |
| Specifications A′ / B / C | Used to separate the effects of period and corpus: A′ is the old archive, 2015–2024; B is the old archive, 2020–2024; C is the new crawl, 2020–2024 (main specification) |
| diagnostic | Label for results estimated without the full set of control variables; see [`PROJECT.md`](PROJECT.md) §7 |

## Documents

All documents below are in Chinese.

| Document | Contents |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | Full write-up of the findings. Read its four rules that apply throughout before citing any result |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Known limitations. Any citation of a result must also cite this document |
| [`PROJECT.md`](PROJECT.md) | Single source of truth for the research design and variable definitions. When the code disagrees with it, the code is changed |
| [`REPLICATION.md`](REPLICATION.md) | Rebuilds every table from the raw data; every command has been run and verified |
| [`docs/PLAN_V2.md`](docs/PLAN_V2.md) | The v2 work plan (for reference, not a specification) |
| [`audit/`](audit/) | Audit reports and audit tables for each stage |

## Data

| Path | Contents | Version-controlled |
|---|---|---|
| `data/pttcc/stock_<year>.jsonl` | New corpus, 2019–2024, 158,395 posts / 15.63 million comments | ✗ (3.33 GB) |
| `data/pttweb/` → v1 | Old mirror archive, used for specifications A′ / B | ✗ (symlink) |
| `data/raw/`, `data/twse/`, `data/external/` → v1 | Prices, trading by the three major institutional investor groups, ex-rights/ex-dividend events, capital reductions | ✗ (symlink) |
| `data/universe_267.csv`, `config/universe.yaml` | Research universe | ✓ |

Market data cover 2014-01 to 2025-03, which spans the whole v2 period, and were not
re-collected. The corpus and market data are not under version control and must be obtained
separately to rebuild; see [`REPLICATION.md`](REPLICATION.md) §0.

## Reproduction

```bash
pip install -r requirements.txt
python3 -m pytest -q                  # 327 tests
python3 -m src.audit_expected         # check 154 frozen values
```

The full steps for rebuilding every table from raw data are in
[`REPLICATION.md`](REPLICATION.md).

## Naming rules

When citing this project, the following terms have fixed wording (full list in
[`LIMITATIONS.md`](LIMITATIONS.md) §17):

| Do not write | Write instead |
|---|---|
| ASVI / SVI | abnormal PTT attention |
| retail order imbalance / net retail buying | non-institutional order imbalance (excluding Taiwan's three major institutional investor groups) |
| Taiwan stocks / Taiwan listed companies | the 267 stocks covered by this sample |
| investable performance | mechanical annualization on an ex-post universe |
| all PTT discussion | posts covered by this corpus and not deleted as of the crawl |

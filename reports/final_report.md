# SearchLab: Final Research Report

## Question and outcome

This completed study asked whether a lightweight model could allocate cross-encoder reranking under a limited CPU budget. The data showed substantial selection headroom and a useful retrieval-disagreement heuristic. The two fixed learned routers did not beat that heuristic at the primary 30% budget. We retained and reported the unsuccessful models, benchmarked the systems workload, and closed the study without further development-set-driven model search.

## Data and protocol

The BEIR distribution of NFCorpus contains 3,633 shared documents. We audited 2,590 training queries and 324 development queries. Six training queries in duplicate groups overlapping development were excluded, leaving 2,584 routing-training queries and 2,566 groups. Grouping used normalized exact matching or token-set Jaccard >= 0.85 with connected components. It does not guarantee semantic deduplication.

BM25 and a pinned all-MiniLM-L6-v2 model each retrieved 100 documents. Equal-weight RRF with constant 60 produced a fixed top-20 pool. A pinned ms-marco-MiniLM-L2-v2 cross-encoder reranked exactly that pool on CPU FP32. Pair length was limited to 256 tokens, batch size was 8 within a query, and PyTorch used four intra-op and one inter-op thread. No query-level neural batching was used for reranking.

The primary score was mean binary-relevance NDCG@10: original grades > 0 mapped to 1, originals retained, ideal DCG computed from all positive qrels, and unjudged documents scored as zero. Training grades were all 1; development included 1 and 2. Scores must not be compared directly with graded-relevance BEIR leaderboard figures.

The routing label was reranked NDCG minus base NDCG. Positive/negative classification used a 1e-6 tolerance. Budgets were 0%, 10%, 30%, 50%, 70%, and 100%, with floor rounding and predefined ID tie breaks. Random averaged seeds 0–9.

## Opportunity analysis

| Gain class | Train count | Dev count |
|---|---:|---:|
| Positive | 823 | 113 |
| Zero | 978 | 127 |
| Negative | 783 | 84 |

There were 1,606 nonzero training examples. Development positive gains averaged +0.118362 and negative gains -0.095200. Zero gain was not simply an already-perfect-ranking phenomenon: 82 of 127 zero-gain development queries had no judged positive candidate, seven had perfect base NDCG, and 38 were other unchanged-score cases. Incomplete judgments limit any semantic interpretation.

At 30%, the diagnostic oracle improved development NDCG by +0.040642, exceeding the +0.016599 improvement from always reranking. Since candidate pools were fixed, this came from avoiding harmful/no-benefit actions rather than finding additional documents. Oracle decisions use outcomes and are unavailable at inference.

## Strategy comparison

| Development policy | Calls / 324 | Binary NDCG@10 |
|---|---:|---:|
| Never | 0 | 0.301644 |
| Always | 324 | 0.318243 |
| Random | 97 | 0.306626 |
| RRF score margin | 97 | 0.306594 |
| Retrieval disagreement | 97 | 0.313037 |
| Logistic regression | 97 | 0.304314 |
| Regression tree | 97 | 0.310388 |
| Oracle | 97 | 0.342286 |

Retrieval disagreement prioritized low overlap between the independent BM25 and dense top-20 result sets. Its development improvement over Random was +0.006411, with a conditional paired group-bootstrap 95% interval [+0.001054, +0.012586]. Score margin did not establish an advantage over Random.

## Learned predictability

The two predefined models were StandardScaler plus LogisticRegression(C=1, max_iter=1000) for positive gain and DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42) for continuous gain. Inputs were 12 predefined query/retrieval features. Judgments, actual gains, candidate relevance counts, and cross-encoder outputs were excluded from inference inputs. Preprocessing was fitted inside each training fold.

| Held-out fold | Fit / validation queries | Logistic minus disagreement @30% | Tree minus disagreement @30% |
|---|---|---:|---:|
| 1 | 1,722 / 862 | -0.006939 | -0.000461 |
| 2 | 1,723 / 861 | -0.006040 | -0.002906 |
| 3 | 1,723 / 861 | -0.004649 | +0.000674 |

Duplicate groups stayed within folds and none crossed the training/development boundary. Final models fitted all eligible training queries and predicted development without development fitting.

The development difference from disagreement was -0.008723 for logistic regression, interval [-0.017970, +0.000013], and -0.002649 for the tree, interval [-0.007747, +0.002082]. Neither model established improvement, and the fold results did not support stable superiority. The primary learned-predictability criterion was not met.

These 1,000-replicate bootstrap intervals condition on fixed fitted policies and selected queries. They do not refit models, reallocate budgets, correct multiple comparisons, or estimate all training instability. Development was repeatedly inspected and is not an unbiased final test.

## Failure analysis

At 30%, disagreement selected 35 positive, 51 zero, and 11 negative outcomes. Logistic selected 39 positive, 24 zero, and 34 negative outcomes. The tree selected 33 positive, 48 zero, and 16 negative outcomes. Positive-case count did not capture the cost of harmful actions.

The tree and disagreement selected similar counts of candidate-empty queries (39 versus 40), so that single diagnostic did not explain the tree's deficit. The fitted tree used eight distinct leaf predictions and did not split on Jaccard overlap. Coarse ties followed the fixed query-ID rule. The experiments cannot distinguish whether richer features, a different objective, more data, or another inductive bias would help.

The [error analysis](routing_error_analysis.md) includes each model's 20 highest predictions, harmful selected cases, high-gain missed cases, and learned-only versus heuristic-only selections. No additional feature/model search followed these development results.

## CPU feasibility and scope

The [live CPU benchmark](cpu_benchmark.md) measures Never, Always, Disagreement, and the tree with the same development workload, two rounds, reversed policy order, resident assets, and actual retrieval/reranking. Component percentiles, observed batch completion latencies, active-service times, batch wall times, and quality are reported separately. Raw query vectors, candidates, and reranking scores are not reused.

Exact-budget allocation is a batch operation. Completion latency includes batch barriers and waiting; active-service time excludes waiting. The measurements answer a bounded CPU batch-workload question, not production or interactive online P95 performance. Loading, warmup, and offline document embedding are excluded from steady-state measurements and described separately.

| Policy | Mean batch seconds | Completion P50 / P95 seconds | Active-service P50 / P95 milliseconds |
|---|---:|---|---|
| Never | 18.08 | 0.038 / 0.130 | 37.60 / 130.24 |
| Always | 270.18 | 128.267 / 237.608 | 767.70 / 1340.84 |
| Disagreement | 90.89 | 11.807 / 64.979 | 50.29 / 920.97 |
| Regression Tree | 87.44 | 14.070 / 58.395 | 59.23 / 810.41 |

Disagreement reduced mean batch time by 66.36% versus Always, with 70.06% fewer calls and a 0.005206 lower development NDCG. This is a measured quality/compute tradeoff. The tree's small timing advantage over disagreement is confounded by selected-query mix and observed run-to-run noise; it does not reverse the failed learned-quality comparison. Always follows the two-pass schedule, whereas Never can finish queries immediately; completion percentiles must be interpreted under those schedules.

All 2,592 query executions reproduced the original candidate pools and quality scores. Two desktop rounds do not establish timing guarantees. A reproduction audit overlapped part of the first Always trial; all trials were retained and raw per-round times are provided. The second round followed completion of that audit.

## Closure decision

The study established selective-reranking opportunity and a useful simple baseline. It did not establish that the trained routers add value over that baseline. Further model search would consume additional time and increase development-set selection bias. We completed the benchmark and reproducibility work, preserved the negative findings, and ended the project rather than extending its scope.

Test judgments remain sealed. Because no learned policy satisfied the predefined development criterion and no final generalization claim is being made, this report intentionally stops at development evidence. It does not claim that learned routing is impossible, that the heuristic generalizes across datasets, or that a batch speedup proves online latency improvement.

## Reproduction and evidence

The README contains the complete download, audit, model download, retrieval, reranking, routing, validation, and CPU commands. Configuration, dependency lock, model revisions, fingerprints, fold records, CSV summaries, and validation results are versioned. Raw datasets, weights, fitted model binaries, checkpoints, and Parquet outputs are regenerated locally and are not redistributed.

See [gain feasibility](feasibility_results.md), [routing results](routing_results.md), [routing error analysis](routing_error_analysis.md), and [CPU benchmark](cpu_benchmark.md) for the underlying results and caveats.

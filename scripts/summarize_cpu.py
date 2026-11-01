"""Validate measured CPU outputs and create a benchmark report and figure."""
import json

from experiment import ROOT, OUT, digest, save_json

import numpy as np
import pandas as pd


def main():
    frame = pd.read_parquet(OUT / "cpu_benchmark_queries.parquet")
    batches = pd.read_csv(ROOT / "reports/cpu_benchmark_batches.csv")
    summary = pd.read_csv(ROOT / "reports/cpu_benchmark_summary.csv")
    meta = json.loads((OUT / "cpu_benchmark_metadata.json").read_text())
    assert meta["script_sha256"] == digest(ROOT / "scripts/benchmark_cpu.py")
    assert len(frame) == 2 * 4 * 324 and len(batches) == 8
    expected = pd.read_parquet(OUT / "retrieval_results.parquet")
    expected = {qid: values.dropna(subset=["candidate_rank"]).sort_values("candidate_rank").document_id.tolist() for qid, values in expected.loc[expected.split == "dev"].groupby("query_id")}
    mismatch = 0
    for qid, values in frame.groupby("query_id"):
        pools = [list(x) for x in values.candidate_ids]
        assert all(x == pools[0] for x in pools)
        mismatch += int(pools[0] != expected[qid])
        for row in values.itertuples():
            assert set(row.ranking) == set(row.candidate_ids)
            assert row.active_service_seconds > 0
            assert row.completion_latency_seconds >= row.cross_encoder_seconds
    for (round_index, policy), values in frame.groupby(["round", "policy"]):
        assert len(values) == 324 and values.query_id.is_unique
        assert values.selected.sum() == {"never": 0, "always": 324, "disagreement": 97, "regression_tree": 97}[policy]
    gains = pd.read_parquet(OUT / "query_gains.parquet")
    dev_results = pd.read_csv(ROOT / "reports/routing_dev_results.csv")
    quality_checks = {}
    for row in summary.itertuples():
        if row.policy == "never":
            expected_quality = gains.loc[gains.split == "dev", "base_ndcg"].mean()
        elif row.policy == "always":
            expected_quality = gains.loc[gains.split == "dev", "reranked_ndcg"].mean()
        else:
            expected_quality = dev_results.loc[(dev_results.policy == row.policy) & (dev_results.budget == 0.3), "ndcg"].iloc[0]
        quality_checks[row.policy] = {"measured": row.ndcg, "offline": float(expected_quality), "difference": float(row.ndcg - expected_quality)}
        assert np.isclose(row.ndcg, expected_quality, atol=1e-10, rtol=0)
    save_json(ROOT / "reports/cpu_benchmark_validation.json", {"passed": True, "measured_query_executions": len(frame), "rounds": 2, "policies": 4, "live_candidate_pools_differing_from_offline": mismatch, "quality": quality_checks, "checks": ["324 unique queries per policy and round", "identical live candidate pools across policies/rounds", "reranking preserves candidate membership", "exact live rerank-call counts", "live quality matches offline quality"]})
    lines = ["# CPU Benchmark", "", "Measured on the 324 development queries with resident models, a resident BM25 index, and precomputed document embeddings. Every policy recomputes query embeddings, lexical retrieval, dense similarity, and RRF. No query results or cross-encoder scores are reused.", "", "Two rounds use identical query order (seed 42) and reversed policy order. Eight training queries warm the pipeline before measurement. PyTorch uses CPU FP32, four intra-op and one inter-op thread, pair length 256, and batch size 8 within a query. Query embedding uses batch size 1. There is no cross-query model batching.", "", f"Machine: {meta['processor']}; {meta['platform']}. Asset loading and index setup took {meta['startup_seconds']:.2f} seconds and are excluded from steady-state measurements. Offline corpus embedding is a separate preparation cost.", "", "## End-to-end measurements", "", "Exact 30% allocation requires all query scores, so selective policies use a two-pass batch: retrieve the full workload, allocate 97 calls by score/query ID, then rerank selected queries. Always uses the same two-pass schedule; Never can complete each query immediately. Completion latency is observed directly from a query's first-stage start until its ranking is final, including barrier and queue waiting. This is a batch workload, not interactive online request latency.", "", "Active-service time sums measured first-stage and reranking work plus amortized batch-allocation time. It excludes waiting and is reported as a compute diagnostic, not as directly measured online latency. Batch wall time includes the full timed workload. Percentiles pool 648 executions per policy across two rounds; two rounds do not establish statistical performance guarantees.", "", "| Policy | NDCG@10 | Calls / 324 | Mean batch seconds | Completion P50 ms | Completion P95 ms | Active P50 ms | Active P95 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        calls = {"never": 0, "always": 324, "disagreement": 97, "regression_tree": 97}[row.policy]
        lines.append(f"| {row.policy} | {row.ndcg:.6f} | {calls} | {row.mean_batch_seconds:.2f} | {row.completion_latency_p50_ms:.2f} | {row.completion_latency_p95_ms:.2f} | {row.active_service_p50_ms:.2f} | {row.active_service_p95_ms:.2f} |")
    always = summary.loc[summary.policy == "always"].iloc[0]
    lines.extend(["", "## Observed batch savings", ""])
    for row in summary.loc[summary.policy.isin(["disagreement", "regression_tree"])].itertuples():
        lines.append(f"- {row.policy}: {(1 - row.mean_batch_seconds / always.mean_batch_seconds):.2%} lower mean batch wall time than Always, with {1 - 97/324:.2%} fewer reranking calls. Quality differs because query selection differs.")
    lines.extend(["", "The policy-order reversal reduces one order effect but cannot remove thermal throttling, CPU frequency changes, background activity, or OS scheduling noise. A data-audit reproduction check overlapped part of the first Always trial; the second round followed its completion. No trial was discarded. These are descriptive desktop measurements, not an isolated-system microbenchmark. Savings describe this small corpus and this batch implementation. They do not establish online P95 improvements or cross-hardware generalization.", "", "Component P50/P95 values, including query embedding, BM25, dense search, RRF, routing, and cross-encoder execution, are in cpu_benchmark_summary.csv. Component cross-encoder times include zero for skipped queries, so selective-policy P50 can be zero. Raw per-query records retain per-call timings. Raw per-round batch times are in cpu_benchmark_batches.csv. Component percentiles must not be added to obtain completion percentiles.", "", f"Validation checked all {len(frame):,} query executions. Live candidate pools differing from the original quality experiment: {mismatch}. Live mean NDCG matches the offline experiment for each policy. Test judgments were not read.", "", "![Quality and CPU batch time](cpu_quality_latency.png)"])
    (ROOT / "reports/cpu_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in summary.itertuples():
        ax.scatter(row.mean_batch_seconds, row.ndcg, s=60)
        ax.annotate(row.policy, (row.mean_batch_seconds, row.ndcg), xytext=(5, 7), textcoords="offset points")
    ax.set(xlabel="Mean wall time for 324-query CPU batch (seconds)", ylabel="Mean binary NDCG@10", title="Quality versus measured batch execution time")
    ax.margins(x=0.2, y=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/cpu_quality_latency.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()

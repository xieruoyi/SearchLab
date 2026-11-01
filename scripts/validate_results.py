"""Validate persisted outputs and summarize distinct zero-gain cases."""
import json

import numpy as np
import pandas as pd

from experiment import ROOT, OUT, load_data, save_json


def main():
    cfg, _, _, queries, positives = load_data()
    gains = pd.read_parquet(OUT / "query_gains.parquet")
    retrieval = pd.read_parquet(OUT / "retrieval_results.parquet")
    curves = pd.read_csv(ROOT / "reports/budget_curves.csv")
    assert set(gains.query_id) == {q["query_id"] for q in queries}
    assert gains.query_id.is_unique
    assert np.allclose(gains.gain, gains.reranked_ndcg - gains.base_ndcg, atol=1e-15, rtol=0)
    assert gains[["base_ndcg", "reranked_ndcg"]].ge(0).all().all()
    assert gains[["base_ndcg", "reranked_ndcg"]].le(1 + 1e-12).all().all()
    pools = retrieval.dropna(subset=["candidate_rank"]).groupby("query_id")
    for row in gains.itertuples():
        pool = pools.get_group(row.query_id).sort_values("candidate_rank").document_id.tolist()
        assert pool == list(row.candidate_ids)
        assert len(set(row.reranked_ids)) == 20
        assert set(row.reranked_ids) == set(pool)
        assert len(set(pool) & positives[row.query_id]) == row.candidate_positive_count
    diagnostics = {}
    for split, count in (("train", 2584), ("dev", 324)):
        frame = gains.loc[gains.split == split]
        assert len(frame) == count
        rows = curves.loc[curves.split == split]
        for fraction, budget in rows.groupby("budget"):
            fixed = budget.loc[budget.policy != "oracle_at_most"]
            assert (fixed.selected == int(np.floor(fraction * count))).all()
            oracle = budget.loc[budget.policy == "oracle", "ndcg"].iloc[0]
            assert (fixed.ndcg <= oracle + 1e-12).all()
            assert budget.loc[budget.policy == "oracle_at_most", "ndcg"].iloc[0] >= oracle - 1e-12
            if fraction == 0:
                assert np.allclose(budget.ndcg, frame.base_ndcg.mean(), atol=1e-12)
            if fraction == 1:
                assert np.allclose(fixed.ndcg, frame.reranked_ndcg.mean(), atol=1e-12)
        zero = frame.gain.abs() <= cfg["evaluation"]["gain_epsilon"]
        no_candidates = frame.candidate_positive_count == 0
        perfect_base = frame.base_ndcg >= 1 - 1e-12
        diagnostics[split] = {"zero_gain_queries": int(zero.sum()), "zero_without_candidate_positive": int((zero & no_candidates).sum()), "zero_with_perfect_base": int((zero & perfect_base).sum()), "other_zero_gain_queries": int((zero & ~no_candidates & ~perfect_base).sum())}
    save_json(ROOT / "reports/result_validation.json", {"passed": True, "checks": ["all eligible query IDs present exactly once", "persisted candidates match retrieval output", "reranking preserves 20 unique candidates", "gain equals reranked minus base NDCG", "fixed budgets use floor", "oracle dominates same-budget policies", "zero/full budget endpoints match never/always"], "zero_gain_diagnostics": diagnostics})
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

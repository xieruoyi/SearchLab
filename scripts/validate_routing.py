"""Replay stored routing decisions and check baseline and fold integrity."""
import json
import pickle

import numpy as np
import pandas as pd

from experiment import ROOT, OUT, CONFIG, digest, save_json
from train_routing import policy_changes


def main():
    meta = json.loads((OUT / "routing_metadata.json").read_text())
    assert meta["script_sha256"] == digest(ROOT / "scripts/train_routing.py")
    assert meta["features_sha256"] == digest(OUT / "routing_features.parquet")
    names = json.loads(CONFIG.read_text())["routing"]["features"]
    features = pd.read_parquet(OUT / "routing_features.parquet")
    predictions = pd.read_parquet(OUT / "routing_predictions.parquet")
    assert set(features.columns) == set(names) | {"query_id", "split", "group_id"}
    assert np.isfinite(features[names].to_numpy()).all()
    assert predictions.query_id.is_unique and len(predictions) == 2908
    oof = predictions.loc[predictions.split == "train_oof"]
    assert len(oof) == 2584 and (oof.groupby("group_id").fold.nunique() == 1).all()
    assert not set(oof.group_id) & set(predictions.loc[predictions.split == "dev", "group_id"])
    for fold in ("1", "2", "3"):
        held_out = set(oof.loc[oof.fold == fold, "group_id"])
        fit = set(oof.loc[oof.fold != fold, "group_id"])
        assert held_out and fit and not held_out & fit
    dev = features.loc[features.split == "dev"].sort_values("query_id").reset_index(drop=True)
    saved = predictions.loc[predictions.split == "dev"].sort_values("query_id").reset_index(drop=True)
    assert dev.query_id.tolist() == saved.query_id.tolist()
    gains = pd.read_parquet(OUT / "query_gains.parquet")
    dev = dev.merge(gains[["query_id", "gain", "base_ndcg"]], on="query_id", validate="one_to_one")
    results = pd.read_csv(ROOT / "reports/routing_dev_results.csv")
    for name in ("logistic", "regression_tree"):
        path = OUT / "routing_models" / (name + ".pkl")
        assert digest(path) == meta["model_sha256"][path.name]
        with path.open("rb") as source:
            model = pickle.load(source)  # Own locally generated artifact, verified above.
        scores = model.predict_proba(dev[names])[:, 1] if name == "logistic" else model.predict(dev[names])
        assert np.allclose(scores, saved[name], atol=1e-14, rtol=0)
        for row in results.loc[results.policy == name].itertuples():
            changes, chosen = policy_changes(dev, scores, row.budget)
            assert len(chosen) == row.selected
            assert np.isclose(dev.base_ndcg.mean() + changes.mean(), row.ndcg, atol=1e-14, rtol=0)
    prior = pd.read_csv(ROOT / "reports/budget_curves.csv")
    for row in results.loc[results.policy.isin(["random", "score_margin", "disagreement", "oracle"])].itertuples():
        expected = prior.loc[(prior.split == "dev") & (prior.policy == row.policy) & (prior.budget == row.budget)].iloc[0]
        assert np.isclose(expected.ndcg, row.ndcg, atol=1e-14, rtol=0)
        assert expected.selected == row.selected
    save_json(ROOT / "reports/routing_validation.json", {"passed": True, "checks": ["exact pre-rerank feature allowlist", "finite features", "one held-out prediction per training query", "duplicate groups stay in one fold", "train/dev groups disjoint", "stored model and feature fingerprints", "reloaded models reproduce all dev scores", "all dev budget decisions reproduce reported NDCG", "random and heuristic baselines match the feasibility run"]})
    print("Routing integrity validation passed")


if __name__ == "__main__":
    main()

"""Fixed-model grouped CV, development evaluation, and routing error analysis."""
import json
import math
import pickle
from pathlib import Path

from experiment import ROOT, OUT, DATA, CONFIG, digest, save_json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text


def build_features(retrieval, queries, names):
    # Deliberately accepts no judgments, gains, or cross-encoder outputs.
    rows = []
    for qid, frame in retrieval.groupby("query_id", sort=True):
        values = {"query_id": qid, "split": frame.split.iloc[0], "group_id": frame.group_id.iloc[0], "query_word_count": len(queries[qid].split()), "query_character_count": len(queries[qid]), "rrf_top1_top2_margin": float(frame.rrf_margin.iloc[0]), "bm25_dense_top20_jaccard": float(frame.retrieval_jaccard.iloc[0])}
        for retriever in ("bm25", "dense"):
            top = frame.dropna(subset=[retriever + "_rank"]).sort_values(retriever + "_rank").head(20)
            assert len(top) == 20
            scores = top[retriever + "_score"].to_numpy()
            values.update({retriever + "_top1": scores[0], retriever + "_top1_top2_margin": scores[0] - scores[1], retriever + "_top20_mean": scores.mean(), retriever + "_top20_std": scores.std(ddof=0)})
        rows.append(values)
    features = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    assert set(features.columns) == set(names) | {"query_id", "split", "group_id"}
    assert np.isfinite(features[names].to_numpy()).all()
    return features


def policy_changes(frame, scores, fraction):
    # Caller sorts by query ID first; stable sort implements the fixed tie break.
    k = math.floor(fraction * len(frame))
    chosen = np.argsort(-np.asarray(scores), kind="stable")[:k]
    changes = np.zeros(len(frame))
    changes[chosen] = frame.gain.to_numpy()[chosen]
    return changes, chosen


def evaluate(frame, predictions, budgets, seeds, split, fold):
    policies = {**predictions, "score_margin": -frame.rrf_top1_top2_margin.to_numpy(), "disagreement": -frame.bm25_dense_top20_jaccard.to_numpy(), "oracle": frame.gain.to_numpy()}
    rows, outcomes = [], {}
    for budget in budgets:
        for policy, scores in policies.items():
            changes, chosen = policy_changes(frame, scores, budget)
            outcomes[(policy, budget)] = changes
            rows.append({"split": split, "fold": fold, "policy": policy, "budget": budget, "queries": len(frame), "selected": len(chosen), "actual_fraction": len(chosen) / len(frame), "ndcg": float(frame.base_ndcg.mean() + changes.mean()), "gain_over_base": float(changes.mean()), "selected_negative_count": int((frame.gain.to_numpy()[chosen] < -1e-6).sum())})
        random_changes = []
        for seed in seeds:
            chosen = np.random.default_rng(seed).permutation(len(frame))[:math.floor(budget * len(frame))]
            changes = np.zeros(len(frame))
            changes[chosen] = frame.gain.to_numpy()[chosen]
            random_changes.append(changes)
        changes = np.mean(random_changes, axis=0)
        outcomes[("random", budget)] = changes
        rows.append({"split": split, "fold": fold, "policy": "random", "budget": budget, "queries": len(frame), "selected": math.floor(budget * len(frame)), "actual_fraction": math.floor(budget * len(frame)) / len(frame), "ndcg": float(frame.base_ndcg.mean() + changes.mean()), "gain_over_base": float(changes.mean()), "seed_ndcg_std": float(np.std([x.mean() for x in random_changes], ddof=1))})
        for policy in (*policies, "random"):
            assert outcomes[(policy, budget)].mean() <= outcomes[("oracle", budget)].mean() + 1e-12
            if budget == 0:
                assert np.all(outcomes[(policy, budget)] == 0)
            if budget == 1:
                assert np.allclose(outcomes[(policy, budget)], frame.gain.to_numpy())
    return rows, outcomes


def bootstrap(frame, outcomes, scope):
    groups = [np.flatnonzero(frame.group_id.to_numpy() == g) for g in sorted(frame.group_id.unique())]
    generator = np.random.default_rng(42)
    values = {name: [] for name in ("logistic", "regression_tree")}
    for _ in range(1000):
        indices = np.concatenate([groups[i] for i in generator.integers(len(groups), size=len(groups))])
        for name in values:
            diff = outcomes[(name, 0.3)] - outcomes[("disagreement", 0.3)]
            values[name].append(diff[indices].mean())
    return [{"scope": scope, "policy": name, "reference": "disagreement", "budget": 0.3, "difference": float((outcomes[(name, 0.3)] - outcomes[("disagreement", 0.3)]).mean()), "lower_95": float(np.percentile(draws, 2.5)), "upper_95": float(np.percentile(draws, 97.5))} for name, draws in values.items()]


def main():
    cfg = json.loads(CONFIG.read_text())
    names = cfg["routing"]["features"]
    # Freeze this protocol on disk before loading the target gains.
    protocol = {"models": cfg["routing"], "features": names, "cv": "3-fold GroupKFold, no shuffle", "standard_deviation": "population (ddof=0), each retriever's own top 20", "word_count": "whitespace split", "primary_comparison": "30% budget, learned minus disagreement", "classification_target": "gain > 1e-6", "fixed_hyperparameters": True, "dev_used_for_training": False, "bootstrap": "1000 paired duplicate-group resamples of fixed policy outcomes; no refitting or budget reselection", "tie_break": "ascending query ID", "budgets": cfg["evaluation"]["budgets"]}
    protocol_path = ROOT / "configs/routing_protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, "Existing routing protocol differs"
    else:
        save_json(protocol_path, protocol)
    retrieval = pd.read_parquet(OUT / "retrieval_results.parquet")
    rmeta = json.loads((OUT / "retrieval_metadata.json").read_text())
    assert digest(OUT / "retrieval_results.parquet") == rmeta["output_sha256"]
    queries = {}
    for split in ("train", "dev"):
        queries.update({q["_id"]: q["text"] for q in map(json.loads, (DATA / f"{split}_queries.jsonl").read_text(encoding="utf-8").splitlines())})
    features = build_features(retrieval, queries, names)
    features.to_parquet(OUT / "routing_features.parquet", index=False)
    gains = pd.read_parquet(OUT / "query_gains.parquet")
    assert digest(OUT / "query_gains.parquet") == json.loads((OUT / "gain_metadata.json").read_text())["output_sha256"]
    frame = features.merge(gains[["query_id", "split", "group_id", "gain", "base_ndcg", "reranked_ndcg", "candidate_positive_count"]], on=["query_id", "split", "group_id"], validate="one_to_one")
    train = frame.loc[frame.split == "train"].sort_values("query_id").reset_index(drop=True)
    dev = frame.loc[frame.split == "dev"].sort_values("query_id").reset_index(drop=True)
    assert len(train) == 2584 and len(dev) == 324
    assert not set(train.group_id) & set(dev.group_id)
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42)), "regression_tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)}
    cv_rows, dev_rows, intervals, prediction_rows, fold_records = [], [], [], [], []
    oof = {name: np.full(len(train), np.nan) for name in models}
    budgets, seeds = cfg["evaluation"]["budgets"], cfg["evaluation"]["random_seeds"]
    for fold, (fit, valid) in enumerate(GroupKFold(n_splits=3).split(train[names], groups=train.group_id), start=1):
        assert not set(train.iloc[fit].group_id) & set(train.iloc[valid].group_id)
        validation = train.iloc[valid].reset_index(drop=True)
        predictions = {}
        for name, template in models.items():
            model = clone(template)
            target = (train.iloc[fit].gain.to_numpy() > 1e-6).astype(int) if name == "logistic" else train.iloc[fit].gain.to_numpy()
            model.fit(train.iloc[fit][names], target)
            predicted = model.predict_proba(validation[names])[:, 1] if name == "logistic" else model.predict(validation[names])
            assert np.isfinite(predicted).all()
            predictions[name] = predicted
            oof[name][valid] = predicted
        rows, outcomes = evaluate(validation, predictions, budgets, seeds, "train_validation", fold)
        cv_rows.extend(rows)
        intervals.extend(bootstrap(validation, outcomes, f"cv_fold_{fold}"))
        fold_records.append({"fold": fold, "fit_queries": len(fit), "validation_queries": len(valid), "fit_groups": train.iloc[fit].group_id.nunique(), "validation_groups": validation.group_id.nunique(), "group_overlap": 0})
        for index in valid:
            prediction_rows.append({"query_id": train.iloc[index].query_id, "split": "train_oof", "group_id": train.iloc[index].group_id, "fold": fold, **{name: oof[name][index] for name in models}})
        print(f"Completed grouped fold {fold}", flush=True)
    assert all(np.isfinite(x).all() for x in oof.values())
    final_predictions = {}
    model_dir = OUT / "routing_models"
    model_dir.mkdir(exist_ok=True)
    coefficients = []
    for name, template in models.items():
        model = clone(template)
        target = (train.gain.to_numpy() > 1e-6).astype(int) if name == "logistic" else train.gain.to_numpy()
        model.fit(train[names], target)
        final_predictions[name] = model.predict_proba(dev[names])[:, 1] if name == "logistic" else model.predict(dev[names])
        with (model_dir / f"{name}.pkl").open("wb") as output:
            pickle.dump(model, output)
        if name == "logistic":
            scaler = model.named_steps["standardscaler"]
            assert np.allclose(scaler.mean_, train[names].mean().to_numpy())
            importance = model.named_steps["logisticregression"].coef_[0]
        else:
            importance = model.feature_importances_
            (ROOT / "reports/routing_tree.txt").write_text(export_text(model, feature_names=names), encoding="utf-8")
        coefficients.extend({"model": name, "feature": feature, "value": float(value)} for feature, value in zip(names, importance))
    rows, outcomes = evaluate(dev, final_predictions, budgets, seeds, "dev", "full_train_fit")
    dev_rows.extend(rows)
    intervals.extend(bootstrap(dev, outcomes, "dev"))
    for i, row in dev.iterrows():
        prediction_rows.append({"query_id": row.query_id, "split": "dev", "group_id": row.group_id, "fold": "full_train_fit", **{name: final_predictions[name][i] for name in models}})
    cv = pd.DataFrame(cv_rows)
    results = pd.DataFrame(dev_rows)
    ci = pd.DataFrame(intervals)
    cv.to_csv(ROOT / "reports/routing_cv_results.csv", index=False)
    results.to_csv(ROOT / "reports/routing_dev_results.csv", index=False)
    ci.to_csv(ROOT / "reports/routing_intervals.csv", index=False)
    predictions_frame = pd.DataFrame(prediction_rows)
    predictions_frame["fold"] = predictions_frame["fold"].astype(str)
    predictions_frame.to_parquet(OUT / "routing_predictions.parquet", index=False)
    pd.DataFrame(fold_records).to_csv(ROOT / "reports/routing_fold_manifest.csv", index=False)
    pd.DataFrame(coefficients).to_csv(ROOT / "reports/routing_feature_weights.csv", index=False)
    # Preserve all per-query outcomes for paired analyses and error review.
    decisions = []
    for name, scores in final_predictions.items():
        _, chosen = policy_changes(dev, scores, 0.3)
        _, baseline_chosen = policy_changes(dev, -dev.bm25_dense_top20_jaccard.to_numpy(), 0.3)
        for i, row in dev.iterrows():
            decisions.append({"query_id": row.query_id, "model": name, "predicted_score": scores[i], "selected_at_30": i in chosen, "disagreement_selected_at_30": i in baseline_chosen, "actual_gain": row.gain, "base_ndcg": row.base_ndcg, "candidate_positive_count": row.candidate_positive_count})
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(ROOT / "reports/routing_dev_decisions.csv", index=False)
    error_report(dev, decisions, queries)
    plot_curves(cv, results)
    report(cv, results, ci)
    save_json(OUT / "routing_metadata.json", {"script_sha256": digest(__file__), "protocol_sha256": digest(protocol_path), "retrieval_sha256": digest(OUT / "retrieval_results.parquet"), "gains_sha256": digest(OUT / "query_gains.parquet"), "features_sha256": digest(OUT / "routing_features.parquet"), "feature_columns": names, "train_queries": len(train), "dev_queries": len(dev), "folds": fold_records, "test_accessed": False, "checks": ["train/dev groups disjoint", "fold fit/validation groups disjoint", "OOF prediction present for every training query", "training-only scaler mean verified", "finite pre-rerank features", "exact frozen feature allowlist"], "model_sha256": {p.name: digest(p) for p in model_dir.glob("*.pkl")}})


def error_report(dev, decisions, queries):
    lines = ["# Routing Error Analysis", "", "Development analysis at a fixed 30% budget (97/324 queries). Logistic scores estimate positive-gain probability; tree scores estimate continuous gain. They are not interchangeable scales. Candidate coverage and actual gains below are post-hoc diagnostics, never model inputs."]
    baseline = decisions.loc[decisions.model == "logistic"]
    baseline = baseline.loc[baseline.disagreement_selected_at_30]
    lines.extend(["", "## Selection outcomes", "", "| Policy | Positive | Zero | Negative | No judged positive candidate |", "|---|---:|---:|---:|---:|", f"| disagreement | {(baseline.actual_gain > 1e-6).sum()} | {(baseline.actual_gain.abs() <= 1e-6).sum()} | {(baseline.actual_gain < -1e-6).sum()} | {(baseline.candidate_positive_count == 0).sum()} |"])
    for name, values in decisions.groupby("model"):
        selected = values.loc[values.selected_at_30]
        lines.append(f"| {name} | {(selected.actual_gain > 1e-6).sum()} | {(selected.actual_gain.abs() <= 1e-6).sum()} | {(selected.actual_gain < -1e-6).sum()} | {(selected.candidate_positive_count == 0).sum()} |")
    lines.extend(["", "Logistic regression selects more positive-gain queries than disagreement, but also many more negative-gain queries. Positive-query count alone does not measure the quality of budget allocation. The tree selects a similar number of candidate-empty queries to disagreement; that count alone does not explain its deficit. These observations describe outcomes, not established causes of model failure.", "", "The fixed regression tree gives zero split importance to retrieval Jaccard overlap in its full-training fit, while the disagreement baseline ranks directly by it. This is a limitation of this fitted shallow tree and objective; it is not evidence that overlap lacks useful information. The available experiment does not distinguish insufficient features, limited sample size, or unsuitable inductive bias."])
    for name, frame in decisions.groupby("model"):
        lines.extend(["", f"## {name}"])
        selected = frame.loc[frame.selected_at_30]
        lines.append(f"\nSelected outcomes: {(selected.actual_gain > 1e-6).sum()} positive, {(selected.actual_gain.abs() <= 1e-6).sum()} zero, {(selected.actual_gain < -1e-6).sum()} negative.")
        boundary = selected.predicted_score.min()
        tied = frame.predicted_score == boundary
        learned_only = frame.loc[frame.selected_at_30 & ~frame.disagreement_selected_at_30]
        heuristic_only = frame.loc[~frame.selected_at_30 & frame.disagreement_selected_at_30]
        lines.append(f"\nUnique predicted scores: {frame.predicted_score.nunique()}. Budget-boundary score has {tied.sum()} tied queries, of which {(tied & frame.selected_at_30).sum()} are selected by query-ID order. Learned-only selections: {len(learned_only)}, total realized gain {learned_only.actual_gain.sum():+.6f}; disagreement-only selections: {len(heuristic_only)}, total realized gain {heuristic_only.actual_gain.sum():+.6f}. Divide the gain-sum difference by 324 for the policy NDCG difference.")
        cases = {"Highest predicted scores (20 queries)": frame.sort_values(["predicted_score", "query_id"], ascending=[False, True]).head(20), "Most harmful selected queries": selected.sort_values(["actual_gain", "query_id"]).head(20), "Largest missed gains": frame.loc[~frame.selected_at_30].sort_values(["actual_gain", "query_id"], ascending=[False, True]).head(20), "Learned-only selections": frame.loc[frame.selected_at_30 & ~frame.disagreement_selected_at_30].sort_values(["actual_gain", "query_id"]), "Disagreement-only selections": frame.loc[~frame.selected_at_30 & frame.disagreement_selected_at_30].sort_values(["actual_gain", "query_id"], ascending=[False, True])}
        for title, subset in cases.items():
            lines.extend(["", f"### {title}", "", "| Query ID | Query | Predicted score | Actual gain | Base NDCG | Judged positives in pool |", "|---|---|---:|---:|---:|---:|"])
            for row in subset.itertuples():
                text = queries[row.query_id].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {row.query_id} | {text} | {row.predicted_score:.6f} | {row.actual_gain:+.6f} | {row.base_ndcg:.6f} | {row.candidate_positive_count} |")
    (ROOT / "reports/routing_error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(cv, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, frame, title in ((axes[0], cv, "Grouped CV: mean of held-out folds"), (axes[1], dev, "Development")):
        for policy, values in frame.groupby("policy"):
            data = values.groupby("budget").ndcg.mean()
            ax.plot(data.index, data.values, marker="o", label=policy)
        ax.set(title=title, xlabel="Reranking call fraction", ylabel="Mean binary NDCG@10")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/routing_budget_curves.png", dpi=160)
    plt.close(fig)


def report(cv, dev, intervals):
    rows = dev.loc[dev.budget == 0.3]
    lines = ["# Learned Routing Results", "", "Fixed models and features; three-fold grouped training validation followed by full-training fit and development evaluation. No hyperparameter search or test-set access.", "", "## Development at 30% budget", "", "| Policy | NDCG@10 | Gain over base | Difference vs disagreement |", "|---|---:|---:|---:|"]
    baseline = rows.loc[rows.policy == "disagreement", "ndcg"].iloc[0]
    for row in rows.itertuples():
        lines.append(f"| {row.policy} | {row.ndcg:.6f} | {row.gain_over_base:+.6f} | {row.ndcg - baseline:+.6f} |")
    lines.extend(["", "## Primary comparison: learned minus disagreement", "", "| Scope | Model | Difference at 30% | Conditional 95% interval |", "|---|---|---:|---|"])
    for row in intervals.itertuples():
        lines.append(f"| {row.scope} | {row.policy} | {row.difference:+.6f} | [{row.lower_95:+.6f}, {row.upper_95:+.6f}] |")
    lines.extend(["", "## Assessment", "", "The predefined learned-predictability criterion is not met. At the primary 30% budget, both learned models have lower development point estimates than disagreement, and neither paired interval establishes an improvement. Logistic regression trails disagreement in all three held-out training folds; the regression tree wins one fold and loses two. The regression tree performs better than logistic regression on development, but this does not establish an advantage over the main heuristic.", "", "The result supports retaining disagreement as the current routing baseline. Further feature or objective changes should be treated as a new exploratory experiment with a separately frozen protocol. Learned routing and live end-to-end speedups are not validated by this run."])
    lines.extend(["", "Intervals use 1,000 paired duplicate-group resamples of fixed held-out outcomes. Models are not refitted and budget membership is not reselected within replicates. They condition on the fitted models and selected policies, and do not quantify training instability. Intervals are unadjusted for multiple comparisons. CV budgets are allocated separately within each held-out fold. The plot averages fold means; fold-level results remain in the CSV.", "", "Random uses the original ten seeded query permutations, matching the feasibility experiment. Learned tie scores use ascending query ID. Tree leaves produce tied predictions, so budget-boundary selections follow this predefined rule.", "", "The development set is repeatedly used development data, not a final unbiased test. Live end-to-end CPU latency remains unmeasured.", "", "![Budget curves](routing_budget_curves.png)", "", "See routing_error_analysis.md for top predictions, harmful selections, missed gains, and selection differences from disagreement."])
    (ROOT / "reports/routing_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]), flush=True)


if __name__ == "__main__":
    main()

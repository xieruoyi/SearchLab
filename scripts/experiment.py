"""Fixed train/dev retrieval, checkpointed reranking, and gain analysis.

Run with the project virtual environment. Never reads the raw archive or test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import time
import unicodedata
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
DATA = ROOT / "data/processed"
OUT = ROOT / "experiments"
CONFIG = ROOT / "configs/experiment.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def tokens(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def ndcg(ranking, positives):
    if not positives:
        raise ValueError("Query has no positive judgments")
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate document in ranking")
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranking[:10]) if d in positives)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(positives))))
    return dcg / ideal


def ranked(scores, ids, k):
    import numpy as np
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    # ids are pre-sorted: stable sorting implements the document-ID tie break.
    return np.argsort(-np.asarray(scores), kind="stable")[:k].tolist()


def load_data():
    import pandas as pd
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    corpus = [json.loads(x) for x in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus.sort(key=lambda x: x["_id"])
    ids = [x["_id"] for x in corpus]
    texts = [x.get("title", "") + "\n" + x.get("text", "") for x in corpus]
    manifest = pd.read_csv(DATA / "split_manifest.csv", dtype=str)
    records, positives = [], {}
    for split in ("train", "dev"):
        qs = {x["_id"]: x["text"] for x in map(json.loads, (DATA / f"{split}_queries.jsonl").read_text(encoding="utf-8").splitlines())}
        qrels = pd.read_csv(DATA / f"{split}_qrels.tsv", sep="\t", dtype={"query-id": str, "corpus-id": str})
        for q, rows in qrels.groupby("query-id"):
            positives[q] = set(rows.loc[rows.score > 0, "corpus-id"])
        for row in manifest.loc[manifest.official_split == split].sort_values("query_id").itertuples():
            if row.routing_role == "excluded_train_overlap_dev":
                continue
            text = qs[row.query_id]
            assert hashlib.sha256(text.encode()).hexdigest() == row.query_text_sha256
            records.append({"query_id": row.query_id, "split": split, "group_id": row.group_id, "text": text})
    assert len(ids) == 3633
    assert sum(x["split"] == "train" for x in records) == 2584
    assert sum(x["split"] == "dev" for x in records) == 324
    return cfg, ids, texts, records, positives


def setup_torch(cfg):
    import torch
    torch.set_num_threads(cfg["reranker"]["torch_num_threads"])
    torch.set_num_interop_threads(cfg["reranker"]["torch_num_interop_threads"])
    torch.manual_seed(cfg["seed"])


def local_model(entry):
    path = ROOT / ".cache/models" / entry["model"].replace("/", "--") / entry["revision"]
    manifest = json.loads((path / "download_manifest.json").read_text())
    assert manifest["repo"] == entry["model"] and manifest["revision"] == entry["revision"]
    for name, expected in manifest["sha256"].items():
        assert digest(path / name) == expected, f"Model asset changed: {name}"
    return str(path)


def inputs_fingerprint():
    paths = [CONFIG, *sorted(DATA.glob("*"))]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def runtime_metadata():
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(), "packages": {name: version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy", "pandas", "pyarrow", "rank-bm25")}, "torch_threads": 4, "torch_interop_threads": 1, "cross_query_batching": False}


def retrieve():
    import numpy as np
    import pandas as pd
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    cfg, ids, texts, queries, _ = load_data()
    setup_torch(cfg)
    OUT.mkdir(exist_ok=True)
    fingerprint = inputs_fingerprint()
    start = time.perf_counter()
    dcfg = cfg["retrieval"]["dense"]
    model = SentenceTransformer(local_model(dcfg), device="cpu", local_files_only=True)
    model.max_seq_length = dcfg["max_sequence_length"]
    print("Encoding corpus", flush=True)
    docs = model.encode(texts, batch_size=dcfg["batch_size"], normalize_embeddings=True, show_progress_bar=True)
    print("Encoding train/dev queries", flush=True)
    qvecs = model.encode([q["text"] for q in queries], batch_size=dcfg["batch_size"], normalize_embeddings=True, show_progress_bar=True)
    np.save(OUT / "document_embeddings.npy", docs)
    bcfg = cfg["retrieval"]["bm25"]
    bm25 = BM25Okapi([tokens(t) for t in texts], k1=bcfg["k1"], b=bcfg["b"], epsilon=bcfg["epsilon"])
    rows = []
    for qi, query in enumerate(queries):
        bs = bm25.get_scores(tokens(query["text"]))
        ds = docs @ qvecs[qi]
        bi, di = ranked(bs, ids, 100), ranked(ds, ids, 100)
        br, dr = {x: i + 1 for i, x in enumerate(bi)}, {x: i + 1 for i, x in enumerate(di)}
        fused = {i: (1 / (60 + br[i]) if i in br else 0) + (1 / (60 + dr[i]) if i in dr else 0) for i in set(bi) | set(di)}
        pool = sorted(fused, key=lambda i: (-fused[i], ids[i]))[:20]
        cr = {x: i + 1 for i, x in enumerate(pool)}
        jaccard = len(set(bi[:20]) & set(di[:20])) / len(set(bi[:20]) | set(di[:20]))
        margin = fused[pool[0]] - fused[pool[1]]
        for i in sorted(fused):
            rows.append({"query_id": query["query_id"], "split": query["split"], "group_id": query["group_id"], "document_id": ids[i], "bm25_score": float(bs[i]), "dense_score": float(ds[i]), "bm25_rank": br.get(i), "dense_rank": dr.get(i), "rrf_score": fused[i], "candidate_rank": cr.get(i), "rrf_margin": margin, "retrieval_jaccard": jaccard})
        if (qi + 1) % 250 == 0:
            print(f"Retrieved {qi + 1}/{len(queries)}", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "retrieval_results.parquet", index=False)
    save_json(OUT / "retrieval_metadata.json", {"runtime": runtime_metadata(), "inputs": fingerprint, "output_sha256": digest(OUT / "retrieval_results.parquet"), "script_sha256": digest(__file__), "queries": len(queries), "elapsed_seconds": time.perf_counter() - start, "test_accessed": False, "query_order": [q["query_id"] for q in queries]})
    print("Retrieval artifacts saved", flush=True)


def rerank(smoke=False):
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cfg, ids, texts, queries, positives = load_data()
    setup_torch(cfg)
    metadata = json.loads((OUT / "retrieval_metadata.json").read_text())
    assert metadata["inputs"] == inputs_fingerprint(), "Retrieval inputs changed"
    assert metadata["output_sha256"] == digest(OUT / "retrieval_results.parquet")
    retrieved = pd.read_parquet(OUT / "retrieval_results.parquet")
    pools = {q: frame.dropna(subset=["candidate_rank"]).sort_values("candidate_rank") for q, frame in retrieved.groupby("query_id")}
    documents = dict(zip(ids, texts))
    rcfg = cfg["reranker"]
    model_path = local_model(rcfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to("cpu").float().eval()
    run_id = hashlib.sha256(json.dumps({"inputs": inputs_fingerprint(), "retrieval": metadata["output_sha256"], "script": digest(__file__)}, sort_keys=True).encode()).hexdigest()
    if not smoke:
        validation = json.loads((OUT / "smoke_validation.json").read_text())
        assert validation["passed"] and validation["run_id"] == run_id, "Run smoke validation for the current inputs and implementation first"
    checkpoint = OUT / "checkpoints" / run_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    if smoke:
        queries = [q for q in queries if q["split"] == "train"][:8]
    results = []
    started = time.perf_counter()
    for qi, query in enumerate(queries):
        qid = query["query_id"]
        path = checkpoint / (hashlib.sha256(qid.encode()).hexdigest() + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            assert result["query_id"] == qid
            results.append(result)
            continue
        pool = pools[qid]
        candidates = pool.document_id.tolist()
        assert len(candidates) == len(set(candidates)) == 20
        pairs = [(query["text"], documents[d]) for d in candidates]
        scores, lengths = [], []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(pairs), rcfg["batch_size"]):
                batch = pairs[offset:offset + rcfg["batch_size"]]
                encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation="longest_first", max_length=rcfg["max_pair_sequence_length"], return_tensors="pt")
                assert encoded["input_ids"].shape[1] <= 256
                lengths.extend(encoded["attention_mask"].sum(dim=1).tolist())
                logits = model(**encoded).logits
                assert logits.shape == (len(batch), 1)
                scores.extend(logits[:, 0].tolist())
        elapsed = time.perf_counter() - t0
        assert np.isfinite(scores).all()
        order = sorted(range(20), key=lambda i: (-scores[i], candidates[i]))
        reranked = [candidates[i] for i in order]
        assert set(reranked) == set(candidates)
        if smoke:
            # Independent single-pair inference verifies batch association and padding.
            encoded = tokenizer(*pairs[0], truncation="longest_first", max_length=256, return_tensors="pt")
            with torch.inference_mode():
                single = model(**encoded).logits.item()
            assert np.isclose(single, scores[0], atol=1e-4, rtol=1e-4), (single, scores[0])
        base = ndcg(candidates, positives[qid])
        reranked_quality = ndcg(reranked, positives[qid])
        result = {"query_id": qid, "split": query["split"], "group_id": query["group_id"], "base_ndcg": base, "reranked_ndcg": reranked_quality, "gain": reranked_quality - base, "positive_qrels": len(positives[qid]), "candidate_positive_count": len(set(candidates) & positives[qid]), "candidate_positive_recall": len(set(candidates) & positives[qid]) / len(positives[qid]), "rrf_margin": float(pool.rrf_margin.iloc[0]), "retrieval_jaccard": float(pool.retrieval_jaccard.iloc[0]), "candidate_ids": candidates, "reranked_ids": reranked, "cross_encoder_logits": scores, "pair_token_lengths": lengths, "rerank_seconds_diagnostic": elapsed}
        # Smoke results are verified but deliberately not reused by the full run.
        if not smoke:
            save_json(path, result)
        results.append(result)
        if smoke or (qi + 1) % 25 == 0:
            print(f"{'Smoke' if smoke else 'Reranked'} {qi + 1}/{len(queries)}; elapsed={time.perf_counter() - started:.1f}s; last_gain={result['gain']:.6f}", flush=True)
    if smoke:
        save_json(OUT / "smoke_validation.json", {"passed": True, "run_id": run_id, "queries": results, "checks": ["20 unique candidates preserved", "finite scalar logits", "maximum 256 pair tokens", "single-pair versus batch inference agreement", "document-ID score tie break", "hand-computed metric tests run separately"]})
    else:
        pd.DataFrame(results).to_parquet(OUT / "query_gains.parquet", index=False)
        save_json(OUT / "gain_metadata.json", {"runtime": runtime_metadata(), "run_id": run_id, "queries": len(results), "inputs": inputs_fingerprint(), "script_sha256": digest(__file__), "output_sha256": digest(OUT / "query_gains.parquet"), "test_accessed": False, "timing_note": "Diagnostic reranking times include tokenization and initial calls; not a warmup-controlled end-to-end benchmark."})
    print("Reranking stage complete", flush=True)


def report():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = json.loads(CONFIG.read_text())
    frame = pd.read_parquet(OUT / "query_gains.parquet")
    metadata = json.loads((OUT / "gain_metadata.json").read_text())
    assert metadata["inputs"] == inputs_fingerprint()
    assert metadata["output_sha256"] == digest(OUT / "query_gains.parquet")
    assert len(frame) == 2908 and frame.query_id.is_unique
    eps = cfg["evaluation"]["gain_epsilon"]
    summaries, curves, distributions, uncertainty = {}, [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, split in zip(axes, ("train", "dev")):
        df = frame.loc[frame.split == split].sort_values("query_id").reset_index(drop=True)
        n = len(df)
        gains = df.gain.to_numpy()
        base = df.base_ndcg.mean()
        summaries[split] = {"queries": n, "base_ndcg": base, "always_ndcg": df.reranked_ndcg.mean(), "positive_rate": float((gains > eps).mean()), "zero_rate": float((abs(gains) <= eps).mean()), "exact_zero_rate": float((gains == 0).mean()), "negative_rate": float((gains < -eps).mean()), "mean_gain": float(gains.mean()), "zero_gain_no_candidate_positive": int(((abs(gains) <= eps) & (df.candidate_positive_count.to_numpy() == 0)).sum())}
        for category, mask in (("positive", gains > eps), ("zero", abs(gains) <= eps), ("negative", gains < -eps)):
            values = gains[mask]
            distributions.append({"split": split, "category": category, "count": len(values), "mean": float(values.mean()) if len(values) else None, "median": float(np.median(values)) if len(values) else None, **{f"q{q}": float(np.percentile(values, q)) if len(values) else None for q in (5, 25, 75, 95)}})
        orders = {"oracle": np.argsort(-gains, kind="stable"), "score_margin": np.argsort(df.rrf_margin.to_numpy(), kind="stable"), "disagreement": np.argsort(df.retrieval_jaccard.to_numpy(), kind="stable")}
        selected_at_30 = {}
        for budget in cfg["evaluation"]["budgets"]:
            k = math.floor(budget * n)
            for policy, order in orders.items():
                changes = np.zeros(n)
                changes[order[:k]] = gains[order[:k]]
                curves.append({"split": split, "policy": policy, "budget": budget, "selected": k, "actual_fraction": k / n, "ndcg": float(base + changes.mean()), "gain_over_base": float(changes.mean())})
                if budget == 0.3:
                    selected_at_30[policy] = changes
            oracle_positive = orders["oracle"][:k]
            oracle_positive = oracle_positive[gains[oracle_positive] > 0]
            curves.append({"split": split, "policy": "oracle_at_most", "budget": budget, "selected": len(oracle_positive), "actual_fraction": len(oracle_positive) / n, "ndcg": float(base + gains[oracle_positive].sum() / n), "gain_over_base": float(gains[oracle_positive].sum() / n)})
            random_changes = []
            for seed in cfg["evaluation"]["random_seeds"]:
                chosen = np.random.default_rng(seed).permutation(n)[:k]
                changes = np.zeros(n)
                changes[chosen] = gains[chosen]
                random_changes.append(changes)
            average = np.mean(random_changes, axis=0)
            curves.append({"split": split, "policy": "random", "budget": budget, "selected": k, "actual_fraction": k / n, "ndcg": float(base + average.mean()), "gain_over_base": float(average.mean()), "seed_ndcg_std": float(np.std([base + x.mean() for x in random_changes], ddof=1))})
            if budget == 0.3:
                selected_at_30["random"] = average
        # Group bootstrap of fixed policy outcomes; paired differences share draws.
        groups = [np.flatnonzero(df.group_id.to_numpy() == g) for g in sorted(df.group_id.unique())]
        generator = np.random.default_rng(42)
        draws = {name: [] for name in selected_at_30}
        pairs = {name: [] for name in ("score_margin", "disagreement")}
        for _ in range(1000):
            indices = np.concatenate([groups[i] for i in generator.integers(len(groups), size=len(groups))])
            for name, changes in selected_at_30.items():
                draws[name].append(float(changes[indices].mean()))
            for name in pairs:
                pairs[name].append(float((selected_at_30[name][indices] - selected_at_30["random"][indices]).mean()))
        for name, values in draws.items():
            uncertainty.append({"split": split, "comparison": name + " minus never", "lower_95": float(np.percentile(values, 2.5)), "upper_95": float(np.percentile(values, 97.5))})
        for name, values in pairs.items():
            uncertainty.append({"split": split, "comparison": name + " minus random", "lower_95": float(np.percentile(values, 2.5)), "upper_95": float(np.percentile(values, 97.5))})
        ax.hist(gains, bins=50)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title=split.capitalize(), xlabel="Reranked minus base binary NDCG@10", ylabel="Query count")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/gain_distribution.png", dpi=160)
    plt.close(fig)
    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(ROOT / "reports/budget_curves.csv", index=False)
    pd.DataFrame(distributions).to_csv(ROOT / "reports/gain_distribution.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(ROOT / "reports/policy_intervals.csv", index=False)
    save_json(ROOT / "reports/gain_summary.json", summaries)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, split in zip(axes, ("train", "dev")):
        for policy, data in curves_df.loc[curves_df.split == split].groupby("policy"):
            ax.plot(data.budget, data.ndcg, marker="o", label=policy)
        ax.axhline(summaries[split]["base_ndcg"], color="black", linestyle="--", label="never")
        ax.axhline(summaries[split]["always_ndcg"], color="gray", linestyle=":", label="always")
        ax.set(title=split.capitalize(), xlabel="Reranking call fraction", ylabel="Mean binary NDCG@10")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/budget_curves.png", dpi=160)
    plt.close(fig)
    lines = ["# Retrieval and Reranking Feasibility Results", "", "Measured on eligible training queries and the development split. Test judgments were not accessed.", "", "| Metric | Train | Dev |", "|---|---:|---:|"]
    for label, key in (("Queries", "queries"), ("Positive gain rate", "positive_rate"), ("Zero gain rate (absolute gain <= 1e-6)", "zero_rate"), ("Exact zero gain rate", "exact_zero_rate"), ("Negative gain rate", "negative_rate"), ("Mean delta NDCG@10", "mean_gain"), ("Never-rerank NDCG@10", "base_ndcg"), ("Always-rerank NDCG@10", "always_ndcg")):
        vals = [summaries[s][key] for s in ("train", "dev")]
        formatted = [f"{v:.2%}" if "rate" in key else (str(v) if key == "queries" else f"{v:.6f}") for v in vals]
        lines.append(f"| {label} | {' | '.join(formatted)} |")
    for policy in ("oracle", "oracle_at_most", "random", "score_margin", "disagreement"):
        vals = [curves_df.loc[(curves_df.split == s) & (curves_df.budget == 0.3) & (curves_df.policy == policy)].iloc[0] for s in ("train", "dev")]
        lines.append(f"| {policy} @ 30%: NDCG@10 (gain over base) | " + " | ".join(f"{v.ndcg:.6f} ({v.gain_over_base:+.6f})" for v in vals) + " |")
    lines.extend(["", "## Gain magnitude and zero-gain diagnostics", "", "| Split | Category | Count | Mean gain | Median gain |", "|---|---|---:|---:|---:|"])
    for row in distributions:
        mean = f"{row['mean']:.6f}" if row["mean"] is not None else "N/A"
        median = f"{row['median']:.6f}" if row["median"] is not None else "N/A"
        lines.append(f"| {row['split']} | {row['category']} | {row['count']} | {mean} | {median} |")
    for split in ("train", "dev"):
        lines.extend(["", f"{split.capitalize()}: {summaries[split]['zero_gain_no_candidate_positive']} zero-gain queries have no judged positive document in the candidate pool. Query-level base quality and candidate coverage are preserved for further diagnosis."])
    lines.extend(["", "The fixed 30% budget uses floor(0.3 * query_count): 775/2,584 training queries and 97/324 development queries. Random results average seeds 0–9. Score ties use ascending query ID. Oracle is diagnostic and uses observed gains. The at-most-budget oracle may select fewer queries.", "", "Group-bootstrap intervals in policy_intervals.csv resample fixed query-level policy outcomes; they do not refit or reselect the budget in each replicate. Random intervals condition on the ten seeded selections; seed variability is also recorded in budget_curves.csv.", "", "Reranking timings are diagnostic only. A controlled live end-to-end CPU benchmark and learned routing have not been performed.", "", "![Gain distribution](gain_distribution.png)", "", "![Budget curves](budget_curves.png)"])
    (ROOT / "reports/feasibility_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:24]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["retrieve", "smoke", "rerank", "report"])
    args = parser.parse_args()
    if args.stage == "retrieve":
        retrieve()
    elif args.stage in ("smoke", "rerank"):
        rerank(smoke=args.stage == "smoke")
    else:
        report()

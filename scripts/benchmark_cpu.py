"""Warm, two-pass fixed-budget CPU benchmark; never reads test judgments."""
import json
import math
import pickle
import platform
import time

from experiment import ROOT, OUT, CONFIG, load_data, local_model, setup_torch, tokens, ranked, ndcg, digest, save_json

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    cfg, ids, texts, queries, positives = load_data()
    setup_torch(cfg)
    dev = sorted([q for q in queries if q["split"] == "dev"], key=lambda q: q["query_id"])
    order = np.random.default_rng(42).permutation(len(dev))
    dev = [dev[i] for i in order]
    protocol = {"rounds": 2, "policies": ["never", "always", "disagreement", "regression_tree"], "policy_orders": [["never", "always", "disagreement", "regression_tree"], ["regression_tree", "disagreement", "always", "never"]], "query_order": [q["query_id"] for q in dev], "warmup_train_queries": 8, "budget": 0.3, "execution": "two-pass batch: live first-stage for every query, batch-wide allocation, selected-query reranking; no cross-query neural batching", "latency": "observed completion latency measured from each query first-stage start to its final ranking completion; includes batch barrier and queue wait. Active-service time separately sums timed work and amortized batch allocation; not online request latency", "resident_assets": "models, BM25 index, precomputed corpus embeddings; no cached query vectors, candidates, reranking scores, or decisions"}
    save_json(ROOT / "configs/cpu_benchmark_protocol.json", protocol)
    startup = time.perf_counter()
    dcfg, rcfg = cfg["retrieval"]["dense"], cfg["reranker"]
    dense = SentenceTransformer(local_model(dcfg), device="cpu", local_files_only=True)
    dense.max_seq_length = 256
    path = local_model(rcfg)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cross = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).float().eval()
    documents = np.load(OUT / "document_embeddings.npy")
    assert documents.shape == (3633, 384)
    bcfg = cfg["retrieval"]["bm25"]
    bm25 = BM25Okapi([tokens(t) for t in texts], k1=bcfg["k1"], b=bcfg["b"], epsilon=bcfg["epsilon"])
    model_path = OUT / "routing_models/regression_tree.pkl"
    assert digest(model_path) == json.loads((OUT / "routing_metadata.json").read_text())["model_sha256"][model_path.name]
    with model_path.open("rb") as source:
        tree = pickle.load(source)
    names = cfg["routing"]["features"]
    startup_seconds = time.perf_counter() - startup

    def retrieve(query, policy):
        started = time.perf_counter()
        t = time.perf_counter()
        vec = dense.encode([query["text"]], batch_size=1, normalize_embeddings=True, show_progress_bar=False)[0]
        embedding = time.perf_counter() - t
        t = time.perf_counter()
        bs = bm25.get_scores(tokens(query["text"]))
        bi = ranked(bs, ids, 100)
        lexical = time.perf_counter() - t
        t = time.perf_counter()
        ds = documents @ vec
        di = ranked(ds, ids, 100)
        search = time.perf_counter() - t
        t = time.perf_counter()
        br, dr = {i: r + 1 for r, i in enumerate(bi)}, {i: r + 1 for r, i in enumerate(di)}
        fusion = {i: (1 / (60 + br[i]) if i in br else 0) + (1 / (60 + dr[i]) if i in dr else 0) for i in set(bi) | set(di)}
        pool = sorted(fusion, key=lambda i: (-fusion[i], ids[i]))[:20]
        fusion_time = time.perf_counter() - t
        t = time.perf_counter()
        score = 0.0
        if policy == "disagreement":
            score = -len(set(bi[:20]) & set(di[:20])) / len(set(bi[:20]) | set(di[:20]))
        elif policy == "regression_tree":
            f = {"query_word_count": len(query["text"].split()), "query_character_count": len(query["text"]), "rrf_top1_top2_margin": fusion[pool[0]] - fusion[pool[1]], "bm25_dense_top20_jaccard": len(set(bi[:20]) & set(di[:20])) / len(set(bi[:20]) | set(di[:20]))}
            for name, values in (("bm25", bs[bi[:20]]), ("dense", ds[di[:20]])):
                f.update({name + "_top1": values[0], name + "_top1_top2_margin": values[0] - values[1], name + "_top20_mean": values.mean(), name + "_top20_std": values.std(ddof=0)})
            score = float(tree.predict(pd.DataFrame([[f[n] for n in names]], columns=names))[0])
        routing = time.perf_counter() - t
        return {"query_id": query["query_id"], "query": query["text"], "pool": pool, "score": score, "started": started, "first_stage_seconds": time.perf_counter() - started, "embedding_seconds": embedding, "bm25_seconds": lexical, "dense_search_seconds": search, "rrf_seconds": fusion_time, "routing_seconds": routing}

    def rerank(record):
        started = time.perf_counter()
        scores = []
        with torch.inference_mode():
            for offset in range(0, 20, 8):
                indices = record["pool"][offset:offset + 8]
                encoded = tokenizer([record["query"]] * len(indices), [texts[i] for i in indices], padding=True, truncation="longest_first", max_length=256, return_tensors="pt")
                scores.extend(cross(**encoded).logits[:, 0].tolist())
        assert np.isfinite(scores).all()
        ranking = sorted(range(20), key=lambda i: (-scores[i], ids[record["pool"][i]]))
        return [ids[record["pool"][i]] for i in ranking], time.perf_counter() - started

    warmups = [q for q in queries if q["split"] == "train"][:8]
    for query in warmups:
        item = retrieve(query, "regression_tree")
        rerank(item)
    print("Model and pipeline warmup complete", flush=True)
    rows, batches = [], []
    for round_index, policies in enumerate(protocol["policy_orders"], start=1):
        for policy in policies:
            started = time.perf_counter()
            records = []
            for query in dev:
                record = retrieve(query, policy)
                if policy == "never":
                    record["completed"] = time.perf_counter()
                records.append(record)
            t = time.perf_counter()
            if policy == "always":
                chosen = set(range(len(records)))
            elif policy == "never":
                chosen = set()
            else:
                chosen = set(sorted(range(len(records)), key=lambda i: (-records[i]["score"], records[i]["query_id"]))[:math.floor(0.3 * len(records))])
            allocation_seconds = time.perf_counter() - t
            if policy != "never":
                for i, item in enumerate(records):
                    if i not in chosen:
                        item["completed"] = time.perf_counter()
            # Always also uses the two-pass schedule for comparable batch accounting.
            for i, item in enumerate(records):
                if i in chosen:
                    item["ranking"], item["cross_encoder_seconds"] = rerank(item)
                    item["completed"] = time.perf_counter()
                else:
                    item["ranking"] = [ids[j] for j in item["pool"]]
                    item["cross_encoder_seconds"] = 0.0
            batch_seconds = time.perf_counter() - started
            for i, item in enumerate(records):
                rows.append({"round": round_index, "policy": policy, "query_id": item["query_id"], "selected": i in chosen, "ndcg": ndcg(item["ranking"], positives[item["query_id"]]), "completion_latency_seconds": item["completed"] - item["started"], "active_service_seconds": item["first_stage_seconds"] + item["cross_encoder_seconds"] + allocation_seconds / len(records), **{key: item[key] for key in ("embedding_seconds", "bm25_seconds", "dense_search_seconds", "rrf_seconds", "routing_seconds", "cross_encoder_seconds")}, "candidate_ids": [ids[j] for j in item["pool"]], "ranking": item["ranking"]})
            batches.append({"round": round_index, "policy": policy, "queries": len(records), "rerank_calls": len(chosen), "allocation_seconds": allocation_seconds, "batch_seconds": batch_seconds, "queries_per_second": len(records) / batch_seconds})
            pd.DataFrame(rows).to_parquet(OUT / "cpu_benchmark_queries.parquet", index=False)
            pd.DataFrame(batches).to_csv(ROOT / "reports/cpu_benchmark_batches.csv", index=False)
            print(f"Round {round_index} {policy}: {batch_seconds:.1f}s, {len(chosen)} rerank calls", flush=True)
    frame = pd.DataFrame(rows)
    summaries = []
    for policy, data in frame.groupby("policy"):
        row = {"policy": policy, "ndcg": data.ndcg.mean(), "rounds": 2, "queries_per_round": 324}
        for name in ("completion_latency_seconds", "active_service_seconds", "embedding_seconds", "bm25_seconds", "dense_search_seconds", "rrf_seconds", "routing_seconds", "cross_encoder_seconds"):
            row[name.replace("_seconds", "_p50_ms")] = data[name].quantile(0.5) * 1000
            row[name.replace("_seconds", "_p95_ms")] = data[name].quantile(0.95) * 1000
        row["mean_batch_seconds"] = np.mean([b["batch_seconds"] for b in batches if b["policy"] == policy])
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(ROOT / "reports/cpu_benchmark_summary.csv", index=False)
    save_json(OUT / "cpu_benchmark_metadata.json", {"script_sha256": digest(__file__), "protocol_sha256": digest(ROOT / "configs/cpu_benchmark_protocol.json"), "startup_seconds": startup_seconds, "document_embedding_sha256": digest(OUT / "document_embeddings.npy"), "platform": platform.platform(), "processor": platform.processor(), "threads": 4, "interop_threads": 1, "test_accessed": False, "warmup_queries": [q["query_id"] for q in warmups]})


if __name__ == "__main__":
    main()

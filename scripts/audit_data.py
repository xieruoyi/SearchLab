"""Audit BEIR NFCorpus train/dev without opening test qrels.

Python standard library only. Run from any working directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import statistics
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/nfcorpus.zip")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    payload = args.archive.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload).hexdigest()
    if md5 != config["dataset"]["expected_md5"]:
        raise ValueError("Archive does not match the registered BEIR version")
    if sha != config["dataset"]["sha256"]:
        raise ValueError("Archive SHA256 changed")

    qrels = {}
    opened = []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        def read_member(name):
            opened.append(name)
            return z.read(name).decode("utf-8")

        # Test qrels remain inside the archive. Do not read or extract them.
        for split in ("train", "dev"):
            text = read_member(f"nfcorpus/qrels/{split}.tsv")
            rows = list(csv.DictReader(io.StringIO(text, newline=None), delimiter="\t"))
            mapping = defaultdict(dict)
            for row in rows:
                qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
                if did in mapping[qid]:
                    raise ValueError(f"Repeated qrel pair: {split}/{qid}/{did}")
                if score < 0:
                    raise ValueError("Unexpected negative relevance grade")
                mapping[qid][did] = score
            qrels[split] = mapping

        corpus = {}
        for line in read_member("nfcorpus/corpus.jsonl").splitlines():
            doc = json.loads(line)
            if doc["_id"] in corpus:
                raise ValueError("Duplicate corpus ID")
            corpus[doc["_id"]] = doc
        allowed = set(qrels["train"]) | set(qrels["dev"])
        queries = {}
        for line in read_member("nfcorpus/queries.jsonl").splitlines():
            obj = json.loads(line)
            # Shared file contains test queries: retain only train/dev query IDs.
            if obj["_id"] in allowed:
                if obj["_id"] in queries:
                    raise ValueError("Duplicate development query ID")
                queries[obj["_id"]] = obj["text"]

    overlap = set(qrels["train"]) & set(qrels["dev"])
    missing_queries = allowed - set(queries)
    missing_docs = {d for m in qrels.values() for pairs in m.values() for d in pairs} - set(corpus)
    if overlap or missing_queries or missing_docs:
        raise ValueError(f"Integrity failure: overlap={len(overlap)}, missing queries={len(missing_queries)}, missing docs={len(missing_docs)}")
    if any(not text.strip() for text in queries.values()):
        raise ValueError("Empty query text")

    # Label-free duplicate audit. Group token-set Jaccard >= 0.85, including
    # transitive connections. This is a lexical check, not semantic deduplication.
    ids = sorted(queries)
    normalized = {q: normalize(queries[q]) for q in ids}
    tokens = {q: set(normalized[q].split()) for q in ids}
    parent = {q: q for q in ids}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    def union(a, b):
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)

    duplicate_pairs = []
    threshold = config["splits"]["near_duplicate_token_jaccard"]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ta, tb = tokens[a], tokens[b]
            if not ta or not tb or min(len(ta), len(tb)) / max(len(ta), len(tb)) < threshold:
                continue
            jaccard = len(ta & tb) / len(ta | tb)
            if normalized[a] == normalized[b] or jaccard >= threshold:
                union(a, b)
                duplicate_pairs.append({"query_a": a, "query_b": b, "token_jaccard": round(jaccard, 6), "normalized_exact": normalized[a] == normalized[b]})

    groups = defaultdict(list)
    for q in ids:
        groups[find(q)].append(q)
    dev_groups = {find(q) for q in qrels["dev"]}
    excluded = {q for q in qrels["train"] if find(q) in dev_groups}
    manifest = []
    for split, mapping in qrels.items():
        for q in sorted(mapping):
            manifest.append({"query_id": q, "official_split": split, "group_id": find(q),
                             "routing_role": "excluded_train_overlap_dev" if q in excluded else ("fit_group_cv" if split == "train" else "validation"),
                             "query_text_sha256": hashlib.sha256(queries[q].encode()).hexdigest()})

    audit = []
    def add(scope, metric, value, note=""):
        audit.append({"scope": scope, "metric": metric, "value": value, "note": note})

    add("corpus", "documents", len(corpus))
    add("corpus", "empty_title_and_text", sum(not (d.get("title", "") + d.get("text", "")).strip() for d in corpus.values()))
    for split, mapping in qrels.items():
        positive_counts = [sum(s > 0 for s in pairs.values()) for pairs in mapping.values()]
        add(split, "queries_with_qrels", len(mapping))
        add(split, "qrel_pairs", sum(map(len, mapping.values())))
        add(split, "query_groups", len({find(q) for q in mapping}))
        add(split, "normalized_exact_duplicate_excess", len(mapping) - len({normalized[q] for q in mapping}))
        add(split, "queries_without_positive_qrel", sum(n == 0 for n in positive_counts))
        add(split, "positive_docs_per_query_median", statistics.median(positive_counts))
        add(split, "query_word_count_median", statistics.median(len(queries[q].split()) for q in mapping))
        for grade, count in sorted(Counter(s for pairs in mapping.values() for s in pairs.values()).items()):
            add(split, f"qrel_grade_{grade}_pairs", count)
    add("train", "excluded_queries_overlapping_dev_groups", len(excluded))
    add("train", "routing_fit_queries", len(qrels["train"]) - len(excluded))
    add("train", "routing_fit_groups", len({find(q) for q in qrels["train"] if q not in excluded}))
    add("integrity", "train_dev_query_id_overlap", len(overlap))
    add("integrity", "missing_query_ids", len(missing_queries))
    add("integrity", "missing_document_ids", len(missing_docs))
    add("development", "duplicate_pairs", len(duplicate_pairs), "Normalized exact or token-set Jaccard >= 0.85; lexical approximation")
    add("test", "status", "sealed", "qrels/test.tsv not opened; no test counts or metrics produced")

    report = ROOT / "reports"
    write_csv(report / "data_audit.csv", audit, ["scope", "metric", "value", "note"])
    write_csv(report / "duplicate_pairs.csv", duplicate_pairs, ["query_a", "query_b", "token_jaccard", "normalized_exact"])
    write_csv(ROOT / "data/processed/split_manifest.csv", manifest, ["query_id", "official_split", "group_id", "routing_role", "query_text_sha256"])
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for did in sorted(corpus):
            f.write(json.dumps(corpus[did], ensure_ascii=False) + "\n")
    for split, mapping in qrels.items():
        with (processed / f"{split}_queries.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(mapping):
                f.write(json.dumps({"_id": q, "text": queries[q]}, ensure_ascii=False) + "\n")
        with (processed / f"{split}_qrels.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["query-id", "corpus-id", "score"])
            for q in sorted(mapping):
                for did, score in sorted(mapping[q].items()):
                    writer.writerow([q, did, score])
    write_json(report / "audit_provenance.json", {
        "dataset_url": config["dataset"]["url"], "archive_sha256": sha, "archive_md5": md5,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((ROOT / "configs/experiment.json").read_bytes()).hexdigest(),
        "python": sys.version, "platform": platform.platform(), "utc": datetime.now(timezone.utc).isoformat(),
        "opened_archive_members": opened, "test_qrels_opened": False,
        "duplicate_rule": "NFKC/casefold punctuation-free exact OR token-set Jaccard>=0.85; connected components",
    })
    print(json.dumps({"documents": len(corpus), "train_queries": len(qrels["train"]), "dev_queries": len(qrels["dev"]), "excluded_train_queries": len(excluded), "fit_groups": len({find(q) for q in qrels["train"] if q not in excluded}), "duplicate_pairs": len(duplicate_pairs), "test": "sealed"}, indent=2))


if __name__ == "__main__":
    main()

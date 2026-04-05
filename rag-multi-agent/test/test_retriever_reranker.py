import csv
import re
import time

import pandas as pd
import pytest

from multiagent_rag.agents.reranker import Reranker
from multiagent_rag.agents.retriever import Retriever

_tracker = []


def get_tokens(text: str) -> set:
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())


def token_overlap_score(target: str, context: str) -> float:
    target_tokens = get_tokens(target)
    context_tokens = get_tokens(context)
    if not target_tokens:
        return 0.0

    matches = target_tokens.intersection(context_tokens)
    return len(matches) / len(target_tokens)


def is_match(target: str, context: str, threshold=0.7) -> bool:
    return token_overlap_score(target, context) >= threshold


def precision_at_k(retrieved, target, k):
    top_k = retrieved[:k]
    if not top_k or not target:
        return 0.0
    hits = sum(1 for doc in top_k if is_match(target, doc["content"]))
    return hits / k


def recall_at_k(retrieved, target, k):
    top_k = retrieved[:k]
    if not top_k or not target:
        return 0.0
    hit = any(is_match(target, doc["content"]) for doc in top_k)
    return 1.0 if hit else 0.0


def hit_at_k(retrieved, target, k):
    return int(recall_at_k(retrieved, target, k) > 0)


def reciprocal_rank(retrieved, target):
    if not target:
        return 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if is_match(target, doc["content"]):
            return 1.0 / rank
    return 0.0


def load_data():
    path = "data/Retriever_Reranker_Eval_Dataset.csv"
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module", autouse=True)
def print_final_summary():
    yield
    if not _tracker:
        print("\n\n" + "=" * 55)
        print(" RAG PIPELINE EVALUATION SUMMARY (No Data Captured) ")
        print("=" * 55)
        return

    df = pd.DataFrame(_tracker)
    metrics = ["P@1", "R@1", "Hit@1", "P@3", "R@3", "Hit@3", "P@5", "R@5", "Hit@5", "MRR"]

    print("\n\n" + "=" * 55)
    print(f" RAG PIPELINE EVALUATION SUMMARY (Token Overlap Threshold: 70%) ")
    print(f" Total Queries: {len(_tracker)} ")
    print("=" * 55)

    summary = df[metrics].mean()
    for name, value in summary.items():
        print(f"{name:20s}: {value:.4f}")

    avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else 0
    print("-" * 55)
    print(f"{'Avg Latency':20s}: {avg_latency:.2f} ms")
    print("=" * 55 + "\n")


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


@pytest.mark.parametrize("row", load_data())
def test_retrieval_pipeline(retriever, reranker, row):
    query = row["query"]
    target = row["target_document_snippet"]

    start_time = time.perf_counter()
    retrieved = retriever.retrieve(query, k=20)

    if not retrieved:
        _tracker.append({m: 0.0 for m in ["P@1", "R@1", "Hit@1", "P@3", "R@3", "Hit@3", "P@5", "R@5", "Hit@5", "MRR"]})
        _tracker[-1].update({"latency_ms": (time.perf_counter() - start_time) * 1000})
        pytest.fail(f"Retriever found 0 results for: {query}")

    reranked = reranker.rerank(query, retrieved, top_k=5)
    latency_ms = (time.perf_counter() - start_time) * 1000

    metrics = {"P@1": precision_at_k(reranked, target, 1), "R@1": recall_at_k(reranked, target, 1),
        "Hit@1": hit_at_k(reranked, target, 1), "P@3": precision_at_k(reranked, target, 3),
        "R@3": recall_at_k(reranked, target, 3), "Hit@3": hit_at_k(reranked, target, 3),
        "P@5": precision_at_k(reranked, target, 5), "R@5": recall_at_k(reranked, target, 5),
        "Hit@5": hit_at_k(reranked, target, 5), "MRR": reciprocal_rank(reranked, target), "latency_ms": latency_ms}
    _tracker.append(metrics)

    match_found = any(is_match(target, doc["content"]) for doc in reranked)
    if not match_found:
        best_overlap = max(token_overlap_score(target, doc["content"]) for doc in reranked)
        print(f"\n[DEBUG] Query: {query}")
        print(f"[DEBUG] Best Token Overlap Found: {best_overlap:.1%}")

    assert match_found, f"Insufficient information match (below 70%) for: {query}"

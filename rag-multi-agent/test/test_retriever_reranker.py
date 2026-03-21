import csv

import pytest
from sklearn.metrics import classification_report, confusion_matrix

from multiagent_rag.agents.reranker import Reranker

tracker = {"y_true": [], "y_pred": []}


@pytest.fixture(scope="module", autouse=True)
def report_metrics():
    yield
    print("\n\n=== Reranker Metrics ===")
    print(classification_report(tracker["y_true"], tracker["y_pred"], zero_division=0))
    print(confusion_matrix(tracker["y_true"], tracker["y_pred"]))


def load_data():
    with open("data/Retriever_Reranker_Eval_Dataset.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


@pytest.mark.parametrize("row", load_data())
def test_reranker(reranker, row):
    query = row["query"]
    target = row["target_document_snippet"]
    distractor = row["distractor_snippet"]

    docs = [{"content": target}, {"content": distractor}]
    result = reranker.rerank(query, docs, top_k=2)

    is_match = result[0]["content"] == target

    tracker["y_true"].append("Hit")
    tracker["y_pred"].append("Hit" if is_match else "Miss")

    assert is_match

import csv

import pytest

from multiagent_rag.agents.reranker import Reranker


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

    assert result[0]["content"] == target

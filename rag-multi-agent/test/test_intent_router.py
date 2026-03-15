import csv
import time

import pytest
from langchain_core.messages import AIMessage

from multiagent_rag.graph.rag_router import IntentRouter


def load_data():
    with open("data/IntentRouter_Eval_Dataset.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def router():
    return IntentRouter()


@pytest.mark.parametrize("row", load_data())
def test_intent_router(router, row):
    time.sleep(1)

    query = row["query"]
    history_str = row["history_context"]
    expected = row["expected_intent"]

    history = []
    if history_str:
        history.append(AIMessage(content=history_str))

    result = router.route(query, history)
    assert result == expected

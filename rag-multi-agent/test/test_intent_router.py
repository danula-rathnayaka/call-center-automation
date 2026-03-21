import csv
import time

import pytest
from langchain_core.messages import AIMessage
from sklearn.metrics import classification_report, confusion_matrix

from multiagent_rag.graph.rag_router import IntentRouter

tracker = {"y_true": [], "y_pred": []}


@pytest.fixture(scope="module", autouse=True)
def report_metrics():
    yield
    print("\n\n=== Intent Router Metrics ===")
    print(classification_report(tracker["y_true"], tracker["y_pred"], zero_division=0))
    print(confusion_matrix(tracker["y_true"], tracker["y_pred"]))


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

    tracker["y_true"].append(expected)
    tracker["y_pred"].append(result)

    assert result == expected

import csv

import pytest
from sklearn.metrics import classification_report, confusion_matrix

from multiagent_rag.agents.generator import Generator

tracker = {"y_true": [], "y_pred": []}


@pytest.fixture(scope="module", autouse=True)
def report_metrics():
    yield
    print("\n\n=== Generator Metrics ===")
    print(classification_report(tracker["y_true"], tracker["y_pred"], zero_division=0))
    print(confusion_matrix(tracker["y_true"], tracker["y_pred"]))


def load_data():
    with open("data/Generator_Eval_Dataset.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def generator():
    return Generator()


@pytest.mark.parametrize("row", load_data())
def test_generator(generator, row):
    query = row["query"]
    context = row["provided_context"]

    result = generator.generate(query, context, [])

    is_valid = isinstance(result, str) and len(result.strip()) > 0 and "System Error" not in result

    tracker["y_true"].append("Valid")
    tracker["y_pred"].append("Valid" if is_valid else "Invalid")

    assert is_valid

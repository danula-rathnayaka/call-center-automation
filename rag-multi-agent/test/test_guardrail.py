import csv

import pytest
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()

from multiagent_rag.agents.guardrail import Guardrail

tracker = {"y_true": [], "y_pred": []}


@pytest.fixture(scope="module", autouse=True)
def report_metrics():
    yield
    print("\n\n=== Guardrail Metrics ===")
    print(classification_report(tracker["y_true"], tracker["y_pred"], zero_division=0))
    print(confusion_matrix(tracker["y_true"], tracker["y_pred"]))


def load_data():
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "Guardrail_Eval_Dataset.csv")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def guardrail():
    return Guardrail()


@pytest.mark.parametrize("row", load_data())
def test_guardrail(guardrail, row):
    import time
    time.sleep(1.5)

    input_text = row["input_text"]
    expected_safe = row["expected_safe"].strip().upper() == "TRUE"
    expected_reason = row["expected_reason"]
    expected_sanitized = row["expected_sanitized_text"]

    val_result = guardrail.validate(input_text)

    tracker["y_true"].append("Safe" if expected_safe else "Unsafe")
    tracker["y_pred"].append("Safe" if val_result["safe"] else "Unsafe")

    assert val_result["safe"] == expected_safe

    if not expected_safe:
        assert val_result["reason"] == expected_reason

    sanitized_result = guardrail.sanitize_response(input_text)
    assert sanitized_result == expected_sanitized

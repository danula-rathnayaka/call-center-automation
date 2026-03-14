import csv

import pytest

from multiagent_rag.agents.guardrail import Guardrail


def load_data():
    with open("data/Guardrail_Eval_Dataset.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def guardrail():
    return Guardrail()


@pytest.mark.parametrize("row", load_data())
def test_guardrail(guardrail, row):
    input_text = row["input_text"]
    expected_safe = row["expected_safe"].strip().upper() == "TRUE"
    expected_sanitized = row["expected_sanitized_text"]

    val_result = guardrail.validate(input_text)
    assert val_result["safe"] == expected_safe

    sanitized_result = guardrail.sanitize_response(input_text)
    assert sanitized_result == expected_sanitized

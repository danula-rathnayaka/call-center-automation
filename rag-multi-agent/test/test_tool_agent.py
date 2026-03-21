import csv
import json

import pytest
from langchain_core.messages import AIMessage
from sklearn.metrics import classification_report, confusion_matrix

from multiagent_rag.agents.tool_agent import ToolAgent

tracker = {"y_true": [], "y_pred": []}


@pytest.fixture(scope="module", autouse=True)
def report_metrics():
    yield
    print("\n\n=== Tool Agent Metrics ===")
    print(classification_report(tracker["y_true"], tracker["y_pred"], zero_division=0))
    print(confusion_matrix(tracker["y_true"], tracker["y_pred"]))


def load_data():
    with open("data/Tool_Agent_Eval_Dataset.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def agent():
    return ToolAgent()


@pytest.mark.parametrize("row", load_data())
def test_tool_agent(agent, row):
    query = row["query"]
    history_str = row["chat_history"]
    expected_tool = row["expected_tool"]
    expected_params = row["expected_parameters"]

    history = []
    if history_str:
        history.append(AIMessage(content=history_str))

    result = agent.invoke(query, history)

    actual_tool = "NONE"
    if hasattr(result, "tool_calls") and result.tool_calls:
        actual_tool = result.tool_calls[0]["name"]

    tracker["y_true"].append(expected_tool)
    tracker["y_pred"].append(actual_tool)

    if expected_tool == "NONE":
        assert not hasattr(result, "tool_calls") or not result.tool_calls
    else:
        assert hasattr(result, "tool_calls") and result.tool_calls
        call = result.tool_calls[0]
        assert call["name"] == expected_tool

        if expected_params != "MISSING":
            params = json.loads(expected_params)
            for k, v in params.items():
                assert call["args"].get(k) == v

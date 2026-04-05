import warnings

import pytest
from langchain_core.messages import AIMessage
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

from multiagent_rag.agents.tool_agent import ToolAgent

tracker = {"y_true": [], "y_pred": []}


@pytest.fixture(scope="module", autouse=True)
def report_metrics():
    yield
    print("\n\n=== Tool Agent Metrics ===")
    print(classification_report(tracker["y_true"], tracker["y_pred"], zero_division=0))
    print(confusion_matrix(tracker["y_true"], tracker["y_pred"]))

    print("Mismatches:")
    for (y_t, y_p) in zip(tracker["y_true"], tracker["y_pred"]):
        if y_t != y_p:
            pass


test_data = [{"query": "Verify my identity, mobile 0712345678 and NIC 987654321V", "chat_history": "",
              "expected_tool": "verify_customer_identity",
              "expected_parameters": {"mobile_number": "0712345678", "national_id": "987654321V"}},
    {"query": "Hi, I need verification. Number 0771122334, NIC 199912345678.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0771122334", "national_id": "199912345678"}},
    {"query": "I am 0755555555 and NIC is 888888888V, please verify me.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0755555555", "national_id": "888888888V"}},
    {"query": "Verify account. Mobile: 0722222222, nic: 123456789V", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0722222222", "national_id": "123456789V"}},
    {"query": "Please check my identity for 0788888888 with NIC 999999999V", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0788888888", "national_id": "999999999V"}},
    {"query": "NIC is 111111111V and mobile 0744444444. Verify my identity.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0744444444", "national_id": "111111111V"}},
    {"query": "Check if I am verified. 0711111111 and 222222222V.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0711111111", "national_id": "222222222V"}},
    {"query": "Identity check. 0700000000, 333333333V.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0700000000", "national_id": "333333333V"}},
    {"query": "Run a verification for 0777777777 using NIC 444444444V.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0777777777", "national_id": "444444444V"}},
    {"query": "Verify identity 0799999999 NIC 555555555V", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0799999999", "national_id": "555555555V"}},

    {"query": "What is my balance? mobile 0712345678", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0712345678"}},
    {"query": "Check balance for 0771122334", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0771122334"}},
    {"query": "Can you check my account balance on 0755555555?", "chat_history": "",
     "expected_tool": "check_account_balance", "expected_parameters": {"mobile_number": "0755555555"}},
    {"query": "Remaining data and voice for 0722222222.", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0722222222"}},
    {"query": "Data balance for 0788888888", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0788888888"}},
    {"query": "Voice balance for 0744444444", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0744444444"}},
    {"query": "Please retrieve my balance information. 0711111111", "chat_history": "",
     "expected_tool": "check_account_balance", "expected_parameters": {"mobile_number": "0711111111"}},
    {"query": "What's my current balance for 0700000000?", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0700000000"}},
    {"query": "I need my balance, number is 0777777777", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0777777777"}},
    {"query": "Bal for 0799999999", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0799999999"}},
    {"query": "check my account status, my mobile is 0764123123", "chat_history": "",
     "expected_tool": "check_account_balance", "expected_parameters": {"mobile_number": "0764123123"}},

    {"query": "Please check identity for mobile 0770000111 and nic 11110000V", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0770000111", "national_id": "11110000V"}},
    {"query": "Verify me, 0715555555 and NIC 123123123V", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0715555555", "national_id": "123123123V"}},
    {"query": "My nic is 555555555V, check mobile 0781234567", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0781234567", "national_id": "555555555V"}},
    {"query": "I am looking for verification: 0729998887, 888888888V", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0729998887", "national_id": "888888888V"}},
    {"query": "Verify this user please. 0778889990 and 999111222V.", "chat_history": "",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0778889990", "national_id": "999111222V"}},

    {"query": "Remaining account balance for 0712223334?", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0712223334"}},
    {"query": "Find the balance for number 0773334445", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0773334445"}},
    {"query": "Give me balance of 0787776665", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0787776665"}},
    {"query": "I need data status for 0751112223", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0751112223"}},
    {"query": "Voice usage limits for 0702223334", "chat_history": "", "expected_tool": "check_account_balance",
     "expected_parameters": {"mobile_number": "0702223334"}},

    {"query": "0712345678 and my NIC is 987654321V",
     "chat_history": "Please provide your mobile number and NIC to verify your identity.",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0712345678", "national_id": "987654321V"}},
    {"query": "My number is 0771122334", "chat_history": "What is your mobile number to check the balance?",
     "expected_tool": "check_account_balance", "expected_parameters": {"mobile_number": "0771122334"}},
    {"query": "999999999V",
     "chat_history": "You provided the mobile 0788888888, please also provide your NIC to verify.",
     "expected_tool": "verify_customer_identity",
     "expected_parameters": {"mobile_number": "0788888888", "national_id": "999999999V"}}, {"query": "0788888888",
                                                                                            "chat_history": "You provided your NIC 999999999V for verification, now please provide your mobile number.",
                                                                                            "expected_tool": "verify_customer_identity",
                                                                                            "expected_parameters": {
                                                                                                "mobile_number": "0788888888",
                                                                                                "national_id": "999999999V"}},
    {"query": "It is 0711111111.", "chat_history": "Could you give me your phone number to see the account balance?",
     "expected_tool": "check_account_balance", "expected_parameters": {"mobile_number": "0711111111"}},

    {"query": "Hello", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Hi there", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Good morning", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Thank you", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Who are you?", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "What is the weather today?", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Can you sing a song?", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Tell me a joke.", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "I am angry", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Talk to a human", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "I want to escalate this", "chat_history": "", "expected_tool": "NONE", "expected_parameters": None},
    {"query": "Hello I need my data status for 0713333333", "chat_history": "",
     "expected_tool": "check_account_balance", "expected_parameters": {"mobile_number": "0713333333"}},
    {"query": "Does LankaLink have new plans?", "chat_history": "", "expected_tool": "NONE",
     "expected_parameters": None},
    {"query": "How do I setup a new sim card?", "chat_history": "", "expected_tool": "NONE",
     "expected_parameters": None}, ]


@pytest.fixture(scope="module")
def agent():
    return ToolAgent()


@pytest.mark.parametrize("row", test_data)
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

        if expected_params and isinstance(expected_params, dict):
            for k, v in expected_params.items():
                assert call["args"].get(k) == v

import random
from langchain_core.tools import tool


@tool
def check_data_balance(mobile_number: str):
    """
    Retrieves the current data balance and validity for a given mobile number.
    Use this when a user asks about their data, gb remaining, or balance.
    """
    if "077" not in mobile_number:
        return {"error": "Invalid Number. Must start with 077."}

    return {
        "mobile": mobile_number,
        "balance_data": "12.5 GB",
        "balance_voice": "450 Mins",
        "validity": "2026-12-31",
        "plan": "Unlimited Web Plus"
    }


@tool
def verify_identity(mobile_number: str, nic_last_4: str):
    """
    Verifies a customer's identity using their Mobile Number and last 4 digits of NIC.
    Use this before performing sensitive actions like SIM replacement.
    """
    if nic_last_4 == "1234":
        return {"status": "verified", "name": "John Doe", "segment": "Gold"}
    else:
        return {"status": "failed", "reason": "NIC mismatch"}


# List of tools to export
crm_tools = [check_data_balance, verify_identity]
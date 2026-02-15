import random
from langchain_core.tools import tool
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def check_data_balance(mobile_number: str):
    """
    Retrieves the current data balance and validity for a given mobile number.
    Use this when a user asks about their data, gb remaining, or balance.
    """
    logger.info(f"Checking data balance for mobile number: {mobile_number}")
    if "077" not in mobile_number:
        logger.warning(f"Validation failed for mobile number: {mobile_number}")
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
    logger.info(f"Verifying identity for mobile number: {mobile_number}")
    if nic_last_4 == "1234":
        logger.info(f"Identity verification successful for {mobile_number}")
        return {"status": "verified", "name": "John Doe", "segment": "Gold"}
    else:
        logger.warning(f"Identity verification failed for {mobile_number}. NIC mismatch.")
        return {"status": "failed", "reason": "NIC mismatch"}


# List of tools to export
crm_tools = [check_data_balance, verify_identity]
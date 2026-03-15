import json
import os
from typing import Any, Dict

import requests
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

TOOLS_FILE = os.path.join(os.path.dirname(__file__), "registered_tools.json")


def execute_api_call(api_url: str, http_method: str, **kwargs) -> Dict[str, Any]:
    logger.info(f"Executing dynamic tool at {api_url} with args: {kwargs}")
    try:
        if http_method.upper() == "GET":
            response = requests.get(api_url, params=kwargs)
        else:
            response = requests.post(api_url, json=kwargs)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {str(e)}")
        return {"error": f"Failed to execute action. API returned: {str(e)}"}


def get_dynamic_tools() -> list[StructuredTool]:
    tools = []
    if not os.path.exists(TOOLS_FILE):
        logger.warning(f"No dynamic tools found at {TOOLS_FILE}")
        return tools

    with open(TOOLS_FILE, "r", encoding="utf-8") as f:
        try:
            registered_tools = json.load(f)
        except json.JSONDecodeError:
            logger.error("Failed to parse registered_tools.json")
            return tools

    type_mapping = {"string": str, "integer": int, "boolean": bool, "number": float}

    for config in registered_tools:
        tool_name = config["tool_name"]
        description = config["description"]
        api_url = config["api_url"]
        http_method = config.get("http_method", "POST")

        fields = {}
        params_schema = config.get("parameters", {})
        properties = params_schema.get("properties", {})
        required_fields = params_schema.get("required", [])

        for param_name, param_info in properties.items():
            param_type_str = param_info.get("type", "string")
            python_type = type_mapping.get(param_type_str, str)
            param_desc = param_info.get("description", "")

            if param_name in required_fields:
                fields[param_name] = (python_type, Field(..., description=param_desc))
            else:
                fields[param_name] = (python_type, Field(None, description=param_desc))

        args_schema = create_model(f"{tool_name}Schema", **fields)

        def create_tool_func(url, method):
            def func(**kwargs):
                return execute_api_call(url, method, **kwargs)

            return func

        dynamic_tool = StructuredTool.from_function(
            func=create_tool_func(api_url, http_method),
            name=tool_name,
            description=description,
            args_schema=args_schema
        )
        tools.append(dynamic_tool)

    logger.info(f"Successfully loaded {len(tools)} dynamic tools.")
    return tools

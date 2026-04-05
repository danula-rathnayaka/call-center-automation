import json
import os
from typing import Any, Dict

import requests
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

TOOLS_FILE = os.path.join(os.path.dirname(__file__), "registered_tools.json")

_tools_cache: list = []
_cache_mtime: float = 0.0


def execute_api_call(api_url: str, http_method: str, **kwargs) -> Dict[str, Any]:
    logger.info(f"Executing dynamic tool at {api_url} with args: {kwargs}")
    try:
        if http_method.upper() == "GET":
            response = requests.get(api_url, params=kwargs, timeout=10)
        else:
            response = requests.post(api_url, json=kwargs, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": f"Failed to execute action. API returned: {str(e)}"}


def get_dynamic_tools() -> list[StructuredTool]:
    global _tools_cache, _cache_mtime

    if not os.path.exists(TOOLS_FILE):
        logger.warning(f"No dynamic tools found at {TOOLS_FILE}")
        return []

    try:
        current_mtime = os.path.getmtime(TOOLS_FILE)
    except OSError:
        return _tools_cache

    if current_mtime == _cache_mtime and _tools_cache:
        return _tools_cache

    try:
        with open(TOOLS_FILE, "r", encoding="utf-8") as f:
            registered_tools = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to parse registered_tools.json: {e}")
        return _tools_cache

    type_mapping = {"string": str, "integer": int, "boolean": bool, "number": float}
    tools = []

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
            python_type = type_mapping.get(param_info.get("type", "string"), str)
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

        tools.append(StructuredTool.from_function(func=create_tool_func(api_url, http_method), name=tool_name,
            description=description, args_schema=args_schema, ))

    _tools_cache = tools
    _cache_mtime = current_mtime
    logger.info(f"Loaded {len(tools)} dynamic tools (cache refreshed)")
    return tools

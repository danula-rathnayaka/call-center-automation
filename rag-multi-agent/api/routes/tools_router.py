import json
import logging
import os

from api.schemas import ToolRegistrationRequest
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Dynamic Tools"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "src", "multiagent_rag", "tools"))
TOOLS_FILE = os.path.join(TOOLS_DIR, "registered_tools.json")

os.makedirs(TOOLS_DIR, exist_ok=True)


@router.post("")
async def register_tool(tool_req: ToolRegistrationRequest):
    try:
        existing_tools = []
        if os.path.exists(TOOLS_FILE):
            with open(TOOLS_FILE, "r", encoding="utf-8") as f:
                try:
                    existing_tools = json.load(f)
                except json.JSONDecodeError:
                    existing_tools = []

        properties = {}
        required = []
        for param in tool_req.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            required.append(param.name)

        new_tool_config = {
            "tool_name": tool_req.tool_name,
            "description": tool_req.description,
            "api_url": tool_req.api_url,
            "http_method": tool_req.http_method,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

        existing_tools = [t for t in existing_tools if t.get("tool_name") != tool_req.tool_name]
        existing_tools.append(new_tool_config)

        with open(TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_tools, f, indent=4)

        return {
            "status": "success",
            "message": f"Tool '{tool_req.tool_name}' registered successfully.",
            "tool": new_tool_config
        }

    except Exception as e:
        logger.error(f"Error registering tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to register tool: {str(e)}")


@router.get("")
async def get_registered_tools():
    if not os.path.exists(TOOLS_FILE):
        return []
    with open(TOOLS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

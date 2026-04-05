import json
import logging
import os
import threading

from api.schemas import ToolRegistrationRequest, ToolListResponse, ToolEntry, DeleteToolResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["Dynamic Tools"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "src", "multiagent_rag", "tools"))
TOOLS_FILE = os.path.join(TOOLS_DIR, "registered_tools.json")
os.makedirs(TOOLS_DIR, exist_ok=True)

_write_lock = threading.Lock()


def _load_tools() -> list:
    if not os.path.exists(TOOLS_FILE):
        return []
    with open(TOOLS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_tools(tools: list) -> None:
    with open(TOOLS_FILE, "w", encoding="utf-8") as f:
        json.dump(tools, f, indent=4)


@router.post(
    "",
    summary="Register or update a dynamic CRM tool",
    description=(
        "Registers a new tool that the AI agent can call when handling customer service queries. "
        "If a tool with the same `tool_name` already exists it is replaced (upsert behaviour). "
        "Provide the tool name, a description the LLM will use to decide when to invoke it, "
        "the backend API URL the tool calls, the HTTP method (`GET` or `POST`), "
        "and the list of parameters with name, type, and description. "
        "The change takes effect on the next agent invocation — no server restart needed."
    ),
)
async def register_tool(tool_req: ToolRegistrationRequest):
    try:
        with _write_lock:
            existing_tools = _load_tools()

            properties = {}
            required = []
            for param in tool_req.parameters:
                properties[param.name] = {"type": param.type, "description": param.description}
                required.append(param.name)

            new_tool_config = {
                "tool_name": tool_req.tool_name,
                "description": tool_req.description,
                "api_url": tool_req.api_url,
                "http_method": tool_req.http_method,
                "parameters": {"type": "object", "properties": properties, "required": required},
            }

            existing_tools = [t for t in existing_tools if t.get("tool_name") != tool_req.tool_name]
            existing_tools.append(new_tool_config)
            _save_tools(existing_tools)

        return {
            "status": "success",
            "message": f"Tool '{tool_req.tool_name}' registered successfully.",
            "tool": new_tool_config,
        }
    except Exception as e:
        logger.error(f"Error registering tool: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register tool: {str(e)}")


@router.get(
    "",
    response_model=ToolListResponse,
    summary="List all registered dynamic CRM tools",
    description=(
        "Returns every tool currently loaded from `registered_tools.json`. "
        "Each entry includes the tool name, description, API URL, and HTTP method. "
        "Use this in the admin panel to show which tools the AI agent has access to, "
        "and as the testing_dataset source for a tool management table where admins can review and delete tools."
    ),
)
async def list_tools():
    """Return all dynamically registered CRM tools."""
    tools = _load_tools()
    entries = [
        ToolEntry(
            tool_name=t.get("tool_name", ""),
            description=t.get("description", ""),
            api_url=t.get("api_url", ""),
            http_method=t.get("http_method", "POST"),
        )
        for t in tools
    ]
    return ToolListResponse(total=len(entries), tools=entries)


@router.delete(
    "/{tool_name}",
    response_model=DeleteToolResponse,
    summary="Remove a registered CRM tool by name",
    description=(
        "Deletes the named tool from `registered_tools.json` and immediately invalidates the in-memory "
        "tool cache so the AI agent stops using it without a server restart. "
        "Returns `404` if no tool with that name exists. "
        "Use this when decommissioning a CRM API endpoint or removing tools that are no longer needed."
    ),
)
async def delete_tool(tool_name: str):
    """Remove a registered tool by name. Takes effect immediately — no server restart needed."""
    try:
        with _write_lock:
            existing_tools = _load_tools()
            filtered = [t for t in existing_tools if t.get("tool_name") != tool_name]

            if len(filtered) == len(existing_tools):
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool '{tool_name}' not found in registered tools."
                )

            _save_tools(filtered)
            logger.info(f"Tool '{tool_name}' deleted from registered_tools.json")

        try:
            from multiagent_rag.tools import crm_tools as _crm
            _crm._cache_mtime = 0.0
            _crm._tools_cache = []
        except Exception:
            pass

        return DeleteToolResponse(
            status="deleted",
            tool_name=tool_name,
            message=f"Tool '{tool_name}' has been removed and will no longer be available to the agent.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tool '{tool_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool: {str(e)}")

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .contracts import Contracts


class MCPToolError(RuntimeError):
    """The gateway answered the call with an error result (not a transport failure)."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"MCP tool {tool_name} failed: {message or 'unknown error'}")
        self.tool_name = tool_name
        self.message = message

    @property
    def not_found(self) -> bool:
        lowered = self.message.lower()
        return "not found" in lowered or "not_found" in lowered or "404" in lowered

    @property
    def forbidden(self) -> bool:
        lowered = self.message.lower()
        return "forbidden" in lowered or "403" in lowered or "scope" in lowered


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    properties: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict, compare=False)


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


class EvidenceGateway:
    def __init__(self, session: ClientSession, contracts: Contracts) -> None:
        self._session = session
        self._contracts = contracts
        self._specs: dict[str, ToolSpec] | None = None

    async def list_tools(self) -> list[str]:
        return sorted(await self.tool_specs())

    async def tool_specs(self) -> dict[str, ToolSpec]:
        if self._specs is None:
            response = await self._session.list_tools()
            specs: dict[str, ToolSpec] = {}
            for tool in response.tools:
                schema = _attr(tool, "input_schema", "inputSchema", default={}) or {}
                specs[tool.name] = ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    properties=tuple((schema.get("properties") or {}).keys()),
                    required=tuple(schema.get("required") or ()),
                    input_schema=schema,
                )
            self._specs = specs
        return self._specs

    async def call(self, tool_name: str, *, case_id: str, **arguments: Any) -> dict[str, Any]:
        payload = {"case_id": case_id, **arguments}
        result = await self._session.call_tool(tool_name, arguments=payload)
        content = _attr(result, "content", default=[]) or []
        if _attr(result, "is_error", "isError", default=False):
            message = " ".join(block.text for block in content if getattr(block, "text", None))
            raise MCPToolError(tool_name, message)
        evidence = _attr(result, "structured_content", "structuredContent")
        if evidence is None:
            text_blocks = [block.text for block in content if getattr(block, "text", None)]
            if len(text_blocks) != 1:
                raise ValueError(f"MCP tool {tool_name} did not return one evidence object")
            evidence = json.loads(text_blocks[0])
        self._contracts.validate_evidence(evidence, f"MCP tool {tool_name}")
        return evidence


@asynccontextmanager
async def connect_gateway(
    endpoint: str, team_api_key: str, contracts: Contracts
) -> AsyncIterator[EvidenceGateway]:
    headers = {"Authorization": f"Bearer {team_api_key}"}
    timeout = httpx2.Timeout(300.0, connect=30.0, write=30.0, pool=30.0)
    async with (
        httpx2.AsyncClient(headers=headers, timeout=timeout) as http_client,
        streamable_http_client(endpoint, http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield EvidenceGateway(session, contracts)

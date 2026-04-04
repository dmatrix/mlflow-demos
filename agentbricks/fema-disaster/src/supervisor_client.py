"""Backend client for querying the AgentBricks Supervisor endpoint.

Adapted from src/04_query_supervisor.py to work outside Databricks notebook
context — uses WorkspaceClient for auth (OAuth in Databricks Apps, env vars locally).
"""

import os
import requests
from databricks.sdk import WorkspaceClient


def get_workspace_client() -> WorkspaceClient:
    """Return a WorkspaceClient using the SDK's credential resolution chain."""
    return WorkspaceClient()


def query_supervisor(client: WorkspaceClient, endpoint_name: str, query: str) -> dict:
    """Query the AgentBricks Supervisor endpoint.

    Returns:
        {"answer": str, "raw_response": dict}
    """
    host = client.config.host.rstrip("/")
    headers = {
        "Authorization": f"Bearer {client.config.token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{host}/serving-endpoints/{endpoint_name}/invocations",
        headers=headers,
        json={"input": [{"role": "user", "content": query}]},
    )
    response.raise_for_status()
    result = response.json()

    answer = _extract_answer(result)
    return {"answer": answer, "raw_response": result}


def _extract_answer(result: dict) -> str:
    """Extract the final assistant text from the supervisor response."""
    if "output" in result and isinstance(result["output"], list):
        for msg in reversed(result["output"]):
            if msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                if isinstance(content, list):
                    texts = [
                        c["text"]
                        for c in content
                        if c.get("type") == "output_text" and c.get("text")
                    ]
                    if texts:
                        return "\n".join(texts)
                elif isinstance(content, str) and content:
                    return content

    if "choices" in result and result["choices"]:
        return result["choices"][0]["message"]["content"]

    return str(result)

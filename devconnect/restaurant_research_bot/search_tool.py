"""
Web search tool for the multi-turn conversational bot.

Provides a Tavily-backed web_search() function and the OpenAI function-calling
schema the LLM uses to invoke it.

IMPORTANT: Tavily (and all search APIs) are stateless -- each call receives only
the query string. The LLM must construct complete, self-contained queries that
bake in any relevant context from prior conversation turns.
"""

import os
import requests
import mlflow
from mlflow.entities import SpanType


# OpenAI function-calling schema -- passed to the LLM in every chat request.
# The description tells the LLM when to use the tool; the parameter description
# reminds it to write complete, context-aware queries.
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this when you need "
            "facts you don't know or that may have changed: restaurant menus, "
            "business hours, food safety ratings, nutrition info, current news."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Must be specific and self-contained. "
                        "Include all relevant context from the conversation "
                        "(e.g. allergen constraints, location, restaurant name) "
                        "because this API has no memory of prior turns."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


@mlflow.trace(span_type=SpanType.TOOL, name="web_search")
def web_search(query: str, max_results: int = 3) -> str:
    """
    Execute a web search via Tavily and return a formatted string of results.

    Decorated with @mlflow.trace so each search call appears as a child TOOL span
    inside the parent handle_message() CHAT_MODEL span in MLflow.

    Args:
        query: Search query. Should be complete and self-contained.
        max_results: Maximum number of results to return (default: 3).

    Returns:
        Formatted string with a Tavily summary (if available) and top results.

    Raises:
        requests.HTTPError: If the Tavily API returns an error.
        KeyError: If TAVILY_API_KEY is not set in the environment.
    """
    api_key = os.environ["TAVILY_API_KEY"]
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}")
    for r in data.get("results", [])[:max_results]:
        parts.append(f"- [{r['title']}]({r['url']})\n  {r['content'][:300]}")
    return "\n\n".join(parts) if parts else "No results found."

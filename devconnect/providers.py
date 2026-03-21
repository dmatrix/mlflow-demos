"""
LLM client factory for the devconnect demo agents.
"""

import os
from openai import OpenAI


def get_client(provider: str, **kwargs) -> OpenAI:
    """
    Return an OpenAI-compatible client for the given provider.

    Args:
        provider: "openai" or "databricks"
        **kwargs: api_key / token / host overrides (fall back to env vars)
    """
    if provider == "databricks":
        token = kwargs.get("token") or os.environ["DATABRICKS_TOKEN"]
        host  = kwargs.get("host")  or os.environ["DATABRICKS_HOST"]
        return OpenAI(
            api_key=token,
            base_url=f"{host}/serving-endpoints",
        )
    if provider == "openai":
        api_key = kwargs.get("api_key") or os.environ["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)

    raise ValueError(f"Unknown provider: {provider!r}. Use 'openai' or 'databricks'.")

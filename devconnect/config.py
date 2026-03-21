"""
Agent configuration for the devconnect demo agents.
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any


@dataclass
class AgentConfig:
    model: str
    provider: Literal["openai", "databricks"] = "openai"
    mlflow_experiment: str = "web-search-bot"
    temperature: float = 0.2
    api_key: Optional[str] = None
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None

    def __post_init__(self):
        if self.provider == "databricks":
            if self.databricks_host is None:
                self.databricks_host = os.environ.get("DATABRICKS_HOST")
            if self.databricks_token is None:
                self.databricks_token = os.environ.get("DATABRICKS_TOKEN")
        if self.provider == "openai" and self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    def get_provider_kwargs(self) -> Dict[str, Any]:
        if self.provider == "openai":
            return {"api_key": self.api_key} if self.api_key else {}
        return {
            k: v for k, v in {
                "token": self.databricks_token,
                "host":  self.databricks_host,
            }.items() if v
        }

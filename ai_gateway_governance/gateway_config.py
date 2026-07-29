"""Unity AI Gateway v2 configuration helpers.

The v2 AI Gateway is configured through the Databricks UI. This module
provides helpers to verify connectivity and display the expected config.
"""

import time
from dataclasses import dataclass, field

import requests as http_requests

# Retry transient rate-limit / server / guardrail-backend failures. A real
# guardrail block is 400 and is never retried.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


@dataclass
class GatewayConfig:
    endpoint_name: str
    models: list[str]
    catalog_name: str
    schema_name: str
    table_name_prefix: str = "coding_agents"

    pii_behavior: str = "BLOCK"
    safety_enabled: bool = True
    invalid_keywords: list = field(default_factory=list)
    valid_topics: list = field(default_factory=list)

    inference_table_enabled: bool = True
    usage_tracking_enabled: bool = True


def verify_gateway(host: str, token: str, endpoint_name: str) -> dict:
    """Send a lightweight request to verify the guarded serving endpoint is reachable.

    Guardrails are attached to the serving endpoint, so we hit its
    /serving-endpoints/<endpoint>/invocations URL. The endpoint fronts a single
    model, so the request body carries no `model` field.
    """
    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
    backoff = INITIAL_BACKOFF
    resp = None
    for attempt in range(MAX_RETRIES):
        resp = http_requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"messages": [{"role": "user", "content": "Say ok"}], "max_tokens": 5},
            timeout=30,
        )
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
            time.sleep(backoff)
            backoff *= 2
            continue
        break
    return {"status": resp.status_code, "reachable": resp.status_code == 200}


def print_gateway_summary(config: GatewayConfig, host: str, token: str) -> None:
    """Verify gateway connectivity and display the expected configuration."""
    result = verify_gateway(host, token, config.endpoint_name)

    print(f"{'=' * 60}")
    print(f"  AI Gateway Configuration: {config.endpoint_name}")
    print(f"{'=' * 60}")

    status = "CONNECTED" if result["reachable"] else f"ERROR (HTTP {result['status']})"
    print(f"\n  Gateway Status:   {status}")
    print(f"  Gateway URL:      {host.rstrip('/')}/serving-endpoints/{config.endpoint_name}/invocations")
    print(f"  Route:            {config.endpoint_name}")
    print(f"  Models:           {', '.join(config.models)}")

    print("\n  Guardrails (configured via UI):")
    print(f"    PII:              {config.pii_behavior}")
    print(f"    Safety:           {config.safety_enabled}")
    if config.invalid_keywords:
        print(f"    Invalid Keywords: {config.invalid_keywords}")
    if config.valid_topics:
        print(f"    Valid Topics:     {config.valid_topics}")

    if config.inference_table_enabled:
        print("\n  Inference Tables:")
        print(f"    Enabled:  {config.inference_table_enabled}")
        print(f"    Location: {config.catalog_name}.{config.schema_name}.{config.table_name_prefix}*")

    print("\n  Usage Tracking:")
    print(f"    Enabled:  {config.usage_tracking_enabled}")

    print(f"\n{'=' * 60}")

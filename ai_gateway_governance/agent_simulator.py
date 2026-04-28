"""Simulate coding agents sending requests through Unity AI Gateway."""

import time
from dataclasses import dataclass

import mlflow
import requests

MAX_RETRIES = 5
INITIAL_BACKOFF = 2


@dataclass
class SimulatedAgent:
    name: str
    display_name: str
    system_prompt: str


@dataclass
class GatewayClient:
    url: str
    token: str


def create_gateway_client(host: str, token: str) -> GatewayClient:
    """Create a client pointing at the v2 Unity AI Gateway."""
    url = f"{host.rstrip('/')}/ai-gateway/mlflow/v1/chat/completions"
    return GatewayClient(url=url, token=token)


@mlflow.trace(span_type="CHAT_MODEL", name="gateway_request")
def send_request(
    client: GatewayClient,
    agent: SimulatedAgent,
    messages: list[dict],
    model: str,
) -> dict:
    """Send a chat completion through the gateway and return a result dict."""
    full_messages = [{"role": "system", "content": agent.system_prompt}] + messages

    mlflow.update_current_trace(tags={"agent": agent.name})

    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                client.url,
                headers={"Authorization": f"Bearer {client.token}"},
                json={"model": model, "messages": full_messages, "max_tokens": 1024},
                timeout=120,
            )

            if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code != 200:
                return {
                    "agent": agent.display_name,
                    "status": resp.status_code,
                    "content": None,
                    "tokens": None,
                    "error": resp.text[:300],
                }

            data = resp.json()
            usage = data.get("usage", {})
            return {
                "agent": agent.display_name,
                "status": 200,
                "content": data["choices"][0]["message"]["content"],
                "tokens": {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
                "error": None,
            }
        except Exception as e:
            return {
                "agent": agent.display_name,
                "status": 500,
                "content": None,
                "tokens": None,
                "error": str(e),
            }


def run_scenario(
    client: GatewayClient,
    agent: SimulatedAgent,
    scenario: dict,
    model: str,
) -> dict:
    """Run a single scenario and return the result with scenario metadata."""
    result = send_request(client, agent, scenario["messages"], model)
    result["scenario"] = scenario["name"]
    result["description"] = scenario["description"]
    result["expected_outcome"] = scenario["expected_outcome"]
    result["guardrail_type"] = scenario["guardrail_type"]

    actual = "blocked" if result["status"] != 200 else "allowed"
    result["actual_outcome"] = actual
    result["pass"] = actual == scenario["expected_outcome"]
    return result


def print_result(result: dict) -> None:
    """Pretty-print a single scenario result."""
    passed = result["pass"]
    status_icon = "PASS" if passed else "FAIL"
    outcome_icon = "BLOCKED" if result["actual_outcome"] == "blocked" else "ALLOWED"

    print(f"  [{status_icon}] {result['description']}")
    print(f"    Agent:    {result['agent']}")
    print(f"    Status:   {result['status']} ({outcome_icon})")

    if result["actual_outcome"] == "allowed" and result["tokens"]:
        t = result["tokens"]
        print(f"    Tokens:   {t['total']} (in: {t['input']}, out: {t['output']})")

    if result.get("content"):
        print("-------------------------------- RESPONSE --------------------------------")
        preview = result["content"][:750]
        if len(result["content"]) > 750:
            preview += "..."
        print(f"    Response: {preview}")
        print("-------------------------------- RESPONSE --------------------------------")

    if result["error"]:
        error_preview = result["error"][:250]
        print(f"    Message:  {error_preview}")

    print()

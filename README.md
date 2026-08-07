# MLflow Demos

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow 3.15](https://img.shields.io/badge/mlflow-3.15-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-7C3AED?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![OpenAI](https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white)](https://openai.com)
[![Tavily](https://img.shields.io/badge/Tavily-web%20search-00B4D8)](https://tavily.com)

Demo agents and notebooks for [MLflow](https://mlflow.org) GenAI evaluation, built for DevConnect and AI conferences.

## What's here

### `agentbricks/fema-disaster`

A Databricks Asset Bundle that deploys a multi-agent supervisor for FEMA disaster response, using only Databricks-native components. A supervisor agent routes queries to either a Genie Space for structured data (disaster counts, federal aid, severity trends) or a Knowledge Assistant backed by Vector Search for policy retrieval (evacuation protocols, aid eligibility, safety guidelines). Includes MLflow GenAI evaluation with built-in scorers and per-judge assessments.

Deploys to serverless compute with two `databricks bundle` commands. See the [full README](agentbricks/fema-disaster/README.md).

### `devconnect/restaurant_research_bot`

A multi-turn agent that researches restaurants with live web search, evaluated by MLflow session-level judges. Built for caterers, Caspers Kitchens clients, and anyone researching restaurants for:

* **Food allergies** — dishes and restaurants that accommodate dietary restrictions (peanut-free, gluten-free, vegan)
* **Ratings and recommendations** — highly rated restaurants by neighborhood, cuisine, or preference
* **Safety inspections** — health inspection scores and recent violations for a specific restaurant
* **Menus and hours** — current hours, menus, and allergen-friendly options
* **Personalized advice** — synthesized across turns, with the agent remembering preferences as the conversation goes

Three session-level judges score it:

| Judge | Measures |
|---|---|
| `conversation_coherence` | Does the conversation flow logically across turns? |
| `context_retention` | Does the agent remember prior constraints (allergies, location, preferences)? |
| `search_quality` | Did the agent search when needed and skip when it wasn't? |

Runs as a CLI or notebook, against OpenAI or Databricks-hosted models. See the [full README](devconnect/README.md) for credentials, run modes, and the four scenarios.

### `unity_ai_gateway_governance/`

A notebook demo of [Unity AI Gateway](https://www.databricks.com/blog/governing-coding-agent-sprawl-unity-ai-gateway) as centralized governance for coding agents. Five simulated agents — Cursor, Claude Code, Codex CLI, Gemini CLI, Pi — each with its own persona prompt, send 10 coding requests apiece across three provider-specific governed model services (Claude, OpenAI, Gemini). Each service is a Unity Catalog securable with its own guardrails, inference table, and rate limits; all three are reached through one gateway URL.

* **PII detection** — blocks SSNs, credit cards, emails, and phone numbers. A block returns HTTP 200 with a denying `databricks_service_policy` naming the policy and what it found
* **Safety and prompt-injection filters** — block jailbreaks and malware requests. Defense in depth: borderline content the gateway allows through, the model still refuses
* **Inference tables** — requests logged to Delta, one table per service, for audit and cost queries via Genie
* **Usage tracking** — per-provider cost and token attribution
* **Rate limiting** — per-service QPM and TPM ceilings, returning HTTP 429 without reaching the model

Needs a Databricks workspace with Unity Catalog and three AI Gateway model services. See the [full README](unity_ai_gateway_governance/README.md).

## Quickstart

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An OpenAI API key
- A [Tavily](https://tavily.com) API key, for web search and devconnect and resturant bot demo
- MLflow 3.14+

### Install

```bash
uv sync
```

### Set up a demo

Each demo owns its own setup — credentials, run modes, and walkthrough:

- **`devconnect/restaurant_research_bot`** — [devconnect/README.md](devconnect/README.md)
- **`unity_ai_gateway_governance/`** — [unity_ai_gateway_governance/README.md](unity_ai_gateway_governance/README.md)
- **`agentbricks/fema-disaster`** — [agentbricks/fema-disaster/README.md](agentbricks/fema-disaster/README.md)

## Stack

- **MLflow 3.15** — experiment tracking, tracing, and `mlflow.genai.evaluate()`
- **OpenAI / Databricks** — agent and judge LLMs, switchable via `--provider`
- **Tavily** — real-time web search
- **LiteLLM** — provider abstraction

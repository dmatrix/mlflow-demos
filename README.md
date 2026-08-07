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
- A [Tavily](https://tavily.com) API key, for web search

### Install

```bash
uv sync
```

### Configure credentials

Copy the `env-template` at the repo root into `devconnect/restaurant_research_bot/` as `.env`:

```bash
cp env-template devconnect/restaurant_research_bot/.env
```

Then fill in:

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
OPENAI_API_BASE=https://api.openai.com/v1
MLFLOW_TRACKING_URI=http://localhost:5000

# Databricks (only if using --provider databricks)
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
```

### Run the CLI

```bash
# Start the MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# All scenarios, with OpenAI
uv run mlflow-restaurant-research-bot

# One scenario
uv run mlflow-restaurant-research-bot --scenario allergen

# Databricks-hosted models
uv run mlflow-restaurant-research-bot \
  --provider databricks \
  --model databricks-gpt-5-mini
```

### Run the notebook

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000   # tracking server at http://localhost:5000
jupyter notebook devconnect/restaurant_research_bot/restaurant_research_agent_devconnect.ipynb
```

### Databricks

Set these instead of an OpenAI key:

```bash
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
TAVILY_API_KEY=tvly-...
```

## Scenarios

| Scenario | Key challenge |
|---|---|
| `restaurant` | Multi-turn discovery; turn 4 synthesizes without re-searching |
| `safety` | Resolves implicit references ("that restaurant") into concrete search queries |
| `allergen` | Carries a peanut allergy into a later search without being reminded |
| `nosearch` | Stays within general knowledge for all four turns; correct behavior is zero searches |

## Stack

- **MLflow 3.15** — experiment tracking, tracing, and `mlflow.genai.evaluate()`
- **OpenAI / Databricks** — agent and judge LLMs, switchable via `--provider`
- **Tavily** — real-time web search
- **LiteLLM** — provider abstraction

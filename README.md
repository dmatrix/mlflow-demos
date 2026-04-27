# MLflow Demos

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow 3.10](https://img.shields.io/badge/mlflow-3.10-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-7C3AED?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![OpenAI](https://img.shields.io/badge/OpenAI-compatible-412991?logo=openai&logoColor=white)](https://openai.com)
[![Tavily](https://img.shields.io/badge/Tavily-web%20search-00B4D8)](https://tavily.com)

Demo agents and notebooks showcasing [MLflow](https://mlflow.org) GenAI evaluation capabilities, built for DevConnect and AI Conferences

## What's Here

### `agentbricks/fema-disaster`

A **Databricks Asset Bundle** that deploys a multi-agent supervisor for FEMA disaster response using 100% Databricks-native components. A Supervisor Agent routes queries to a **Genie Space** (structured data: disaster counts, federal aid, severity trends) and a **Knowledge Assistant** (policy retrieval: evacuation protocols, aid eligibility, safety guidelines) backed by Vector Search. Includes MLflow GenAI evaluation with built-in scorers and individual judge assessments.

Deploys to serverless compute with two `databricks bundle` commands. See the [full README](agentbricks/fema-disaster/README.md) for setup instructions.

### `devconnect/restaurant_research_bot`

A multi-turn conversational agent that researches restaurants using live web search, evaluated with MLflow session-level judges.

This agent can be used by caterers, **Caspers Kitchens** clients or customers, and anyone interested in researching restaurants for the following scenarios:

* **Food allergies** — identify dishes and restaurants that accommodate specific dietary restrictions (e.g. peanut-free, gluten-free, vegan)
* **Restaurant ratings & recommendations** — discover highly rated restaurants by neighborhood, cuisine, or preference
* **Food safety inspections** — look up health inspection scores and recent violation reports for a specific restaurant
* **Menu & hours** — find current operating hours, menus, and vegetarian or allergen-friendly options
* **Personalized recommendations** — get synthesized advice across multiple turns, with the agent remembering your preferences throughout the conversation

The agent is evaluated along **three dimensions** using MLflow session-level judges:

| Judge | Measures |
|---|---|
| `conversation_coherence` | Does the conversation flow logically across turns? |
| `context_retention` | Does the agent remember prior constraints (allergies, location, preferences)? |
| `search_quality` | Did the agent search when needed and skip when it wasn't? |

### `ai_gateway_governance/`

A notebook-driven demo showing how [Unity AI Gateway](https://www.databricks.com/blog/governing-coding-agent-sprawl-unity-ai-gateway) provides centralized governance for coding agents. Simulates four agents (Cursor, Claude Code, Codex CLI, Gemini CLI) sending requests through a single gateway endpoint and demonstrates:

* **PII detection** — blocks requests containing SSNs, credit cards, etc.
* **Safety filters** — blocks prompt injection and harmful content
* **Inference tables** — all requests (including blocked) logged to Delta in Unity Catalog
* **Usage tracking** — unified cost and token tracking across agents

Requires a Databricks workspace with a serving endpoint and Unity Catalog. See the [full README](ai_gateway_governance/README.md) for setup instructions.

## Quickstart

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An OpenAI API key
- A [Tavily](https://tavily.com) API key (for web search)

### Install

```bash
uv sync
```

### Configure credentials

An `env-template` file is included at the repo root with all required variables. Copy it into `devconnect/restaurant_research_bot/` and rename it `.env`:

```bash
cp env-template devconnect/restaurant_research_bot/.env
```

Then fill in your values:

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
OPENAI_API_BASE=https://api.openai.com/v1

# Databricks host and token (only if running from local host)
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
DATABRICKS_TOKEN=dapi...

#
AI_GATEWAY_ENDPOINT_NAME=<ai-gateway-endpoint-name>
AI_GATEWAY_MODEL="databricks-model-name"
UC_CATALOG="your_catalog_name"
UC_SCHEMA="your_schema_name"
```

### Run the CLI

```bash
# start the MLFlwo tracking server 
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
# Default: runs all scenarios with OpenAI
uv run mlflow-restaurant-research-bot

# Run a specific scenario
uv run mlflow-restaurant-research-bot --scenario allergen

# Run with Databricks-hosted models
uv run mlflow-restaurant-research-bot \
  --provider databricks \
  --model databricks-gpt-5-mini
```

### Run the notebook

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000        # start the tracking server at http://localhost:5000
jupyter notebook devconnect/restaurant_research_bot/restaurant_research_agent_devconnect.ipynb
```

### Databricks

Set these instead of an OpenAI key:

```bash
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=dapi...
TAVILY_API_KEY=tvly-...
```

## Available Scenarios

| Scenario | Key challenge |
|---|---|
| `restaurant` | Multi-turn restaurant discovery; turn 4 synthesises without re-searching |
| `safety` | Resolves implicit references ("that restaurant") into concrete search queries |
| `allergen` | Silently carries a peanut allergy constraint into a later search without being told to |
| `nosearch` | Stays within general knowledge for all four turns; correct behaviour = zero searches |

## Stack

- **MLflow 3.10** — experiment tracking, tracing, and `mlflow.genai.evaluate()`
- **OpenAI / Databricks** — agent and judge LLMs (switchable via `--provider`)
- **Tavily** — real-time web search tool
- **LiteLLM** — provider abstraction layer

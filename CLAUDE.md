# CLAUDE.md

## Project overview

Demo agents and notebooks showcasing MLflow GenAI evaluation and Databricks governance capabilities. Contains two main demos: a multi-turn restaurant research agent (`devconnect/resturant_research_bot`) evaluated with MLflow session-level judges, and an AI Gateway governance demo (`ai_gateway_governance/`) showing centralized guardrails for coding agents.

## Layout

```
devconnect/
  config.py                        # AgentConfig dataclass (provider, model, temperature)
  mlflow_config.py                 # setup_mlflow_tracking() helper
  providers.py                     # LiteLLM-based provider abstraction
  resturant_research_bot/
    resturant_research_agent_cls.py  # ResturantResearchAgent — core agent class
    resturant_research_agent.py      # CLI entry point (argparse → agent)
    resturant_research_agent_devconnect.ipynb  # demo notebook
    scenarios.py                     # get_scenario_*() functions
    prompts.py                       # system prompt + three judge instruction strings
    search_tool.py                   # Tavily web_search() tool wrapper

ai_gateway_governance/
  ai_gateway_demo.ipynb            # Main demo notebook (4 acts)
  gateway_config.py                # GatewayConfig dataclass + SDK helpers for AI Gateway
  agent_simulator.py               # SimulatedAgent, GatewayClient, request handling with retry
  scenarios.py                     # Test payloads: clean requests, PII, prompt injection
  prompts.py                       # System prompts for each coding agent persona
  observability.py                 # SQL query templates for inference tables
  images/
    ai_gateway_architecture.svg    # Architecture diagram
```

## Running the demo

```bash
# Install
uv sync

# Start MLflow tracking server (required)
mlflow ui   # → http://localhost:5000

# Run CLI (OpenAI)
uv run mlflow-resturant-research-bot

# Run a specific scenario
uv run mlflow-resturant-research-bot --scenario allergen

# Run with Databricks
uv run mlflow-resturant-research-bot --provider databricks --model databricks-gpt-5-mini
```

## Environment variables

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | OpenAI provider (default) |
| `TAVILY_API_KEY` | Web search (all providers) |
| `DATABRICKS_HOST` | Databricks provider |
| `DATABRICKS_TOKEN` | Databricks provider |

Place these in `devconnect/resturant_research_bot/.env` — loaded automatically via `python-dotenv`.

## Key concepts

**Session-level evaluation:** All three judges use the `{{ conversation }}` template, which tells `mlflow.genai.evaluate()` to aggregate all turns in a session before scoring — not evaluate turn by turn.

**Stateless search:** `web_search()` (Tavily) receives only the query string with no conversation history. The agent must construct self-contained queries that carry prior context (e.g. resolving "that restaurant" → "Nopa San Francisco").

**Three judges** (defined in `prompts.py`, instantiated in `ResturantResearchAgent.__init__`):
- `conversation_coherence` — bool, does the conversation flow logically?
- `context_retention` — excellent/good/fair/poor, does the agent remember prior constraints?
- `search_quality` — necessary/unnecessary/skipped, did the agent search at the right times?

## Scenarios

| Key | Name | What it tests |
|---|---|---|
| `restaurant` | Restaurant Research | Multi-turn discovery; turn 4 must synthesise without re-searching |
| `safety` | Food Safety Research | Resolves implicit reference ("that restaurant") into a concrete search query |
| `allergen` | Silent Allergen Carryover | Peanut allergy stated once in turn 1; must silently appear in a turn-4 search query |
| `nosearch` | No-Search Needed | Correct behaviour = zero searches (general knowledge only) |

## Providers

Switching between OpenAI and Databricks is done via `AgentConfig.provider`. The judge model URI must match: `openai:/<model>` or `databricks:/<model>`. LiteLLM handles the underlying API calls.

## Package management

Uses `uv`. Do not use `pip install` directly — use `uv add <package>` to keep `pyproject.toml` and `uv.lock` in sync.

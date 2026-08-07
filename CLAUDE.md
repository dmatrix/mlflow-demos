# CLAUDE.md

## Project overview

Demo agents and notebooks for MLflow GenAI evaluation and Databricks governance. Three demos:

- `devconnect/restaurant_research_bot` — multi-turn restaurant research agent, evaluated with MLflow session-level judges
- `unity_ai_gateway_governance/` — centralized guardrails for coding agents across three provider-specific governed model services
- `agentbricks/fema-disaster` — Databricks Asset Bundle deploying a multi-agent FEMA disaster-response supervisor

## Layout

```
devconnect/
  config.py                        # AgentConfig dataclass (provider, model, temperature)
  mlflow_config.py                 # setup_mlflow_tracking() helper
  providers.py                     # LiteLLM-based provider abstraction
  restaurant_research_bot/
    restaurant_research_agent_cls.py  # RestaurantResearchAgent — core agent class
    restaurant_research_agent.py      # CLI entry point (argparse → agent)
    restaurant_research_agent_devconnect.ipynb  # demo notebook
    scenarios.py                     # get_scenario_*() functions
    prompts.py                       # system prompt + three judge instruction strings
    search_tool.py                   # Tavily web_search() tool wrapper

unity_ai_gateway_governance/
  ai_gateway_demo.ipynb            # Main demo notebook (8 acts)
  gateway_config.py                # GatewayConfig + verify_gateway, fetch_service_config per model service
  agent_simulator.py               # SimulatedAgent, GatewayClient, policy-block detection, request retry
  scenarios.py                     # Guardrail payloads (PII, injection, unsafe) + clean-scenario builder
  clean_tasks.py                   # 15 coding tasks per agent (10 used by default)
  prompts.py                       # System prompts for each coding agent persona
  observability.py                 # SQL query templates for inference tables + system.ai_gateway.usage
  images/
    ai_gateway_architecture.svg    # Architecture diagram (PNG rendered from it, checked in)

agentbricks/fema-disaster/         # Asset Bundle: supervisor + Genie Space + Knowledge Assistant
```

## Running the demo

```bash
# Install
uv sync

# Start MLflow tracking server (required)
mlflow ui   # → http://localhost:5000

# Run CLI (OpenAI)
uv run mlflow-restaurant-research-bot

# Run a specific scenario
uv run mlflow-restaurant-research-bot --scenario allergen

# Run with Databricks
uv run mlflow-restaurant-research-bot --provider databricks --model databricks-gpt-5-mini
```

## Environment variables

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | OpenAI provider (default) |
| `TAVILY_API_KEY` | Web search (all providers) |
| `DATABRICKS_HOST` | Databricks provider |
| `DATABRICKS_TOKEN` | Databricks provider |

Place these in `devconnect/restaurant_research_bot/.env`, loaded via `python-dotenv`.

The gateway demo has its own `.env` and variables — see `unity_ai_gateway_governance/env-template`.

## Key concepts

**Session-level evaluation:** all three judges use the `{{ conversation }}` template, which tells `mlflow.genai.evaluate()` to aggregate every turn in a session before scoring, rather than scoring turn by turn.

**Stateless search:** `web_search()` (Tavily) gets only the query string, with no conversation history. The agent has to build self-contained queries that carry prior context, resolving "that restaurant" to "Nopa San Francisco" itself.

**Three judges** (defined in `prompts.py`, instantiated in `RestaurantResearchAgent.__init__`):
- `conversation_coherence` — bool, does the conversation flow logically?
- `context_retention` — excellent/good/fair/poor, does the agent remember prior constraints?
- `search_quality` — necessary/unnecessary/skipped, did the agent search at the right times?

**Guardrail blocks are HTTP 200:** in the gateway demo, a denied request returns 200 with a `databricks_service_policy` object where `action == "deny"`. Checking the status code will not find blocks — see `detect_policy_block` in `agent_simulator.py`. Denied requests never reach the model, so they never appear in the inference tables either; MLflow traces hold that evidence.

**PII runs on responses too:** the PII policy inspects both request (`pre_call`) and response (`post_call`), so a harmless prompt can be denied for what the model wrote back. Asking for a `pyproject.toml` gets denied when the model fills in an author email. Keep clean prompts free of anything that invites PII-shaped output.

## Scenarios

| Key | Name | What it tests |
|---|---|---|
| `restaurant` | Restaurant Research | Multi-turn discovery; turn 4 must synthesize without re-searching |
| `safety` | Food Safety Research | Resolves an implicit reference ("that restaurant") into a concrete search query |
| `allergen` | Silent Allergen Carryover | Peanut allergy stated once in turn 1; must reappear in a turn-4 search query unprompted |
| `nosearch` | No-Search Needed | Correct behavior is zero searches — general knowledge only |

## Providers

Switch between OpenAI and Databricks via `AgentConfig.provider`. The judge model URI has to match: `openai:/<model>` or `databricks:/<model>`. LiteLLM handles the underlying calls.

## Package management

Uses `uv`. Don't call `pip install` directly — use `uv add <package>` so `pyproject.toml` and `uv.lock` stay in sync.

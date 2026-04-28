# Governing Coding Agent Sprawl with Unity AI Gateway

![AI Gateway Architecture](./images/ai_gateway_architecture.png)

**The problem:** Your organization has dozens of developers using Cursor, Claude Code, Codex CLI, and Gemini CLI. Each agent calls a different LLM provider with its own API key. You have no visibility into who is spending what, no guardrails against data leaks, and no audit trail.

**The solution:** Route every coding agent through a single Unity AI Gateway endpoint. This notebook demonstrates the three governance pillars:

| Pillar | What it does |
|--------|--------------|
| **Security & Audit** | Guardrails (PII detection, prompt injection, safety filters), all requests logged to Unity Catalog |
| **Cost Management** | Rate limiting (QPM/TPM), unified billing, budget allocation per user/group |
| **Observability** | Inference tables in Delta, per-user metrics, usage dashboards |

> **Reference:** [Governing Coding Agent Sprawl with Unity AI Gateway](https://www.databricks.com/blog/governing-coding-agent-sprawl-unity-ai-gateway)

## What the demo covers

The notebook walks through four acts:

1. **Act 1 — Configure the Gateway** — Programmatic setup of PII detection (BLOCK mode), safety filters, inference tables, and usage tracking via the Databricks SDK
2. **Act 2 — Simulate the Coding Agent Swarm** — Four simulated coding agents (Cursor, Claude Code, Codex CLI, Gemini CLI) send legitimate coding requests through the same gateway endpoint
3. **Act 3 — Guardrails in Action** — PII (SSNs, credit cards) and prompt injection attempts are blocked in real time with HTTP 400 responses
4. **Act 4 — The Audit Trail** — Query inference tables in Delta showing all requests, including blocked ones, for compliance and cost tracking

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- An existing serving endpoint backed by a foundation model (e.g., `databricks-gpt-5-4`, `databricks-claude-sonnet-4`)
- `CAN_MANAGE` permission on the serving endpoint (to configure AI Gateway)
- `CREATE TABLE` permission on the target catalog/schema (for inference tables)
- Databricks personal access token (for local runs)

## Running locally

1. Create a `.env` file from the template and fill in your values:

    ```bash
    cd ai_gateway_governance
    cp env-template .env
    ```

    | Variable | Description |
    |----------|-------------|
    | `DATABRICKS_HOST` | Workspace URL (e.g., `https://<workspace>.cloud.databricks.com`) |
    | `DATABRICKS_TOKEN` | Personal access token |
    | `AI_GATEWAY_ENDPOINT_NAME` | Name of your serving endpoint |
    | `AI_GATEWAY_MODEL` | Foundation model name (e.g., `databricks-gpt-5-4`) |
    | `UC_CATALOG` | Unity Catalog catalog for inference tables |
    | `UC_SCHEMA` | Unity Catalog schema for inference tables |

2. Install dependencies and launch the notebook:

    ```bash
    uv sync
    jupyter notebook ai_gateway_demo.ipynb
    ```

3. Run Acts 1–3 interactively. Act 4 (inference table queries) requires a Databricks runtime — the notebook prints the raw SQL instead when running locally.

## Deploying to Databricks

This project uses [Declarative Automation Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html) to deploy the notebook and all supporting modules to a Databricks workspace.

1. Install the Databricks CLI (if not already installed):

    ```bash
    brew install databricks/tap/databricks
    ```

2. Authenticate with your workspace:

    ```bash
    databricks auth login --host https://<your-workspace>.cloud.databricks.com
    ```

3. Validate and deploy:

    ```bash
    cd ai_gateway_governance
    databricks bundle validate
    databricks bundle deploy
    ```

4. Open `ai_gateway_demo` in your Databricks workspace and run Acts 1–4 interactively. The notebook auto-detects the Databricks runtime and fetches host/token via `dbutils` — no `.env` file needed.

> **Tip:** Update `databricks.yml` to change the target workspace or add additional targets (e.g., staging, production).

## File structure

```
ai_gateway_governance/
├── databricks.yml          # Declarative Automation Bundle configuration
├── ai_gateway_demo.ipynb   # Demo notebook (runs locally and on Databricks)
├── gateway_config.py       # GatewayConfig dataclass + SDK helpers for AI Gateway setup
├── agent_simulator.py      # SimulatedAgent, GatewayClient, request handling with retry
├── scenarios.py            # Test payloads: clean requests, PII, prompt injection
├── prompts.py              # System prompts for each coding agent persona
├── observability.py        # SQL query templates for inference tables
├── images/
│   └── ai_gateway_architecture.png
├── env-template            # Environment variable template (local use)
└── README.md
```

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

The notebook walks through five acts:

1. **Act 1 — Configure the Gateway** — Programmatic setup of PII detection (BLOCK mode), safety filters, inference tables, and usage tracking via the Databricks SDK
2. **Act 2 — Simulate the Coding Agent Swarm** — Four simulated coding agents (Cursor, Claude Code, Codex CLI, Gemini CLI) send legitimate coding requests through the same gateway endpoint
3. **Act 3 — Guardrails in Action** — PII (SSNs, credit cards), prompt injection, and unsafe content attempts are blocked in real time with HTTP 400 responses
4. **Act 4 — The Audit Trail** — Use Databricks Genie to explore the inference table in plain English: all requests, blocked requests, guardrail outcomes — no SQL required
5. **Act 5 — Usage Tracking** — Query token consumption and latency breakdowns (allowed vs. blocked) and hourly endpoint usage via Genie natural language questions

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks personal access token (for local runs)

### Configure AI Gateway in the Databricks UI

Before running the notebook, set up your AI Gateway endpoint in the Databricks workspace:

1. **Create or select a AI Gateway** — Add a new **AI Gateway Endpoint** in your workspace and choose an existing foundation model endpoint (e.g., `databricks-gpt-5-4`, `databricks-claude-sonnet-4`) or create a new one.

2. **Enable AI Gateway** — on the endpoint's detail page, open the **AI Gateway** tab and click **Enable**.

3. **Select the model** — confirm the foundation model backing the endpoint. This becomes the `MODEL` variable in the notebook config cell.

4. **Configure guardrails** — under **Guardrails**, enable the following:
   - **PII Detection** — set mode to **Block** to reject requests containing SSNs, credit card numbers, and other sensitive data
   - **Jailbreak and Prompt Injection** — enable to block DAN prompts and attempts to extract system instructions
   - ** Unsafe Content ** -- enable to block Unsafe Content

5. **Enable inference tables** — under **Inference Tables**, turn on logging and point it at your Unity Catalog destination (`CATALOG.SCHEMA`). This powers the audit trail in Act 4 and the cost queries in Act 5.

6. **Enable usage tracking** — turn on **Usage Tracking** to capture per-request token counts.

Once the endpoint is configured, copy the endpoint name into the `ENDPOINT_NAME` variable in the notebook's config cell.


### Set up a Genie Space

Before presenting Acts 4 and 5:

1. In your Databricks workspace, open **Genie Spaces** and create a new space.
2. Add `CATALOG.SCHEMA.{SCHEMA}_payload` as a data source — this is the inference table created in step 5 above (e.g. `jules_catalog.unity-ai-gateway-demo.unity-ai-gateway-demo_payload`).
3. Keep the Genie space open during the demo. Acts 4 and 5 provide ready-made questions to paste directly into the space — no code execution is required.

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

3. Run Acts 1–3 interactively.

   > **Acts 4 and 5 require a Databricks workspace.** They use Databricks Genie to query the inference tables — deploy the notebook to Databricks (see below) and open the Genie space alongside it to use the provided questions.

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
├── scenarios.py            # Test payloads: clean requests, PII, prompt injection, unsafe content
├── prompts.py              # System prompts for each coding agent persona
├── observability.py        # SQL query templates for inference tables
├── images/
│   └── ai_gateway_architecture.png
├── env-template            # Environment variable template (local use)
└── README.md
```

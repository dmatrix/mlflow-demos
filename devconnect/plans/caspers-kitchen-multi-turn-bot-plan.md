# Plan: Multi-Turn Conversational Bot for Casper's Kitchens

## The Core Problem

The LLM has no direct knowledge of or access to Casper's data — not the Delta tables, not
the PDFs, not the live event stream. This is the gap the original plan glossed over. The
solution is that Casper's already solves this problem through **deployed Model Serving
endpoints**. The multi-turn bot does not query data itself; it delegates to those endpoints
and wraps them in a session-tracked conversation loop.

---

## What's Already Deployed

| Endpoint | What it answers | How it works |
|----------|----------------|--------------|
| `{CATALOG}-menu-supervisor` | Menu, nutrition, allergens, inspection scores, cross-domain | Routes internally to Genie (SQL) + Menu KA (PDFs) + Inspection KA (PDFs) |
| `{CATALOG}-complaint-agent` | Order status, delays, refund recommendations | DSPy ReAct agent calling UC tools: `get_order_overview()`, `get_order_timing()`, `get_location_timings()` |
| `{CATALOG}-menu-knowledge` | Dish descriptions, ingredients, prep details from PDFs | Agent Bricks Knowledge Assistant over 16 menu PDFs |
| `{CATALOG}-inspection-knowledge` | Violation details, inspector notes, corrective actions | Agent Bricks Knowledge Assistant over 12 inspection report PDFs |

Each endpoint exposes an **OpenAI-compatible chat API** (`/serving-endpoints/{name}/invocations`)
callable via `WorkspaceClient` or the OpenAI SDK pointed at the Databricks workspace.

---

## Correct Architecture

```
User (turn 1, 2, 3, ...)
        |
        v
CaspersKitchenAgent.handle_message(message, session_id)
  |
  |-- @mlflow.trace  +  mlflow.update_current_trace(session=session_id)
  |-- maintains full session_histories[session_id]
  |
  v
IntentRouter  (a small LLM call or keyword rules)
  |
  +---> ORDER intent    ---> complaint_agent endpoint
  |                          (handles order_id lookup, delay check, refund)
  |
  +---> MENU / SAFETY   ---> menu_supervisor endpoint
  |     / NUTRITION /        (internally routes to Genie, Menu KA, Inspection KA)
  |     RECOMMENDATION
  |
  +---> SMALL TALK /    ---> direct LLM call  (no backend needed)
        CLARIFICATION
  |
  v
Response text from whichever backend answered
  |
  v
Appended to session_histories[session_id]
  |
  v
MLflow trace tagged with session_id
```

The multi-turn bot's job is:
1. **Classify intent** each turn
2. **Call the right endpoint** with the current message (and relevant history)
3. **Store the turn** in session history
4. **Tag the trace** with the session ID so MLflow can group turns

The endpoints already handle data retrieval. The bot handles conversation management.

---

## Why Two Endpoints Instead of One

The **menu supervisor** only coordinates menu/safety/nutrition questions.
Order-level queries (live event data, delivery timing, refund logic) live in the
**complaint agent**, which has UC tool functions registered against `{CATALOG}.ai`
that query `lakeflow.all_events` and compute delivery percentiles.

A future enhancement could add the complaint agent as a 4th sub-agent to the supervisor,
making routing fully automatic — but for the initial bot, explicit intent routing is
simpler and more transparent.

---

## How Each Scenario Actually Gets Answered

### Scenario: Food Safety Inquiry

```
User: "I heard one of your Chicago locations had food safety issues recently."

  IntentRouter  →  SAFETY  →  menu_supervisor endpoint
    menu_supervisor internally routes to: Inspection KA
    Inspection KA retrieves from: /Volumes/{CATALOG}/food_safety/reports/ (PDFs)
    Returns: "Chicago Jan 2024: score 69 (F), critical violation — employee
              handling food without gloves (code V-103, 3-day deadline)"

User: "What corrective action was required?"

  IntentRouter  →  SAFETY  →  menu_supervisor endpoint
    history includes prior turn, so bot knows we're still on Chicago
    Inspection KA retrieves: corrective action details from the same report
    Returns: "Rearrange refrigeration, retrain staff on storage hierarchy"
```

The LLM never sees the PDF or the Delta table. The supervisor endpoint handles all of that.

### Scenario: Allergen-Safe Recommendation

```
User: "I'm vegetarian with a severe peanut allergy. Near San Francisco. Budget $15."

  IntentRouter  →  MENU/RECO  →  menu_supervisor endpoint
    menu_supervisor routes to: Genie space
    Genie queries: SELECT ... FROM menu_documents.silver_items
                   WHERE is_allergen_free = true AND price <= 15 ...
    Returns: structured list of qualifying dishes

User: "Are any NootroNourish bowls peanut-free?"

  IntentRouter  →  MENU  →  menu_supervisor endpoint
    menu_supervisor routes to: Menu KA (for prose detail) + Genie (for allergen flag)
    Returns: specific NootroNourish items with allergen citations from PDF
```

The multi-turn bot's role: ensure the **peanut allergy constraint** stated in turn 1
is still present in the message history sent to the supervisor in turn 4.

### Scenario: Delayed Delivery Complaint

```
User: "My Wok This Way order hasn't arrived. Order ID ORD-7842."

  IntentRouter  →  ORDER  →  complaint_agent endpoint
    complaint_agent calls: get_order_overview("ORD-7842")
                           get_order_timing("ORD-7842")
                           get_location_timings("san_francisco")
    Compares actual delivery time to P75 benchmark, decides: delayed
    Returns: "Your order is 45 minutes past the P75 delivery time for this location.
              I recommend a full refund."
```

---

## File Structure

```
demos/multi-turn-bot/
├── README.md
├── caspers_bot.py           # CLI entry point
├── caspers_bot_cls.py       # CaspersKitchenAgent class
├── intent_router.py         # Classifies each turn's intent
├── endpoint_clients.py      # Thin wrappers for calling each deployed endpoint
├── prompts.py               # System context + judge instructions
├── scenarios.py             # Conversation test cases
└── caspers_bot.ipynb        # Interactive notebook version
```

---

## Step-by-Step Implementation

### Step 1 — `endpoint_clients.py`

Thin wrappers that call each Model Serving endpoint via the OpenAI SDK.
These are the only place actual data retrieval happens.

```python
from openai import OpenAI
import os

def _make_client(host: str, token: str) -> OpenAI:
    return OpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints"
    )

def call_supervisor(
    message: str,
    history: list[dict],
    endpoint_name: str,
    host: str,
    token: str
) -> str:
    """
    Call the Multi-Agent Supervisor endpoint.
    Handles: menu questions, nutrition, allergens, inspection/safety, recommendations.
    The supervisor internally routes to Genie, Menu KA, or Inspection KA.

    history: list of {"role": "user"/"assistant", "content": "..."} from prior turns.
             Pass this so the supervisor has context (e.g., the peanut allergy from turn 1).
    """
    client = _make_client(host, token)
    messages = history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=endpoint_name,
        messages=messages
    )
    return response.choices[0].message.content


def call_complaint_agent(
    message: str,
    history: list[dict],
    endpoint_name: str,
    host: str,
    token: str
) -> str:
    """
    Call the Complaint/Refund Agent endpoint.
    Handles: order status, delivery delays, refund requests.
    The agent internally calls UC tools: get_order_overview(), get_order_timing(), etc.
    """
    client = _make_client(host, token)
    messages = history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=endpoint_name,
        messages=messages
    )
    return response.choices[0].message.content
```

> **Note on history passing:** The supervisor and complaint agent are stateless —
> they don't remember prior turns. The multi-turn bot must reconstruct context
> by passing the accumulated `history` list on every call. This is exactly the
> conversation management the bot adds on top.

---

### Step 2 — `intent_router.py`

Classifies each user message so the bot knows which endpoint to call.
Use a fast LLM call or simple keyword heuristics — this does not need to be perfect.

```python
from openai import OpenAI
from typing import Literal

Intent = Literal["order", "menu_safety", "smalltalk"]

_ROUTER_PROMPT = """Classify this customer message into one of three categories:

- order: questions or complaints about a specific delivery, order ID, missing items,
         refund requests, delivery timing
- menu_safety: questions about menu items, dishes, ingredients, allergens, calories,
               prices, food safety scores, inspection results, recommendations
- smalltalk: greetings, thanks, clarifications with no domain-specific content

Respond with exactly one word: order, menu_safety, or smalltalk.

Message: {message}
"""

def classify_intent(message: str, client: OpenAI, model: str) -> Intent:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _ROUTER_PROMPT.format(message=message)}],
        max_tokens=5,
        temperature=0
    )
    raw = response.choices[0].message.content.strip().lower()
    if raw in ("order", "menu_safety", "smalltalk"):
        return raw
    # Default: if ambiguous, send to supervisor (safe fallback)
    return "menu_safety"
```

---

### Step 3 — `caspers_bot_cls.py`

The agent class. Manages conversation history, routes each turn, tags MLflow traces.

```python
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from openai import OpenAI
from typing import Dict, List, Literal
from intent_router import classify_intent
from endpoint_clients import call_supervisor, call_complaint_agent
from prompts import (
    get_coherence_judge_instructions,
    get_context_retention_judge_instructions,
    get_domain_routing_judge_instructions,
    SMALLTALK_SYSTEM_PROMPT
)

class CaspersKitchenAgent:

    def __init__(
        self,
        host: str,
        token: str,
        supervisor_endpoint: str,       # e.g. "caspersdev-menu-supervisor"
        complaint_endpoint: str,        # e.g. "caspersdev-complaint-agent"
        router_model: str,              # fast model for intent classification
        judge_model: str,               # model URI for session-level judges
        mlflow_experiment: str = "caspers-multi-turn-bot",
        catalog: str = "caspersdev",
    ):
        self.host = host
        self.token = token
        self.supervisor_endpoint = supervisor_endpoint
        self.complaint_endpoint = complaint_endpoint
        self.catalog = catalog

        # Client for intent routing + smalltalk (uses base workspace LLM, not an agent endpoint)
        self.router_client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
        self.router_model = router_model

        self.mlflow_experiment = mlflow_experiment
        self.session_histories: Dict[str, List[dict]] = {}
        self._init_judges(judge_model)

    def _init_judges(self, judge_model: str):
        """
        All three judges use {{ conversation }} --> automatically session-level in MLflow 3.7.
        """
        self.coherence_judge = make_judge(
            name="conversation_coherence",
            model=judge_model,
            instructions=get_coherence_judge_instructions(),
            feedback_value_type=bool
        )
        self.context_judge = make_judge(
            name="context_retention",
            model=judge_model,
            instructions=get_context_retention_judge_instructions(),
            feedback_value_type=Literal["excellent", "good", "fair", "poor"]
        )
        self.routing_judge = make_judge(
            name="domain_routing",
            model=judge_model,
            instructions=get_domain_routing_judge_instructions(),
            feedback_value_type=bool
        )

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="handle_caspers_message")
    def handle_message(self, message: str, session_id: str) -> str:
        # --- Critical: links this trace to the session ---
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id}
        )

        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        history = self.session_histories[session_id]

        # --- Classify intent ---
        intent = classify_intent(message, self.router_client, self.router_model)

        # --- Route to the right backend ---
        if intent == "order":
            reply = call_complaint_agent(
                message, history,
                self.complaint_endpoint, self.host, self.token
            )
        elif intent == "menu_safety":
            reply = call_supervisor(
                message, history,
                self.supervisor_endpoint, self.host, self.token
            )
        else:  # smalltalk
            reply = self._smalltalk(message, history)

        # --- Update session history ---
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        return reply

    def _smalltalk(self, message: str, history: list) -> str:
        """Direct LLM call for greetings/clarifications -- no backend needed."""
        messages = [{"role": "system", "content": SMALLTALK_SYSTEM_PROMPT}]
        messages += history
        messages.append({"role": "user", "content": message})
        response = self.router_client.chat.completions.create(
            model=self.router_model,
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content

    def run_conversation(self, messages: List[str], session_id: str) -> List[str]:
        replies = []
        for i, msg in enumerate(messages):
            print(f"\nTurn {i+1}/{len(messages)}")
            print(f"  User: {msg}")
            reply = self.handle_message(msg, session_id)
            print(f"  Bot:  {reply}")
            replies.append(reply)
        return replies

    def evaluate_session(self, session_id: str, run_id: str) -> dict:
        """
        Evaluate all traces in this MLflow run using mlflow.genai.evaluate().

        run_id comes from the mlflow.start_run() context in the CLI.
        All handle_message() traces produced inside that run share this run_id,
        so filtering by run_id gives us exactly the turns for this session.
        """
        experiment = mlflow.get_experiment_by_name(self.mlflow_experiment)
        session_traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            filter_string=f"run_id = '{run_id}'"
        )

        if len(session_traces) == 0:
            raise ValueError(f"No traces found for run_id '{run_id}'")

        print(f"\nEvaluating session '{session_id}' ({len(session_traces)} traces)...")

        # mlflow.genai.evaluate() fans out to each judge, returns a result DataFrame.
        # Session-level judges (those using {{ conversation }}) receive ALL traces
        # aggregated into a single conversation view rather than one row per trace.
        eval_results = mlflow.genai.evaluate(
            data=session_traces,
            scorers=[self.coherence_judge, self.context_judge, self.routing_judge]
        )

        result_df = eval_results.result_df

        # Column names follow the pattern "<judge_name>/value" and
        # "<judge_name>/justification". Search by substring to be version-tolerant.
        def extract(keyword, suffix):
            cols = [c for c in result_df.columns
                    if keyword in c.lower() and suffix in c.lower()]
            if not cols:
                return None
            series = result_df[cols[0]].dropna()
            return series.iloc[0] if len(series) > 0 else None

        return {
            "session_id": session_id,
            "num_traces": len(session_traces),
            "coherence": {
                "feedback_value": extract("coherence", "/value"),
                "rationale":      extract("coherence", "/justification") or "",
                "passed":         extract("coherence", "/value"),
            },
            "context_retention": {
                "feedback_value": extract("context", "/value"),
                "rationale":      extract("context", "/justification") or "",
            },
            "domain_routing": {
                "feedback_value": extract("routing", "/value"),
                "rationale":      extract("routing", "/justification") or "",
                "passed":         extract("routing", "/value"),
            },
        }
```

---

### Step 4 — `prompts.py`

System context for smalltalk + judge instructions (all use `{{ conversation }}`).

```python
SMALLTALK_SYSTEM_PROMPT = """You are a friendly assistant for Casper's Kitchens,
a ghost kitchen platform with 16 brands across San Francisco, Silicon Valley,
Bellevue, and Chicago. Answer greetings and clarifications briefly and warmly.
Do not make up order details, menu items, or inspection data."""


def get_coherence_judge_instructions() -> str:
    return """Evaluate the coherence of this Casper's Kitchens customer conversation.

{{ conversation }}

Does the conversation flow logically? Does the assistant:
- Give responses that follow from prior turns?
- Avoid contradictions (e.g., recommending a dish it said was unavailable earlier)?
- Handle topic shifts (orders → menus → safety) without losing context?

Value: True if coherent, False if there are significant coherence issues.
Rationale: 2-3 sentences covering logical flow, consistency, and continuity.
"""


def get_context_retention_judge_instructions() -> str:
    return """Evaluate context retention in this Casper's Kitchens assistant conversation.

{{ conversation }}

Key facts to watch for: order IDs, allergen constraints, location, dietary preferences,
complaint details, budget. Did the assistant carry these forward across turns?

EXCELLENT: All key facts remembered and applied in every relevant turn.
GOOD: Most facts retained; minor lapses that don't derail the conversation.
FAIR: Occasionally re-asks for info already provided; forgets constraints.
POOR: Treats each turn independently; ignores prior user-stated constraints.

Value: excellent, good, fair, or poor.
Rationale: Cite specific turns where context was or wasn't retained.
"""


def get_domain_routing_judge_instructions() -> str:
    return """Evaluate domain-shift handling in this Casper's Kitchens conversation.

{{ conversation }}

The conversation may span orders, menus, food safety, and recommendations.
Did the assistant recognize domain shifts and route them appropriately?
Did it carry relevant cross-domain context (e.g., a peanut allergy mentioned
during a menu question should still apply during a recommendation question)?

Value: True if domain shifts were handled gracefully, False otherwise.
Rationale: Cite specific turns where domain shifts occurred and how they were handled.
"""
```

---

### Step 5 — `scenarios.py`

Test cases grounded in Casper's actual data. Each includes the expected intent sequence
so routing correctness can be verified.

```python
def get_scenario_food_safety_inquiry():
    return {
        "name": "Food Safety Inquiry",
        "session_id": "session-safety-001",
        "expected_intents": ["menu_safety", "menu_safety", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "good",
        "messages": [
            "I heard one of your Chicago locations had food safety issues recently.",
            "What exactly was the violation and when did it happen?",
            "Was there a corrective action taken? Is it safe to order from there now?",
            "What is the current inspection score for that Chicago location?"
        ]
        # Backend: menu_supervisor --> Inspection KA --> Chicago PDF reports
        # Inspection KA returns: Jan 2024 score 69 (F), critical violation code V-103,
        # employee handling food without gloves, 3-day corrective deadline
    }


def get_scenario_order_complaint():
    return {
        "name": "Delayed Delivery Complaint",
        "session_id": "session-complaint-001",
        "expected_intents": ["order", "order", "order", "order"],
        "expected_coherence": True,
        "expected_retention": "excellent",
        "messages": [
            "My Wok This Way order was supposed to arrive an hour ago. Order ID ORD-7842.",
            "The app shows delivered but I never received it. I'm at 123 Main St SF.",
            "Yes, I checked with my neighbors and building lobby. Nothing.",
            "What are my options for a refund or reorder?"
        ]
        # Backend: complaint_agent --> get_order_overview(ORD-7842)
        #                          --> get_order_timing(ORD-7842)
        #                          --> get_location_timings("san_francisco")
        # Agent compares actual vs P75 benchmark, recommends refund
    }


def get_scenario_allergen_recommendation():
    return {
        "name": "Allergen-Safe Recommendation",
        "session_id": "session-allergen-001",
        "expected_intents": ["menu_safety", "menu_safety", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "excellent",
        "messages": [
            "I'm looking for something healthy for dinner tonight.",
            "I'm vegetarian and have a severe peanut allergy.",
            "I'm near the San Francisco location. Budget around $15.",
            "The NootroNourish menu sounds interesting -- are any of their bowls peanut-free?"
        ]
        # Backend: menu_supervisor --> Genie (allergen_free items under $15)
        #                          --> Menu KA (NootroNourish PDF for bowl details)
        # Critical test: peanut allergy from turn 2 must be in history sent on turn 4
    }


def get_scenario_cross_domain():
    return {
        "name": "Cross-Domain: Menu + Safety + Recommendation",
        "session_id": "session-crossdomain-001",
        "expected_intents": ["menu_safety", "menu_safety", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "excellent",
        "messages": [
            "What Asian brands do you run and what cuisines do they cover?",
            "I have a shellfish allergy -- which Wok This Way dishes should I avoid?",
            "Is Wok This Way in San Francisco safe from a food safety standpoint?",
            "Given my shellfish allergy, which of your brands would you recommend tonight?"
        ]
        # Backend: all routes through menu_supervisor
        # Turn 1: Genie --> brand table
        # Turn 2: Menu KA --> Wok This Way PDF allergen info
        # Turn 3: Inspection KA --> SF inspection reports
        # Turn 4: Genie (allergen_free) + shellfish constraint from turn 2 in history
    }


def get_scenario_mixed_order_and_menu():
    return {
        "name": "Mixed: Complaint Then Menu Exploration",
        "session_id": "session-mixed-001",
        "expected_intents": ["order", "order", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "good",
        "messages": [
            "My order ORD-5521 was missing the Kung Pao Chicken.",
            "Yes, I'd like a partial refund for the missing item.",
            "Actually, while I have you -- is Kung Pao Chicken high in sodium?",
            "And are there lower-sodium alternatives at Wok This Way?"
        ]
        # Tests intent routing flip from ORDER to MENU_SAFETY mid-conversation
        # Tests whether history (the specific dish name) carries into menu query
    }
```

---

### Step 6 — `caspers_bot.py` (CLI Entry Point)

```python
"""
Usage:
  python caspers_bot.py --catalog caspersdev --scenario safety
  python caspers_bot.py --catalog caspersdev  # all scenarios
"""
import argparse, os, mlflow
from caspers_bot_cls import CaspersKitchenAgent
from scenarios import get_all_scenarios, get_scenario_by_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog",    default="caspersdev")
    parser.add_argument("--scenario",   default=None)
    parser.add_argument("--router-model",  default="databricks-gpt-4o-mini")
    parser.add_argument("--judge-model",   default="databricks-gemini-2-5-flash")
    args = parser.parse_args()

    HOST  = os.environ["DATABRICKS_HOST"]
    TOKEN = os.environ["DATABRICKS_TOKEN"]

    mlflow.set_experiment("caspers-multi-turn-bot")

    agent = CaspersKitchenAgent(
        host=HOST,
        token=TOKEN,
        supervisor_endpoint=f"{args.catalog}-menu-supervisor",
        complaint_endpoint=f"{args.catalog}-complaint-agent",
        router_model=args.router_model,
        judge_model=f"openai:/{args.judge_model}",
        mlflow_experiment="caspers-multi-turn-bot",
        catalog=args.catalog,
    )

    scenarios = [get_scenario_by_name(args.scenario)] if args.scenario else get_all_scenarios()

    for scenario in scenarios:
        print(f"\n{'='*60}\nScenario: {scenario['name']}\n{'='*60}")

        # Each scenario runs inside its own MLflow run.
        # All handle_message() traces created inside the with-block share this run_id.
        # evaluate_session() uses that run_id to find exactly the right traces.
        with mlflow.start_run(run_name=scenario["name"]) as run:
            agent.run_conversation(scenario["messages"], scenario["session_id"])
            results = agent.evaluate_session(scenario["session_id"], run.info.run_id)

        coh = results["coherence"]
        ctx = results["context_retention"]
        rte = results["domain_routing"]
        print(f"\nCoherence:         {'PASS' if coh['passed'] else 'FAIL'} ({coh['feedback_value']})")
        print(f"Context Retention: {str(ctx['feedback_value']).upper()}")
        print(f"Domain Routing:    {'PASS' if rte['passed'] else 'FAIL'} ({rte['feedback_value']})")

if __name__ == "__main__":
    main()
```

---

## Conversation Flow Diagram (Food Safety Example)

```
User: "I heard Chicago had food safety issues"
  |
  v
classify_intent()  --> "menu_safety"
  |
  v
call_supervisor(message, history=[], endpoint="caspersdev-menu-supervisor")
  |
  v
  [inside the supervisor endpoint -- already deployed]
  Supervisor routes to: Inspection KA
  Inspection KA retrieves from: /Volumes/caspersdev/food_safety/reports/ (PDFs)
  Returns: "Jan 2024: score 69 (F), critical violation V-103..."
  |
  v
history = [
  {role: user,      content: "I heard Chicago had food safety issues"},
  {role: assistant, content: "Jan 2024: score 69 (F), critical violation V-103..."}
]
MLflow trace tagged: mlflow.trace.session = "session-safety-001"

User: "What corrective action was required?"
  |
  v
classify_intent()  --> "menu_safety"
  |
  v
call_supervisor(message, history=[... 2 prior messages ...], endpoint="caspersdev-menu-supervisor")
  ^
  The history is what gives the supervisor context that we're talking about Chicago.
  The supervisor (or Inspection KA) uses this to answer without re-stating the question.
  |
  v
Returns: "Corrective action: rearrange refrigeration, retrain staff..."
MLflow trace tagged: mlflow.trace.session = "session-safety-001"
```

---

## Session Evaluation Flow

```
After all turns are complete:

mlflow.search_traces(filter="session = 'session-safety-001'")
  --> [trace_turn_1, trace_turn_2, trace_turn_3, trace_turn_4]

coherence_judge(session=traces)   --> {{ conversation }} template aggregates all 4 turns
context_judge(session=traces)     --> Did bot remember we were discussing Chicago?
routing_judge(session=traces)     --> Did every turn correctly route to menu_safety?
```

---

## Prerequisites Before Running

The following Casper's stages must have been deployed (via `databricks bundle deploy -t menus`):

| Stage completed | Provides |
|-----------------|---------|
| `menu_knowledge_agent` | `{CATALOG}-menu-knowledge` endpoint |
| `inspection_knowledge_agent` | `{CATALOG}-inspection-knowledge` endpoint |
| `menu_genie` | Genie space ID |
| `menu_supervisor` | `{CATALOG}-menu-supervisor` endpoint |
| `complaint_agent` | `{CATALOG}-complaint-agent` endpoint + UC tools |

---

## Key Differences from the Reference Implementation

| Aspect | Reference (TechCorp) | Casper's Kitchens |
|--------|----------------------|-------------------|
| Data access | None -- pure LLM | Delegates to deployed endpoints |
| Routing | Single endpoint | Intent router → supervisor OR complaint agent |
| History passing | LLM maintains context | Bot passes full `history` to stateless endpoints |
| New files | N/A | `intent_router.py`, `endpoint_clients.py` |
| Judges | Coherence + Context Retention | + Domain Routing (new) |
| Scenarios | Generic tech support | Grounded in Casper's real data and endpoints |

---

## Implementation Order

1. [ ] `endpoint_clients.py` — verify supervisor and complaint endpoints respond correctly
2. [ ] `intent_router.py` — test classification on the scenario messages
3. [ ] `prompts.py` — system context and judge instructions
4. [ ] `scenarios.py` — conversation test cases
5. [ ] `caspers_bot_cls.py` — wire everything together with MLflow tracing
6. [ ] `caspers_bot.py` — CLI entry point
7. [ ] `caspers_bot.ipynb` — interactive notebook version
8. [ ] Run all scenarios, review in MLflow UI, tune judge instructions if needed

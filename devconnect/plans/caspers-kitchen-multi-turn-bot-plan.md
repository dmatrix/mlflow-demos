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

> **Deployment note:** The menu supervisor is in the `menus` DABs target; the complaint
> agent is in the `complaints` target. They may not both be deployed at the same time.
> The bot must handle complaint agent absence gracefully (see *Complaint Agent Optionality*).

---

## What to Reuse from the Restaurant Research Bot

The restaurant bot (`devconnect/restaurant_research_bot/`) establishes the patterns this
bot should follow. The table below maps each pattern to its source file and what changes
for Casper's.

| Pattern | Source file | Reuse | Changes for Casper's |
|---------|-------------|-------|---------------------|
| **MLflow Prompt Registry** | `restaurant_research_bot/prompts.py` | Same `_register_if_missing()` + `register_all_prompts()` + `get_*()` accessor pattern | New prompts prefixed `ck-`; 6 prompts instead of 4 (adds smalltalk + intent-router) |
| **AgentConfig dataclass** | `devconnect/config.py` | Reuse directly | Always `provider="databricks"` |
| **Provider factory** | `devconnect/providers.py` | Reuse `get_client("databricks", ...)` | No OpenAI provider path needed |
| **MLflow setup** | `devconnect/mlflow_config.py` | Reuse `setup_mlflow_tracking()` | Different experiment name |
| **Session history** | `restaurant_research_agent_cls.py` | Same `Dict[str, List[dict]]`, user/assistant pairs only | Identical |
| **MLflow tracing + session tagging** | `restaurant_research_agent_cls.py` | Same `@mlflow.trace` + `mlflow.update_current_trace(metadata={"mlflow.trace.session": ...})` | Identical |
| **Session-level judges** | `restaurant_research_agent_cls.py` | Same `make_judge()` with `{{ conversation }}` template | 3rd judge: `domain_routing` (bool) replaces `search_quality` |
| **evaluate_session()** | `restaurant_research_agent_cls.py` | Same `extract()` + `extract_reason()` helper pattern | Different keyword for 3rd judge |
| **Scenario structure** | `restaurant_research_bot/scenarios.py` | Same `_session_counters` + `_next_session_id()` + `get_all_scenarios()` + `get_scenario_by_name()` | Adds `requires_complaint_agent` field |
| **CLI entry point** | `restaurant_research_agent.py` | Same argparse + `start_run()` per scenario + result printing | Different args; skips scenarios when complaint agent unavailable |
| **pyproject.toml script** | `pyproject.toml` line 19 | Add parallel entry | `mlflow-caspers-kitchen-bot` |

**What's genuinely new** (no restaurant bot equivalent):
- `endpoint_clients.py` — thin wrappers calling supervisor + complaint agent
- `intent_router.py` — LLM-based intent classification
- Complaint agent optionality logic

---

## Correct Architecture

```
User (turn 1, 2, 3, ...)
        |
        v
CaspersKitchenAgent.handle_message(message, session_id)
  |
  |-- @mlflow.trace(span_type=SpanType.CHAT_MODEL)
  |-- mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
  |-- maintains full session_histories[session_id]
  |
  v
classify_intent()  [@mlflow.trace(span_type=SpanType.CHAIN)]
  |       loads prompt from MLflow Prompt Registry (ck-intent-router)
  |
  +---> ORDER intent    ---> call_complaint_agent()  [@mlflow.trace(span_type=SpanType.TOOL)]
  |                          (handles order_id lookup, delay check, refund)
  |                          Falls back gracefully if endpoint unavailable
  |
  +---> MENU / SAFETY   ---> call_supervisor()  [@mlflow.trace(span_type=SpanType.TOOL)]
  |     / NUTRITION /        (internally routes to Genie, Menu KA, Inspection KA)
  |     RECOMMENDATION
  |
  +---> SMALL TALK /    ---> _smalltalk()  (direct LLM call, no backend needed)
        CLARIFICATION        loads prompt from MLflow Prompt Registry (ck-smalltalk-prompt)
  |
  v
Response text from whichever backend answered
  |
  v
Appended to session_histories[session_id]  (user/assistant pairs only)
  |
  v
MLflow trace tagged with session_id, intent visible as child span
```

The multi-turn bot's job is:
1. **Classify intent** each turn (visible as a CHAIN child span in the trace)
2. **Call the right endpoint** with the current message and relevant history (visible as a TOOL child span)
3. **Store the turn** in session history (user/assistant pairs only; discard intermediates)
4. **Tag the trace** with the session ID so MLflow can group turns for evaluation

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

## Complaint Agent Optionality

The complaint agent is in the `complaints` DABs target, separate from the `menus` target.
When only `menus` is deployed, the complaint agent endpoint won't exist. This affects
multiple files:

1. **`caspers_kitchen_agent_cls.py`**: Constructor accepts `complaint_endpoint: str | None`.
   When `None`, `handle_message()` routes `order` intents to a polite fallback message
   ("I can help with menu and food safety questions. For order issues, please contact
   support...") instead of calling an endpoint.

2. **`caspers_kitchen_agent.py` (CLI)**: `--no-complaint-agent` flag. By default, the
   complaint endpoint name is `{catalog}-complaint-agent`.

3. **`scenarios.py`**: Each scenario dict has a `requires_complaint_agent: bool` field.
   The `order_complaint` and `mixed` scenarios set this to `True`.

4. **CLI loop**: Skips scenarios where `requires_complaint_agent=True` and the complaint
   endpoint is unavailable, printing a clear skip message.

---

## How Each Scenario Actually Gets Answered

### Scenario: Food Safety Inquiry

```
User: "I heard one of your Chicago locations had food safety issues recently."

  classify_intent()  →  "menu_safety"  [CHAIN span]
  call_supervisor(message, history=[], endpoint)  [TOOL span]
    menu_supervisor internally routes to: Inspection KA
    Inspection KA retrieves from: /Volumes/{CATALOG}/food_safety/reports/ (PDFs)
    Returns: "Chicago Jan 2024: score 69 (F), critical violation — employee
              handling food without gloves (code V-103, 3-day deadline)"

User: "What corrective action was required?"

  classify_intent()  →  "menu_safety"  [CHAIN span]
  call_supervisor(message, history=[... 2 prior messages ...], endpoint)  [TOOL span]
    history includes prior turn, so bot knows we're still on Chicago
    Inspection KA retrieves: corrective action details from the same report
    Returns: "Rearrange refrigeration, retrain staff on storage hierarchy"
```

The LLM never sees the PDF or the Delta table. The supervisor endpoint handles all of that.

### Scenario: Allergen-Safe Recommendation

```
User: "I'm vegetarian with a severe peanut allergy. Near San Francisco. Budget $15."

  classify_intent()  →  "menu_safety"  [CHAIN span]
  call_supervisor(message, history=[], endpoint)  [TOOL span]
    menu_supervisor routes to: Genie space
    Genie queries: SELECT ... FROM menu_documents.silver_items
                   WHERE is_allergen_free = true AND price <= 15 ...
    Returns: structured list of qualifying dishes

User: "Are any NootroNourish bowls peanut-free?"

  classify_intent()  →  "menu_safety"  [CHAIN span]
  call_supervisor(message, history=[... prior messages ...], endpoint)  [TOOL span]
    menu_supervisor routes to: Menu KA (for prose detail) + Genie (for allergen flag)
    Returns: specific NootroNourish items with allergen citations from PDF
```

The multi-turn bot's role: ensure the **peanut allergy constraint** stated in turn 1
is still present in the message history sent to the supervisor in turn 4.

### Scenario: Delayed Delivery Complaint

```
User: "My Wok This Way order hasn't arrived. Order ID ORD-7842."

  classify_intent()  →  "order"  [CHAIN span]
  call_complaint_agent(message, history=[], endpoint)  [TOOL span]
    complaint_agent calls: get_order_overview("ORD-7842")
                           get_order_timing("ORD-7842")
                           get_location_timings("san_francisco")
    Compares actual delivery time to P75 benchmark, decides: delayed
    Returns: "Your order is 45 minutes past the P75 delivery time for this location.
              I recommend a full refund."
```

> **If complaint agent is unavailable:** The bot returns a polite fallback message
> and the `order_complaint` scenario is skipped during evaluation.

---

## File Structure

```
devconnect/caspers_kitchen_bot/
├── __init__.py                    # Public exports (CaspersKitchenAgent + scenario getters)
├── prompts.py                     # Registry-backed: system, smalltalk, intent-router, 3 judges
├── endpoint_clients.py            # call_supervisor(), call_complaint_agent() with @mlflow.trace
├── intent_router.py               # classify_intent() with @mlflow.trace
├── caspers_kitchen_agent_cls.py   # CaspersKitchenAgent class
├── caspers_kitchen_agent.py       # CLI entry point (argparse → agent)
├── scenarios.py                   # 5 conversation test cases
├── .env                           # DATABRICKS_HOST, DATABRICKS_TOKEN (git-ignored)
└── caspers_kitchen_agent.ipynb    # Interactive notebook version (later)
```

---

## Step-by-Step Implementation

### Step 1 — `prompts.py`

Follow the exact registry pattern from `devconnect/restaurant_research_bot/prompts.py`.

**Six registered prompts** (prefixed `ck-` to avoid collisions with the restaurant bot):

| Registry name | Type | Template content |
|--------------|------|-----------------|
| `ck-system-prompt` | System context | Casper's Kitchen ghost kitchen context (16 brands, 4 cities, SF/SV/Bellevue/Chicago) |
| `ck-smalltalk-prompt` | Smalltalk | Brief, warm; no data fabrication about orders/menus/inspections |
| `ck-intent-router` | Classification | Classify `{message}` → order/menu_safety/smalltalk (uses `.format()`, not `{{ }}`) |
| `ck-judge-conversation-coherence` | Judge (bool) | Uses `{{ conversation }}`; evaluates logical flow, contradictions, topic shifts |
| `ck-judge-context-retention` | Judge (4-level) | Uses `{{ conversation }}`; watches order IDs, allergens, location, dietary prefs, budget |
| `ck-judge-domain-routing` | Judge (bool) | Uses `{{ conversation }}`; evaluates domain-shift handling and cross-domain context |

Structure:
```python
# Registry name constants
SYSTEM_PROMPT_NAME           = "ck-system-prompt"
SMALLTALK_PROMPT_NAME        = "ck-smalltalk-prompt"
INTENT_ROUTER_NAME           = "ck-intent-router"
COHERENCE_JUDGE_NAME         = "ck-judge-conversation-coherence"
CONTEXT_RETENTION_JUDGE_NAME = "ck-judge-context-retention"
DOMAIN_ROUTING_JUDGE_NAME    = "ck-judge-domain-routing"

# Template source strings (used only for initial registration)
_SYSTEM_PROMPT_TEMPLATE = """..."""
_SMALLTALK_TEMPLATE = """..."""
_INTENT_ROUTER_TEMPLATE = """..."""
_COHERENCE_TEMPLATE = """..."""
_CONTEXT_RETENTION_TEMPLATE = """..."""
_DOMAIN_ROUTING_TEMPLATE = """..."""

# Registration (idempotent)
def _register_if_missing(name: str, template: str, commit_message: str) -> None:
    if mlflow.genai.load_prompt(name, allow_missing=True) is None:
        mlflow.genai.register_prompt(name, template, commit_message=commit_message)

def register_all_prompts() -> None: ...

# Public accessors — always fetch from the registry by name
def get_system_prompt() -> str: ...
def get_smalltalk_prompt() -> str: ...
def get_intent_router_prompt() -> str: ...
def get_coherence_judge_instructions() -> str: ...
def get_context_retention_judge_instructions() -> str: ...
def get_domain_routing_judge_instructions() -> str: ...
```

Judge instruction templates (same content as the original plan but loaded from registry):

**Coherence judge** (`ck-judge-conversation-coherence`):
```
Evaluate the coherence of this Casper's Kitchens customer conversation.

{{ conversation }}

Does the conversation flow logically? Does the assistant:
- Give responses that follow from prior turns?
- Avoid contradictions (e.g., recommending a dish it said was unavailable earlier)?
- Handle topic shifts (orders → menus → safety) without losing context?

Value: True if coherent, False if there are significant coherence issues.
Rationale: 2-3 sentences covering logical flow, consistency, and continuity.
```

**Context retention judge** (`ck-judge-context-retention`):
```
Evaluate context retention in this Casper's Kitchens assistant conversation.

{{ conversation }}

Key facts to watch for: order IDs, allergen constraints, location, dietary preferences,
complaint details, budget. Did the assistant carry these forward across turns?

EXCELLENT: All key facts remembered and applied in every relevant turn.
GOOD: Most facts retained; minor lapses that don't derail the conversation.
FAIR: Occasionally re-asks for info already provided; forgets constraints.
POOR: Treats each turn independently; ignores prior user-stated constraints.

Value: excellent, good, fair, or poor.
Rationale: Cite specific turns where context was or wasn't retained.
```

**Domain routing judge** (`ck-judge-domain-routing`):
```
Evaluate domain-shift handling in this Casper's Kitchens conversation.

{{ conversation }}

The conversation may span orders, menus, food safety, and recommendations.
Did the assistant recognize domain shifts and route them appropriately?
Did it carry relevant cross-domain context (e.g., a peanut allergy mentioned
during a menu question should still apply during a recommendation question)?

Value: True if domain shifts were handled gracefully, False otherwise.
Rationale: Cite specific turns where domain shifts occurred and how they were handled.
```

---

### Step 2 — `endpoint_clients.py`

Thin wrappers that call each Model Serving endpoint via the OpenAI SDK.
Key improvements over the original plan:

1. **Reuse the `OpenAI` client** passed in from the agent class (created once via
   `devconnect.providers.get_client("databricks", ...)`), don't create per-call.
2. **Add `@mlflow.trace(span_type=SpanType.TOOL)`** so endpoint calls appear as child
   spans in the trace (same pattern as `search_tool.py` in the restaurant bot).
3. **Complaint agent fallback**: `call_complaint_agent()` catches connection/404 errors
   and returns a structured fallback message.

```python
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI


@mlflow.trace(span_type=SpanType.TOOL, name="call_supervisor")
def call_supervisor(
    message: str,
    history: list[dict],
    client: OpenAI,
    endpoint_name: str,
) -> str:
    """
    Call the Multi-Agent Supervisor endpoint.
    Handles: menu questions, nutrition, allergens, inspection/safety, recommendations.
    The supervisor internally routes to Genie, Menu KA, or Inspection KA.

    history: list of {"role": "user"/"assistant", "content": "..."} from prior turns.
             Pass this so the supervisor has context (e.g., the peanut allergy from turn 1).
    """
    messages = history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=endpoint_name,
        messages=messages,
    )
    return response.choices[0].message.content


@mlflow.trace(span_type=SpanType.TOOL, name="call_complaint_agent")
def call_complaint_agent(
    message: str,
    history: list[dict],
    client: OpenAI,
    endpoint_name: str,
) -> str:
    """
    Call the Complaint/Refund Agent endpoint.
    Handles: order status, delivery delays, refund requests.
    The agent internally calls UC tools: get_order_overview(), get_order_timing(), etc.

    Returns a fallback message if the endpoint is unavailable.
    """
    try:
        messages = history + [{"role": "user", "content": message}]
        response = client.chat.completions.create(
            model=endpoint_name,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "I apologize, but I'm currently unable to look up order information. "
            "For order issues, please contact Casper's Kitchens support directly."
        )
```

> **Note on history passing:** The supervisor and complaint agent are stateless —
> they don't remember prior turns. The multi-turn bot must reconstruct context
> by passing the accumulated `history` list on every call. This is exactly the
> conversation management the bot adds on top.

---

### Step 3 — `intent_router.py`

Classifies each user message so the bot knows which endpoint to call.
Key improvements over the original plan:

1. **Loads router prompt from the MLflow Prompt Registry** via `get_intent_router_prompt()`
   so it can be tuned in the MLflow UI without code changes.
2. **`@mlflow.trace(span_type=SpanType.CHAIN)`** so the classification appears as a
   child span in the trace.

```python
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI
from typing import Literal

from devconnect.caspers_kitchen_bot.prompts import get_intent_router_prompt

Intent = Literal["order", "menu_safety", "smalltalk"]


@mlflow.trace(span_type=SpanType.CHAIN, name="classify_intent")
def classify_intent(message: str, client: OpenAI, model: str) -> Intent:
    prompt = get_intent_router_prompt().format(message=message)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip().lower()
    if raw in ("order", "menu_safety", "smalltalk"):
        return raw
    # Default: if ambiguous, send to supervisor (safe fallback)
    return "menu_safety"
```

---

### Step 4 — `scenarios.py`

Follow the exact pattern from `devconnect/restaurant_research_bot/scenarios.py`:
`_session_counters` + `_next_session_id()` + `get_all_scenarios()` + `get_scenario_by_name()`.

Each scenario dict adds a `requires_complaint_agent: bool` field.

```python
from collections import defaultdict
from typing import Any, Dict, List

_session_counters: Dict[str, int] = defaultdict(int)

def _next_session_id(scenario: str) -> str:
    _session_counters[scenario] += 1
    return f"session-{scenario}-{_session_counters[scenario]:03d}"
```

**Five scenarios:**

| Short key | Name | Intent sequence | Requires complaint agent |
|-----------|------|----------------|------------------------|
| `food_safety` | Food Safety Inquiry | all `menu_safety` | No |
| `order_complaint` | Delayed Delivery Complaint | all `order` | **Yes** |
| `allergen` | Allergen-Safe Recommendation | all `menu_safety` | No |
| `cross_domain` | Cross-Domain: Menu + Safety | all `menu_safety` | No |
| `mixed` | Mixed: Complaint Then Menu | `order` → `menu_safety` | **Yes** |

```python
def get_scenario_food_safety() -> Dict[str, Any]:
    return {
        "name": "Food Safety Inquiry",
        "session_id": _next_session_id("food_safety"),
        "expected_intents": ["menu_safety", "menu_safety", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "good",
        "requires_complaint_agent": False,
        "messages": [
            "I heard one of your Chicago locations had food safety issues recently.",
            "What exactly was the violation and when did it happen?",
            "Was there a corrective action taken? Is it safe to order from there now?",
            "What is the current inspection score for that Chicago location?",
        ],
        # Backend: menu_supervisor --> Inspection KA --> Chicago PDF reports
    }


def get_scenario_order_complaint() -> Dict[str, Any]:
    return {
        "name": "Delayed Delivery Complaint",
        "session_id": _next_session_id("order_complaint"),
        "expected_intents": ["order", "order", "order", "order"],
        "expected_coherence": True,
        "expected_retention": "excellent",
        "requires_complaint_agent": True,
        "messages": [
            "My Wok This Way order was supposed to arrive an hour ago. Order ID ORD-7842.",
            "The app shows delivered but I never received it. I'm at 123 Main St SF.",
            "Yes, I checked with my neighbors and building lobby. Nothing.",
            "What are my options for a refund or reorder?",
        ],
        # Backend: complaint_agent --> get_order_overview(ORD-7842)
        #                          --> get_order_timing(ORD-7842)
        #                          --> get_location_timings("san_francisco")
    }


def get_scenario_allergen_recommendation() -> Dict[str, Any]:
    return {
        "name": "Allergen-Safe Recommendation",
        "session_id": _next_session_id("allergen"),
        "expected_intents": ["menu_safety", "menu_safety", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "excellent",
        "requires_complaint_agent": False,
        "messages": [
            "I'm looking for something healthy for dinner tonight.",
            "I'm vegetarian and have a severe peanut allergy.",
            "I'm near the San Francisco location. Budget around $15.",
            "The NootroNourish menu sounds interesting -- are any of their bowls peanut-free?",
        ],
        # Critical test: peanut allergy from turn 2 must be in history sent on turn 4
    }


def get_scenario_cross_domain() -> Dict[str, Any]:
    return {
        "name": "Cross-Domain: Menu + Safety + Recommendation",
        "session_id": _next_session_id("cross_domain"),
        "expected_intents": ["menu_safety", "menu_safety", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "excellent",
        "requires_complaint_agent": False,
        "messages": [
            "What Asian brands do you run and what cuisines do they cover?",
            "I have a shellfish allergy -- which Wok This Way dishes should I avoid?",
            "Is Wok This Way in San Francisco safe from a food safety standpoint?",
            "Given my shellfish allergy, which of your brands would you recommend tonight?",
        ],
        # Turn 1: Genie --> brand table
        # Turn 2: Menu KA --> Wok This Way PDF allergen info
        # Turn 3: Inspection KA --> SF inspection reports
        # Turn 4: Genie (allergen_free) + shellfish constraint from turn 2 in history
    }


def get_scenario_mixed_order_and_menu() -> Dict[str, Any]:
    return {
        "name": "Mixed: Complaint Then Menu Exploration",
        "session_id": _next_session_id("mixed"),
        "expected_intents": ["order", "order", "menu_safety", "menu_safety"],
        "expected_coherence": True,
        "expected_retention": "good",
        "requires_complaint_agent": True,
        "messages": [
            "My order ORD-5521 was missing the Kung Pao Chicken.",
            "Yes, I'd like a partial refund for the missing item.",
            "Actually, while I have you -- is Kung Pao Chicken high in sodium?",
            "And are there lower-sodium alternatives at Wok This Way?",
        ],
        # Tests intent routing flip from ORDER to MENU_SAFETY mid-conversation
        # Tests whether history (the specific dish name) carries into menu query
    }


def get_all_scenarios() -> List[Dict[str, Any]]:
    return [
        get_scenario_food_safety(),
        get_scenario_order_complaint(),
        get_scenario_allergen_recommendation(),
        get_scenario_cross_domain(),
        get_scenario_mixed_order_and_menu(),
    ]


def get_scenario_by_name(name: str) -> Dict[str, Any]:
    lookup = {s["name"]: s for s in get_all_scenarios()}
    short_lookup = {
        "food_safety":     get_scenario_food_safety(),
        "order_complaint": get_scenario_order_complaint(),
        "allergen":        get_scenario_allergen_recommendation(),
        "cross_domain":    get_scenario_cross_domain(),
        "mixed":           get_scenario_mixed_order_and_menu(),
    }
    if name in lookup:
        return lookup[name]
    if name in short_lookup:
        return short_lookup[name]
    raise ValueError(
        f"Unknown scenario: '{name}'. "
        f"Available: {list(short_lookup.keys())}"
    )
```

---

### Step 5 — `caspers_kitchen_agent_cls.py`

The agent class. Follow the structure of `devconnect/restaurant_research_bot/restaurant_research_agent_cls.py`.

```python
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from typing import Dict, List, Literal

from devconnect.config import AgentConfig
from devconnect.providers import get_client
from devconnect.caspers_kitchen_bot.prompts import (
    register_all_prompts,
    get_system_prompt,
    get_smalltalk_prompt,
    get_coherence_judge_instructions,
    get_context_retention_judge_instructions,
    get_domain_routing_judge_instructions,
)
from devconnect.caspers_kitchen_bot.intent_router import classify_intent
from devconnect.caspers_kitchen_bot.endpoint_clients import (
    call_supervisor,
    call_complaint_agent,
)


class CaspersKitchenAgent:

    def __init__(
        self,
        config: AgentConfig,              # reused from devconnect.config
        supervisor_endpoint: str,          # e.g. "caspersdev-menu-supervisor"
        complaint_endpoint: str | None,    # None if complaints target not deployed
        router_model: str,                 # fast model for intent classification
        judge_model: str,                  # model URI for session-level judges
        debug: bool = False,
    ):
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config
        self.supervisor_endpoint = supervisor_endpoint
        self.complaint_endpoint = complaint_endpoint
        self.router_model = router_model
        self.judge_model = judge_model
        self.debug = debug
        self.session_histories: Dict[str, List[dict]] = {}
        self._init_judges()

    def _init_judges(self) -> None:
        register_all_prompts()
        self.coherence_judge = make_judge(
            name="conversation_coherence",
            model=self.judge_model,
            instructions=get_coherence_judge_instructions(),
            feedback_value_type=bool,
        )
        self.context_judge = make_judge(
            name="context_retention",
            model=self.judge_model,
            instructions=get_context_retention_judge_instructions(),
            feedback_value_type=Literal["excellent", "good", "fair", "poor"],
        )
        self.routing_judge = make_judge(
            name="domain_routing",
            model=self.judge_model,
            instructions=get_domain_routing_judge_instructions(),
            feedback_value_type=bool,
        )

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="handle_caspers_message")
    def handle_message(self, message: str, session_id: str) -> str:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id}
        )

        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        history = self.session_histories[session_id]

        # Classify intent (appears as CHAIN child span)
        intent = classify_intent(message, self.client, self.router_model)

        # Route to the right backend (each appears as TOOL child span)
        if intent == "order":
            if self.complaint_endpoint is not None:
                reply = call_complaint_agent(
                    message, history, self.client, self.complaint_endpoint
                )
            else:
                reply = (
                    "I can help with menu and food safety questions. "
                    "For order issues, please contact Casper's Kitchens support directly."
                )
        elif intent == "menu_safety":
            reply = call_supervisor(
                message, history, self.client, self.supervisor_endpoint
            )
        else:  # smalltalk
            reply = self._smalltalk(message, history)

        # Persist only user/assistant pairs (same as restaurant bot)
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        return reply

    def _smalltalk(self, message: str, history: list) -> str:
        messages = [{"role": "system", "content": get_smalltalk_prompt()}]
        messages += history
        messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model=self.router_model,
            messages=messages,
            max_tokens=150,
        )
        return response.choices[0].message.content

    def run_conversation(self, messages: List[str], session_id: str) -> List[str]:
        replies = []
        for i, msg in enumerate(messages):
            print(f"\nTurn {i + 1}/{len(messages)}")
            print(f"  User: {msg}")
            reply = self.handle_message(msg, session_id)
            print(f"  Bot:  {reply}")
            replies.append(reply)
        return replies

    def evaluate_session(self, session_id: str, run_id: str) -> Dict:
        experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment)
        if experiment is None:
            raise ValueError(
                f"MLflow experiment '{self.config.mlflow_experiment}' not found."
            )

        session_traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            filter_string=f"run_id = '{run_id}'",
        )

        if len(session_traces) == 0:
            raise ValueError(f"No traces found for run_id '{run_id}'")

        print(f"\nEvaluating session '{session_id}' ({len(session_traces)} traces)...")

        try:
            from IPython.utils.capture import capture_output as _cap
            with _cap():
                eval_results = mlflow.genai.evaluate(
                    data=session_traces,
                    scorers=[self.coherence_judge, self.context_judge, self.routing_judge],
                )
        except ImportError:
            eval_results = mlflow.genai.evaluate(
                data=session_traces,
                scorers=[self.coherence_judge, self.context_judge, self.routing_judge],
            )

        result_df = eval_results.result_df

        def extract(keyword: str, suffix: str):
            cols = [c for c in result_df.columns
                    if keyword in c.lower() and suffix in c.lower()]
            if not cols:
                return None
            series = result_df[cols[0]].dropna()
            return series.iloc[0] if len(series) > 0 else None

        def extract_reason(keyword: str):
            val = extract(keyword, "/reason")
            return val if val is not None else (extract(keyword, "/justification") or "")

        return {
            "session_id": session_id,
            "num_traces": len(session_traces),
            "coherence": {
                "feedback_value": extract("coherence", "/value"),
                "rationale":      extract_reason("coherence"),
                "passed":         extract("coherence", "/value"),
            },
            "context_retention": {
                "feedback_value": extract("context", "/value"),
                "rationale":      extract_reason("context"),
            },
            "domain_routing": {
                "feedback_value": extract("routing", "/value"),
                "rationale":      extract_reason("routing"),
                "passed":         extract("routing", "/value"),
            },
        }
```

---

### Step 6 — `caspers_kitchen_agent.py` (CLI Entry Point)

Follow the pattern from `devconnect/restaurant_research_bot/restaurant_research_agent.py`.

```python
"""
CLI entry point for the Casper's Kitchen multi-turn bot evaluation.

Usage:
  # Set credentials
  export DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
  export DATABRICKS_TOKEN=dapi...

  # Run all scenarios
  uv run mlflow-caspers-kitchen-bot

  # Run a specific scenario
  uv run mlflow-caspers-kitchen-bot --scenario food_safety

  # Skip complaint agent scenarios
  uv run mlflow-caspers-kitchen-bot --no-complaint-agent

  # View results
  mlflow ui
"""

import argparse
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from devconnect.config import AgentConfig
from devconnect.mlflow_config import setup_mlflow_tracking
from devconnect.caspers_kitchen_bot.caspers_kitchen_agent_cls import CaspersKitchenAgent
from devconnect.caspers_kitchen_bot.scenarios import get_all_scenarios, get_scenario_by_name

EXPERIMENT_NAME = "caspers-multi-turn-bot"


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser(
        description="Casper's Kitchen multi-turn bot with MLflow session evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--catalog", default="caspersdev",
                        help="UC catalog name (default: caspersdev)")
    parser.add_argument("--model", default="databricks-gpt-4o-mini",
                        help="Router/smalltalk model (default: databricks-gpt-4o-mini)")
    parser.add_argument("--judge-model", default="databricks-gemini-2-5-flash",
                        help="Judge model (default: databricks-gemini-2-5-flash)")
    parser.add_argument("--scenario", default=None,
                        help="Run a single scenario: food_safety, order_complaint, allergen, cross_domain, mixed")
    parser.add_argument("--no-complaint-agent", action="store_true",
                        help="Skip scenarios requiring the complaint agent")
    parser.add_argument("--debug", action="store_true",
                        help="Print evaluation DataFrame columns and extra details")
    args = parser.parse_args()

    judge_model_uri = f"openai:/{args.judge_model}"
    complaint_endpoint = None if args.no_complaint_agent else f"{args.catalog}-complaint-agent"

    print("\n" + "=" * 60)
    print("Casper's Kitchen Bot  |  MLflow Session Evaluation")
    print("=" * 60)
    print(f"\n  Catalog:            {args.catalog}")
    print(f"  Router model:       {args.model}")
    print(f"  Judge model:        {args.judge_model}")
    print(f"  Complaint agent:    {'disabled' if complaint_endpoint is None else complaint_endpoint}")
    print(f"  Experiment:         {EXPERIMENT_NAME}")

    setup_mlflow_tracking(experiment_name=EXPERIMENT_NAME, enable_autolog=True)

    config = AgentConfig(
        model=args.model,
        provider="databricks",
        mlflow_experiment=EXPERIMENT_NAME,
    )

    print("\n  Initialising judges...")
    agent = CaspersKitchenAgent(
        config=config,
        supervisor_endpoint=f"{args.catalog}-menu-supervisor",
        complaint_endpoint=complaint_endpoint,
        router_model=args.model,
        judge_model=judge_model_uri,
        debug=args.debug,
    )

    scenarios = (
        [get_scenario_by_name(args.scenario)]
        if args.scenario
        else get_all_scenarios()
    )

    for scenario in scenarios:
        # Skip scenarios that need the complaint agent when unavailable
        if scenario.get("requires_complaint_agent") and complaint_endpoint is None:
            print(f"\n  SKIP: '{scenario['name']}' (requires complaint agent)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario['name']}")
        print(f"Session:  {scenario['session_id']}")
        print("=" * 60)

        with mlflow.start_run(run_name=scenario["name"]) as run:
            agent.run_conversation(scenario["messages"], scenario["session_id"])
            try:
                results = agent.evaluate_session(
                    scenario["session_id"], run.info.run_id
                )
            except Exception as exc:
                print(f"\n  Evaluation failed: {exc}")
                continue

        coh = results["coherence"]
        ctx = results["context_retention"]
        rte = results["domain_routing"]

        print(f"\n{'─' * 40}")
        print(f"  Coherence:         {'PASS' if coh['passed'] else 'FAIL'}  ({coh['feedback_value']})")
        if coh["rationale"]:
            print(f"    {coh['rationale']}")
        print(f"  Context Retention: {str(ctx['feedback_value']).upper()}")
        if ctx["rationale"]:
            print(f"    {ctx['rationale']}")
        print(f"  Domain Routing:    {'PASS' if rte['passed'] else 'FAIL'}  ({rte['feedback_value']})")
        if rte["rationale"]:
            print(f"    {rte['rationale']}")

    print(f"\n{'=' * 60}")
    print(f"Done. View traces: mlflow ui  →  '{EXPERIMENT_NAME}' experiment")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
```

---

### Step 7 — `__init__.py`

```python
"""
Multi-turn Casper's Kitchen bot with MLflow session-level evaluation.

Demonstrates MLflow session-level evaluation for a multi-agent routing bot:
- Intent-based routing to deployed Databricks Model Serving endpoints
- Session tracking with mlflow.update_current_trace()
- Session-level judges using {{ conversation }} template
- mlflow.genai.evaluate() with coherence, context retention, and domain routing judges
"""

from devconnect.caspers_kitchen_bot.caspers_kitchen_agent_cls import CaspersKitchenAgent
from devconnect.caspers_kitchen_bot.scenarios import (
    get_scenario_food_safety,
    get_scenario_order_complaint,
    get_scenario_allergen_recommendation,
    get_scenario_cross_domain,
    get_scenario_mixed_order_and_menu,
    get_all_scenarios,
    get_scenario_by_name,
)

__all__ = [
    "CaspersKitchenAgent",
    "get_scenario_food_safety",
    "get_scenario_order_complaint",
    "get_scenario_allergen_recommendation",
    "get_scenario_cross_domain",
    "get_scenario_mixed_order_and_menu",
    "get_all_scenarios",
    "get_scenario_by_name",
]
```

---

### Step 8 — `pyproject.toml`

Add a new script entry point:

```toml
[project.scripts]
mlflow-restaurant-research-bot = "devconnect.restaurant_research_bot.restaurant_research_agent:main"
mlflow-caspers-kitchen-bot = "devconnect.caspers_kitchen_bot.caspers_kitchen_agent:main"
```

No new dependencies needed — uses `openai` (already present) to call Databricks endpoints.

---

### Step 9 — Update `CLAUDE.md`

Add a section for the Casper's Kitchen bot parallel to the restaurant bot:
- Layout of `caspers_kitchen_bot/` directory
- Running instructions (`uv run mlflow-caspers-kitchen-bot --catalog caspersdev`)
- Required environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)
- Scenarios table
- Complaint agent optionality behavior

---

## Conversation Flow Diagram (Food Safety Example)

```
User: "I heard Chicago had food safety issues"
  |
  v
classify_intent()  --> "menu_safety"  [CHAIN child span in trace]
  |
  v
call_supervisor(message, history=[], client, "caspersdev-menu-supervisor")  [TOOL child span]
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
MLflow trace tagged: mlflow.trace.session = "session-food_safety-001"

User: "What corrective action was required?"
  |
  v
classify_intent()  --> "menu_safety"  [CHAIN child span]
  |
  v
call_supervisor(message, history=[... 2 prior messages ...], client, endpoint)  [TOOL child span]
  ^
  The history is what gives the supervisor context that we're talking about Chicago.
  The supervisor (or Inspection KA) uses this to answer without re-stating the question.
  |
  v
Returns: "Corrective action: rearrange refrigeration, retrain staff..."
MLflow trace tagged: mlflow.trace.session = "session-food_safety-001"
```

---

## Session Evaluation Flow

```
After all turns are complete:

mlflow.search_traces(filter="run_id = '<run_id>'")
  --> [trace_turn_1, trace_turn_2, trace_turn_3, trace_turn_4]

coherence_judge(session=traces)   --> {{ conversation }} aggregates all 4 turns
context_judge(session=traces)     --> Did bot remember we were discussing Chicago?
routing_judge(session=traces)     --> Did every turn get routed to the right endpoint?
```

---

## Prerequisites Before Running

The following Casper's stages must have been deployed (via `databricks bundle deploy -t menus`):

| Stage completed | Provides | Required |
|-----------------|---------|----------|
| `menu_knowledge_agent` | `{CATALOG}-menu-knowledge` endpoint | Yes |
| `inspection_knowledge_agent` | `{CATALOG}-inspection-knowledge` endpoint | Yes |
| `menu_genie` | Genie space ID | Yes |
| `menu_supervisor` | `{CATALOG}-menu-supervisor` endpoint | Yes |
| `complaint_agent` | `{CATALOG}-complaint-agent` endpoint + UC tools | Optional (complaints target) |

---

## Key Differences from the Restaurant Research Bot

| Aspect | Restaurant Research Bot | Casper's Kitchen Bot |
|--------|----------------------|---------------------|
| Data access | Tavily web search (stateless) | Delegates to deployed Databricks endpoints |
| Core loop | Tool-use loop (LLM decides when to search) | Intent router → endpoint delegation |
| Provider | OpenAI or Databricks (configurable) | Always Databricks |
| Shared code | `AgentConfig`, `get_client()`, `setup_mlflow_tracking()` | Same — reused directly |
| Prompts | 4 registry-backed (`system`, 3 judges) | 6 registry-backed (`system`, `smalltalk`, `intent-router`, 3 judges) |
| 3rd judge | `search_quality` (necessary/unnecessary/skipped) | `domain_routing` (bool — correct routing?) |
| New files | `search_tool.py` | `endpoint_clients.py`, `intent_router.py` |
| Scenarios | Generic restaurant/food/allergen | Grounded in Casper's real data and endpoints |
| Complaint agent | N/A | Optional; graceful fallback when unavailable |

---

## Implementation Order

1. [ ] `prompts.py` — registry-backed prompts (6 prompts, `ck-` prefix)
2. [ ] `endpoint_clients.py` — traced endpoint wrappers with complaint agent fallback
3. [ ] `intent_router.py` — traced intent classification from registry prompt
4. [ ] `scenarios.py` — 5 test cases with `requires_complaint_agent` field
5. [ ] `caspers_kitchen_agent_cls.py` — wire everything together with MLflow tracing
6. [ ] `caspers_kitchen_agent.py` — CLI entry point with `--no-complaint-agent` flag
7. [ ] `__init__.py` — public exports
8. [ ] `pyproject.toml` — add `mlflow-caspers-kitchen-bot` script entry point
9. [ ] `CLAUDE.md` — add Casper's Kitchen bot documentation section
10. [ ] `caspers_kitchen_agent.ipynb` — interactive notebook version (later)
11. [ ] Run all scenarios, review in MLflow UI, tune judge instructions if needed

---

## Verification

1. **Module import**: `python -c "from devconnect.caspers_kitchen_bot import CaspersKitchenAgent"`
2. **Prompt registration**: Run `register_all_prompts()` → verify 6 `ck-*` prompts in MLflow UI
3. **Endpoint connectivity**: Standalone `call_supervisor("What brands do you have?", [], client, endpoint)` returns a real response
4. **Intent classification**: Test `classify_intent()` against each scenario's first message
5. **Single scenario**: `uv run mlflow-caspers-kitchen-bot --scenario food_safety` → traces in MLflow UI with session tags, judges produce scores
6. **All scenarios**: `uv run mlflow-caspers-kitchen-bot` → all 5 (or 3 without complaint agent) run with evaluation results
7. **Complaint fallback**: `uv run mlflow-caspers-kitchen-bot --no-complaint-agent` → order scenarios skipped with warning, menu scenarios pass

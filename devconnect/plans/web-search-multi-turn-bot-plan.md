# Plan: Multi-Turn Conversational Bot with Web Search (No Pre-loaded Data)

## Core Difference from the Casper's Kitchen Plan

The Casper's Kitchen bot delegates to deployed endpoints that already know the data.
This bot has **no pre-loaded data at all** — when the LLM needs a fact it doesn't know,
it calls a `web_search` tool at runtime. The LLM is both the reasoner and the router.

This replaces:

| Casper's Kitchen approach | This approach |
|--------------------------|---------------|
| Intent router → pick an endpoint | LLM decides when to search and what to search for |
| `call_supervisor()` / `call_complaint_agent()` | `web_search(query)` tool call |
| Static data in Delta tables and PDFs | Live web results fetched per turn |
| Requires Databricks deployment | Runs anywhere: local, any cloud, any LLM |

The MLflow session tracking and `mlflow.genai.evaluate()` pattern is identical.

---

## Architecture

```
User (multi-turn chat)
        |
        v
WebSearchAgent.handle_message(message, session_id)
  |
  |- @mlflow.trace  +  mlflow.update_current_trace(session=session_id)
  |- appends user message to session_histories[session_id]
  |
  v
LLM call with tools=[web_search_tool_schema]
  |
  +-- LLM decides: "I can answer from context" --> final response, done
  |
  +-- LLM decides: "I need to search"
        |
        v
        tool_calls = response.choices[0].message.tool_calls
        for each tool_call:
            query = tool_call.function.arguments["query"]
            result = web_search(query)          <-- external search API
            append tool result to messages
        |
        v
        LLM call again with updated messages (tool results included)
        |
        +-- repeat until no more tool_calls (loop)
        |
        v
        final response
  |
  v
Append assistant reply to session_histories[session_id]
  |
  v
Return reply  (MLflow trace closed, tagged with session_id)
```

The tool-use loop is standard OpenAI function calling. The LLM drives it; the agent
just executes calls and feeds results back until the LLM stops requesting tools.

---

## Web Search API Options

The `web_search` tool needs a real search API backend. Choose one:

| API | Free tier | Best for | Notes |
|-----|-----------|----------|-------|
| **Tavily** | 1,000 req/month | LLM use cases | Purpose-built for agents; returns clean excerpts, not raw HTML |
| **Brave Search** | 2,000 req/month | General search | Good coverage, privacy-respecting |
| **SerpAPI** | 100 req/month | Google results | Wraps Google; higher fidelity but lower free quota |
| **Bing Search (Azure)** | Pay-as-you-go | Enterprise | Requires Azure subscription |

**Recommendation: Tavily.** Its results are pre-cleaned for LLM consumption (no HTML
parsing needed) and the API is the simplest to use. One environment variable: `TAVILY_API_KEY`.

> **Important: search APIs are stateless.** Tavily, SerpAPI, and every other search API
> receive only the query string you send them. They have no knowledge of the conversation,
> prior searches, or prior results. Multi-turn context lives entirely in `session_histories`
> inside the agent. The LLM must construct a self-contained query that bakes in any relevant
> constraints or references from prior turns — the system prompt (Step 3) explicitly
> instructs it to do this. The `search_quality` judge (Step 3) evaluates whether it did.

---

## File Structure

```
demos/web-search-bot/           (if adding to Casper's Kitchen repo)
  OR
web_search_bot/                 (standalone project)
├── bot.py                      # CLI entry point
├── bot_cls.py                  # WebSearchAgent class
├── search_tool.py              # web_search() function + OpenAI tool schema
├── prompts.py                  # System prompt + judge instructions
├── scenarios.py                # Conversation test cases
└── bot.ipynb                   # Interactive notebook version
```

---

## Step-by-Step Implementation

### Step 1 — `search_tool.py`

The web search function and the OpenAI-compatible tool schema the LLM uses to call it.

```python
import os
import requests
from mlflow.entities import SpanType
import mlflow

# OpenAI function-calling schema -- passed to the LLM in every request
WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this when you need "
            "facts you don't know or that may have changed: restaurant menus, "
            "business hours, food safety ratings, nutrition info, current news."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and concise."
                }
            },
            "required": ["query"]
        }
    }
}


@mlflow.trace(span_type=SpanType.TOOL, name="web_search")
def web_search(query: str, max_results: int = 3) -> str:
    """
    Execute a web search via Tavily and return a formatted string of results.
    Decorated with @mlflow.trace so each search appears as a child span
    inside the parent handle_message() trace.
    """
    api_key = os.environ["TAVILY_API_KEY"]
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",   # "advanced" is slower but more thorough
            "include_answer": True,    # Tavily's own LLM-generated summary
        },
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    # Format results as a readable string for the LLM
    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}")
    for r in data.get("results", [])[:max_results]:
        parts.append(f"- [{r['title']}]({r['url']})\n  {r['content'][:300]}")
    return "\n\n".join(parts) if parts else "No results found."
```

---

### Step 2 — `bot_cls.py`

The agent class. The tool-use loop lives in `handle_message()`.

```python
import json
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from openai import OpenAI
from typing import Dict, List, Literal
from search_tool import web_search, WEB_SEARCH_TOOL_SCHEMA
from prompts import (
    get_system_prompt,
    get_coherence_judge_instructions,
    get_context_retention_judge_instructions,
    get_search_quality_judge_instructions,
)


class WebSearchAgent:

    def __init__(
        self,
        client: OpenAI,
        model: str,
        judge_model: str,
        mlflow_experiment: str = "web-search-bot",
    ):
        self.client = client
        self.model = model
        self.mlflow_experiment = mlflow_experiment
        self.session_histories: Dict[str, List[dict]] = {}
        self._init_judges(judge_model)

    def _init_judges(self, judge_model: str):
        """All judges use {{ conversation }} -- automatically session-level."""
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
        self.search_quality_judge = make_judge(
            name="search_quality",
            model=judge_model,
            instructions=get_search_quality_judge_instructions(),
            feedback_value_type=Literal["necessary", "unnecessary", "missing"]
        )

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="handle_message")
    def handle_message(self, message: str, session_id: str) -> str:
        # Tag this trace with the session so MLflow can group turns
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id}
        )

        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        history = self.session_histories[session_id]

        # Build message list: system + history + new user message
        messages = [{"role": "system", "content": get_system_prompt()}]
        messages += history
        messages.append({"role": "user", "content": message})

        # Tool-use loop: keep calling the LLM until it stops requesting tools
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[WEB_SEARCH_TOOL_SCHEMA],
                tool_choice="auto",   # LLM decides whether to search
                max_tokens=500
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                # LLM wants to search -- execute each tool call and feed results back
                messages.append(choice.message)   # assistant message with tool_calls

                for tool_call in choice.message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = web_search(args["query"])   # child span in MLflow trace

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                # Loop back: LLM now has the search results and will form a response

            else:
                # LLM is done -- no more tool calls
                reply = choice.message.content
                break

        # Update session history with the full exchange
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        return reply

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
        experiment = mlflow.get_experiment_by_name(self.mlflow_experiment)
        session_traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            filter_string=f"run_id = '{run_id}'"
        )

        if len(session_traces) == 0:
            raise ValueError(f"No traces found for run_id '{run_id}'")

        print(f"\nEvaluating session '{session_id}' ({len(session_traces)} traces)...")

        eval_results = mlflow.genai.evaluate(
            data=session_traces,
            scorers=[self.coherence_judge, self.context_judge, self.search_quality_judge]
        )

        result_df = eval_results.result_df

        def extract(keyword, suffix):
            cols = [c for c in result_df.columns
                    if keyword in c.lower() and suffix in c.lower()]
            if not cols:
                return None
            series = result_df[cols[0]].dropna()
            return series.iloc[0] if len(series) > 0 else None

        return {
            "session_id":  session_id,
            "num_traces":  len(session_traces),
            "coherence": {
                "feedback_value": extract("coherence", "/value"),
                "rationale":      extract("coherence", "/justification") or "",
                "passed":         extract("coherence", "/value"),
            },
            "context_retention": {
                "feedback_value": extract("context", "/value"),
                "rationale":      extract("context", "/justification") or "",
            },
            "search_quality": {
                "feedback_value": extract("search", "/value"),
                "rationale":      extract("search", "/justification") or "",
            },
        }
```

---

### Step 3 — `prompts.py`

System prompt keeps the bot grounded. The third judge (`search_quality`) is new here —
it evaluates whether the bot searched when it should have and didn't search when it
shouldn't have (i.e., it didn't waste searches on things the LLM already knew).

```python
def get_system_prompt() -> str:
    return """You are a helpful assistant that can search the web to answer questions.

Guidelines:
- Use the web_search tool when you need current information, specific facts,
  business details, menus, hours, ratings, or anything you're not confident about.
- Do NOT search for things you already know well (basic facts, general knowledge).
- Remember everything the user has told you across turns. If they mention a dietary
  restriction in turn 1, apply it automatically in turn 4 without re-asking.
- After searching, synthesize the results into a clear, direct answer.
  Do not dump raw search results at the user.
- Keep responses concise -- under 150 words unless listing multiple items.

IMPORTANT -- search query construction:
The web_search tool has NO memory of prior turns. It is a stateless API that receives
only the query string you pass it. You are responsible for constructing complete,
self-contained queries that incorporate any relevant context from the conversation.

Bad (loses context):  web_search("restaurants in Seattle")
Good (carries context): web_search("peanut-free Thai restaurants Seattle")
  because the user stated a peanut allergy two turns ago.

Bad (ambiguous reference):  web_search("that restaurant's hours")
Good (resolves the reference): web_search("Piccolo Sogno Chicago hours")
  because the user was asking about Piccolo Sogno earlier in the conversation.

Always resolve pronouns, implicit references, and prior constraints into the query itself.
"""


def get_coherence_judge_instructions() -> str:
    return """Evaluate the coherence of this multi-turn conversation where the assistant
can search the web for information.

{{ conversation }}

Does the conversation flow logically across turns?
- Are responses relevant to what was asked?
- Does the assistant avoid contradicting itself (e.g., saying a restaurant is open,
  then saying it's closed in a later turn without new information)?
- Are search results properly synthesized into coherent answers, not dumped raw?

Value: True if coherent, False if there are significant coherence issues.
Rationale: 2-3 sentences on flow, consistency, and synthesis quality.
"""


def get_context_retention_judge_instructions() -> str:
    return """Evaluate context retention in this multi-turn conversation.

{{ conversation }}

The assistant should remember key constraints and facts from earlier turns:
dietary restrictions, location preferences, budget, prior search findings, etc.

EXCELLENT: All prior constraints applied automatically in every relevant turn.
GOOD: Most context retained; minor lapses that don't derail the conversation.
FAIR: Occasionally re-asks for info already given; forgets stated preferences.
POOR: Treats each turn independently; ignores constraints stated earlier.

Value: excellent, good, fair, or poor.
Rationale: Cite specific turns where context was or wasn't retained correctly.
"""


def get_search_quality_judge_instructions() -> str:
    return """Evaluate web search usage in this multi-turn conversation.

{{ conversation }}

Assess whether the assistant used web search appropriately:

NECESSARY: Searched at the right times (needed current or specific data) and
           skipped searches when general knowledge sufficed. Good balance.
UNNECESSARY: Over-searched -- triggered searches for things the LLM should
             already know (basic facts, common knowledge), wasting latency.
MISSING: Under-searched -- answered with confident specifics (hours, prices,
         ratings, menus) without searching, risking hallucinated facts.

Value: necessary, unnecessary, or missing.
Rationale: Cite specific turns where search use was appropriate or not.
"""
```

---

### Step 4 — `scenarios.py`

Four scenarios covering distinct test cases. Each is designed around a specific
evaluation challenge; the comments document what each turn tests and what the
correct LLM behavior looks like.

**Design rules for scenarios:**
- At least one turn must require a search and at least one must not (tests both directions)
- Context carryover must be tested by having the user state a constraint *once* and
  *not repeat it* at the turn where a search is triggered — otherwise the test is trivial
- Questions must be conversationally related across all four turns, or the
  context retention judge has nothing meaningful to evaluate
- `expected_search_quality: "necessary"` means the bot used search appropriately
  (searched when needed, skipped when not). It does NOT mean "a search was needed."

```python
def get_scenario_restaurant_research():
    """
    Multi-turn restaurant research session.
    User is planning dinner and narrows down based on constraints revealed across turns.
    Turns 1-3 need searches; turn 4 should NOT search (synthesis from prior turns).
    """
    return {
        "name": "Restaurant Research",
        "session_id": "session-restaurant-001",
        "expected_coherence": True,
        "expected_retention": "excellent",
        "expected_search_quality": "necessary",
        "messages": [
            "What are some highly rated Italian restaurants in Chicago's River North neighborhood?",
            "I'm vegetarian -- which of those have good vegetarian options?",
            "What are the current hours for Piccolo Sogno on West Grand?",
            "Given everything we've discussed, which one would you recommend for tonight?"
            # Turn 4: bot should synthesize from prior turns, NOT search again
        ]
    }


def get_scenario_food_safety():
    """
    User researching food safety scores before choosing a restaurant.
    All turns need searches (current health inspection data).
    Tests whether the bot remembers the city and concern across turns.
    """
    return {
        "name": "Food Safety Research",
        "session_id": "session-safety-001",
        "expected_coherence": True,
        "expected_retention": "good",
        "expected_search_quality": "necessary",
        "messages": [
            "How can I check food safety inspection scores for restaurants in San Francisco?",
            "What's the current health inspection rating for Nopa on Divisadero Street?",
            "Are there any recent violation reports for that restaurant?",
            "Based on all this, would you say it's safe to eat there?"
            # Turn 4: synthesize from prior searches, should not need another search
        ]
    }


def get_scenario_nutrition_and_allergens():
    """
    Silent context carryover test: peanut allergy stated ONCE in turn 1, never repeated.
    Turns 2-3 are conversational (no search needed). Turn 4 requires a search AND the
    bot must silently inject the peanut constraint into the query without being reminded.

    This is the hardest scenario for the stateless search problem: the user doesn't
    say "peanut-free" in turn 4, so the LLM must pull that constraint from history
    and bake it into the search query itself.

    Correct turn 4 query: "Thai restaurants Seattle peanut-free"
    Wrong turn 4 query:   "Thai restaurants Seattle"  <-- drops the constraint
    """
    return {
        "name": "Silent Allergen Carryover",
        "session_id": "session-nutrition-001",
        "expected_coherence": True,
        "expected_retention": "excellent",
        "expected_search_quality": "necessary",
        "messages": [
            "I have a severe peanut allergy. What Thai dishes should I completely avoid?",
            "Good to know. What does Pad See Ew typically taste like?",  # no search needed
            "Is it a common dish to find at Thai restaurants?",           # no search needed
            "Can you find me a Thai restaurant in Seattle?"
            # Turn 4: user does NOT say "peanut-free" -- bot must carry constraint silently.
            # Correct search: "Thai restaurants Seattle peanut-free menu"
            # Wrong search:   "Thai restaurants Seattle"  (drops the allergy)
        ]
    }


def get_scenario_no_search_needed():
    """
    Negative test: a coherent conversation that stays within LLM general knowledge.
    All questions build on the same topic (Italian cooking), but none require
    current or specific data -- no web search should be triggered.

    expected_search_quality "necessary" means: the bot correctly made zero searches.
    (Note: "necessary" = appropriate use of search, which here means not searching.)

    Context retention IS testable here: the cuisine topic and stated preference
    (vegetarian, from turn 2) should carry through to turn 4.
    """
    return {
        "name": "No-Search Needed (General Knowledge)",
        "session_id": "session-nosearch-001",
        "expected_coherence": True,
        "expected_retention": "good",
        "expected_search_quality": "necessary",   # correct behavior = zero searches
        "messages": [
            "What are the main pasta shapes used in Italian cooking and what sauces go with them?",
            "I'm vegetarian -- which of those pasta dishes are easy to make meat-free?",
            "What's the difference between pecorino and parmesan in terms of flavor?",
            "Can you suggest a simple vegetarian pasta recipe using what we've discussed?"
            # Turn 4: bot should answer using the vegetarian constraint from turn 2
            # and the pasta/cheese knowledge from turns 1-3 -- no search needed.
        ]
    }
```

---

### Step 5 — `bot.py` (CLI Entry Point)

```python
"""
Usage:
  # OpenAI
  export OPENAI_API_KEY=sk-...
  export TAVILY_API_KEY=tvly-...
  python bot.py --provider openai

  # Databricks
  export DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
  export DATABRICKS_TOKEN=dapi...
  export TAVILY_API_KEY=tvly-...
  python bot.py --provider databricks --model databricks-gpt-4o-mini
"""
import argparse, os, mlflow
from openai import OpenAI
from bot_cls import WebSearchAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",    choices=["openai", "databricks"], default="openai")
    parser.add_argument("--model",       default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--scenario",    default=None)
    args = parser.parse_args()

    # Model defaults per provider
    if args.model is None:
        args.model = "gpt-4o-mini" if args.provider == "openai" else "databricks-gpt-4o-mini"
    if args.judge_model is None:
        args.judge_model = "gpt-4o-mini" if args.provider == "openai" else "databricks-gemini-2-5-flash"

    # Build OpenAI-compatible client
    if args.provider == "databricks":
        client = OpenAI(
            api_key=os.environ["DATABRICKS_TOKEN"],
            base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints"
        )
        judge_model_uri = f"openai:/{args.judge_model}"
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        judge_model_uri = args.judge_model

    mlflow.set_experiment("web-search-bot")

    agent = WebSearchAgent(
        client=client,
        model=args.model,
        judge_model=judge_model_uri,
        mlflow_experiment="web-search-bot",
    )

    from scenarios import get_all_scenarios, get_scenario_by_name
    scenarios = [get_scenario_by_name(args.scenario)] if args.scenario else get_all_scenarios()

    for scenario in scenarios:
        print(f"\n{'='*60}\nScenario: {scenario['name']}\n{'='*60}")

        with mlflow.start_run(run_name=scenario["name"]) as run:
            agent.run_conversation(scenario["messages"], scenario["session_id"])
            results = agent.evaluate_session(scenario["session_id"], run.info.run_id)

        coh = results["coherence"]
        ctx = results["context_retention"]
        srch = results["search_quality"]
        print(f"\nCoherence:       {'PASS' if coh['passed'] else 'FAIL'} ({coh['feedback_value']})")
        print(f"Context:         {str(ctx['feedback_value']).upper()}")
        print(f"Search Quality:  {str(srch['feedback_value']).upper()}")

if __name__ == "__main__":
    main()
```

---

## What the MLflow Trace Looks Like Per Turn

Each `handle_message()` call produces one trace with child spans inside it:

```
handle_message  [CHAT_MODEL span]
  |
  +-- LLM call #1  (LLM decides to search)
  |
  +-- web_search("Italian restaurants Chicago River North")  [TOOL span]
  |      └─ calls Tavily API, returns 3 results
  |
  +-- LLM call #2  (LLM synthesizes results into answer)
```

If the LLM calls `web_search` twice in one turn (e.g., searches twice to compare two
things), both appear as sibling TOOL spans inside the same CHAT_MODEL span.

If the LLM answers from context with no search, the trace has just one LLM call span.

The `{{ conversation }}` judge sees ALL of these traces grouped together by session.

---

## Session Evaluation Flow

```
mlflow.start_run("Restaurant Research")
  |
  +--> handle_message(turn1, "session-restaurant-001")  --> trace_1 (run_id=X)
  +--> handle_message(turn2, "session-restaurant-001")  --> trace_2 (run_id=X)
  +--> handle_message(turn3, "session-restaurant-001")  --> trace_3 (run_id=X)
  +--> handle_message(turn4, "session-restaurant-001")  --> trace_4 (run_id=X)
  |
  +--> evaluate_session("session-restaurant-001", run_id=X)
        |
        +--> mlflow.search_traces(filter="run_id = 'X'")  --> [trace_1..trace_4]
        |
        +--> mlflow.genai.evaluate(data=traces, scorers=[coherence, context, search_quality])
              |
              +--> coherence_judge({{ conversation }})       -- sees all 4 turns
              +--> context_judge({{ conversation }})         -- sees all 4 turns
              +--> search_quality_judge({{ conversation }})  -- sees turns + tool spans
              |
              +--> eval_results.result_df
                    columns: conversation_coherence/value, context_retention/value,
                             search_quality/value, .../justification
```

---

## Comparison: Web Search Bot vs. Casper's Kitchen Bot

| | Web Search Bot | Casper's Kitchen Bot |
|---|---|---|
| Data source | Live web (Tavily/Brave/Bing) | Pre-loaded Casper's endpoints |
| LLM role | Reasoner + decides when to search | Reasoner only; routing is explicit |
| Routing | LLM-driven via tool calling | Intent router → right endpoint |
| Accuracy | Depends on web results | Grounded in authoritative data |
| Freshness | Always current | Only as fresh as the last pipeline run |
| Hallucination risk | Lower (grounded in search results) | Very low (grounded in real tables/PDFs) |
| Portability | Runs anywhere | Requires Databricks + deployed endpoints |
| New judge | `search_quality` (was the search usage appropriate?) | `domain_routing` (was the right endpoint called?) |

---

## Key Dependencies

```
mlflow >= 3.7.0       # {{ conversation }} template + mlflow.genai.evaluate()
openai >= 1.0.0       # tool calling API
requests              # Tavily HTTP calls (or use tavily-python SDK)
```

Optional: `tavily-python` SDK wraps the HTTP call in one line:
```python
from tavily import TavilyClient
client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
result = client.search(query, max_results=3)
```

---

## Implementation Order

1. [ ] `search_tool.py` — test `web_search()` standalone, confirm Tavily results look right
2. [ ] `prompts.py` — system prompt and judge instructions
3. [ ] `scenarios.py` — write test cases including the no-search negative test
4. [ ] `bot_cls.py` — implement the tool-use loop, test a single turn end-to-end
5. [ ] Verify MLflow traces show `web_search` as a child span inside `handle_message`
6. [ ] `bot.py` — CLI entry point
7. [ ] Run all scenarios, review in `mlflow ui`, tune judge instructions if needed

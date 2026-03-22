"""
Prompts for the web-search multi-turn bot.

Each prompt is registered in the MLflow Prompt Registry under a unique name
and always fetched from the registry by that name. The Python source holds the
initial template text; once registered, the registry copy is authoritative and
can be edited via the MLflow UI without changing code.

Prompt names
------------
SYSTEM_PROMPT_NAME           system_prompt
COHERENCE_JUDGE_NAME         judge_conversation_coherence
CONTEXT_RETENTION_JUDGE_NAME judge_context_retention
SEARCH_QUALITY_JUDGE_NAME    judge_search_quality
"""

import mlflow

# ---------------------------------------------------------------------------
# Registry name constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_NAME           = "system_prompt"
COHERENCE_JUDGE_NAME         = "judge_conversation_coherence"
CONTEXT_RETENTION_JUDGE_NAME = "judge_context_retention"
SEARCH_QUALITY_JUDGE_NAME    = "judge_search_quality"

# ---------------------------------------------------------------------------
# Template source strings (used only for initial registration)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that can search the web to answer questions.

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

Bad (loses context):    web_search("restaurants in Seattle")
Good (carries context): web_search("peanut-free Thai restaurants Seattle")
  because the user stated a peanut allergy two turns ago.

Bad (ambiguous reference):    web_search("that restaurant's hours")
Good (resolves the reference): web_search("Piccolo Sogno Chicago hours")
  because the user was asking about Piccolo Sogno earlier in the conversation.

Always resolve pronouns, implicit references, and prior constraints into the query itself.
"""

_COHERENCE_TEMPLATE = """Evaluate the coherence of this multi-turn conversation where the assistant
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

_CONTEXT_RETENTION_TEMPLATE = """Evaluate context retention in this multi-turn conversation.

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

_SEARCH_QUALITY_TEMPLATE = """Evaluate web search usage in this multi-turn conversation.

{{ conversation }}

Assess whether the assistant used web search appropriately:

NECESSARY: Searched at the right times (needed current or specific data) and
           skipped searches when general knowledge sufficed. Good balance.
UNNECESSARY: Over-searched -- triggered searches for things the LLM should
             already know (basic facts, common knowledge), wasting latency.
SKIPPED: Under-searched -- answered with confident specifics (hours, prices,
         ratings, menus) without searching, risking hallucinated facts.

Value: necessary, unnecessary, or skipped.
Rationale: Cite specific turns where search use was appropriate or not.
"""

# ---------------------------------------------------------------------------
# Registration (idempotent)
# ---------------------------------------------------------------------------

def _register_if_missing(name: str, template: str, commit_message: str) -> None:
    """Register a prompt only if it does not already exist in the registry."""
    if mlflow.genai.load_prompt(name, allow_missing=True) is None:
        mlflow.genai.register_prompt(name, template, commit_message=commit_message)


def register_all_prompts() -> None:
    """
    Register all four prompts in the MLflow Prompt Registry.

    Safe to call multiple times — each prompt is only registered on the first
    call. Subsequent calls are no-ops (the registry copy is not overwritten).
    """
    _register_if_missing(
        SYSTEM_PROMPT_NAME,
        _SYSTEM_PROMPT_TEMPLATE,
        "Agent system prompt with search query construction guidance",
    )
    _register_if_missing(
        COHERENCE_JUDGE_NAME,
        _COHERENCE_TEMPLATE,
        "Session-level judge: conversation coherence (bool)",
    )
    _register_if_missing(
        CONTEXT_RETENTION_JUDGE_NAME,
        _CONTEXT_RETENTION_TEMPLATE,
        "Session-level judge: context retention (excellent/good/fair/poor)",
    )
    _register_if_missing(
        SEARCH_QUALITY_JUDGE_NAME,
        _SEARCH_QUALITY_TEMPLATE,
        "Session-level judge: search quality (necessary/unnecessary/skipped)",
    )


# ---------------------------------------------------------------------------
# Public accessors — always fetch from the registry by name
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    return mlflow.genai.load_prompt(SYSTEM_PROMPT_NAME).template


def get_coherence_judge_instructions() -> str:
    return mlflow.genai.load_prompt(COHERENCE_JUDGE_NAME).template


def get_context_retention_judge_instructions() -> str:
    return mlflow.genai.load_prompt(CONTEXT_RETENTION_JUDGE_NAME).template


def get_search_quality_judge_instructions() -> str:
    return mlflow.genai.load_prompt(SEARCH_QUALITY_JUDGE_NAME).template

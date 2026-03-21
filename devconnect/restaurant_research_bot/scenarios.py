"""
Conversation scenarios for the web-search multi-turn bot evaluation.

Design rules:
- At least one turn must require a search and at least one must not.
- Context carryover is tested by having the user state a constraint ONCE and
  NOT repeat it at the turn where a search is triggered -- otherwise trivial.
- Questions must be conversationally related across all turns so the context
  retention judge has something meaningful to evaluate.
- expected_search_quality "necessary" means the bot used search appropriately
  (searched when needed, skipped when not). It does NOT mean a search was needed.
"""

from collections import defaultdict
from typing import Any, Dict, List

_session_counters: Dict[str, int] = defaultdict(int)


def _next_session_id(scenario: str) -> str:
    _session_counters[scenario] += 1
    return f"session-{scenario}-{_session_counters[scenario]:03d}"


def get_scenario_restaurant_research() -> Dict[str, Any]:
    """
    Multi-turn restaurant research session.
    Turns 1-3 need searches; turn 4 should NOT search (synthesis from prior turns).
    """
    return {
        "name": "Restaurant Research",
        "session_id": _next_session_id("restaurant"),
        "expected_coherence": True,
        "expected_retention": "excellent",
        "expected_search_quality": "necessary",
        "messages": [
            "What are some highly rated Italian restaurants in Chicago's River North neighborhood?",
            "I'm vegetarian -- which of those have good vegetarian options?",
            "What are the current hours for Piccolo Sogno on West Grand?",
            "Given everything we've discussed, which one would you recommend for tonight?",
            # Turn 4: bot should synthesize from prior turns, NOT search again.
        ],
    }


def get_scenario_food_safety() -> Dict[str, Any]:
    """
    User researching food safety scores before choosing a restaurant.
    Turns 1-3 need searches. Turn 4 is synthesis -- no search.
    Tests whether the bot resolves "that restaurant" using conversation history.
    """
    return {
        "name": "Food Safety Research",
        "session_id": _next_session_id("safety"),
        "expected_coherence": True,
        "expected_retention": "good",
        "expected_search_quality": "necessary",
        "messages": [
            "How can I check food safety inspection scores for restaurants in San Francisco?",
            "What's the current health inspection rating for Nopa on Divisadero Street?",
            "Are there any recent violation reports for that restaurant?",
            # Turn 3: "that restaurant" = Nopa. LLM must resolve from history.
            # Correct search: "Nopa restaurant San Francisco health violations"
            # Wrong search:   "that restaurant health violations"
            "Based on all this, would you say it's safe to eat there?",
            # Turn 4: synthesize from prior searches, no search needed.
        ],
    }


def get_scenario_nutrition_and_allergens() -> Dict[str, Any]:
    """
    Silent context carryover test: peanut allergy stated ONCE in turn 1, never repeated.
    Turns 2-3 are conversational (no search needed).
    Turn 4 requires a search AND the bot must silently inject the constraint.

    This is the hardest scenario for the stateless search problem: the user doesn't
    say "peanut-free" in turn 4, so the LLM must pull that constraint from history
    and bake it into the search query itself.

    Correct turn 4 query: "Thai restaurants Seattle peanut-free"
    Wrong turn 4 query:   "Thai restaurants Seattle"  <-- drops the constraint
    """
    return {
        "name": "Silent Allergen Carryover",
        "session_id": _next_session_id("allergen"),
        "expected_coherence": True,
        "expected_retention": "excellent",
        "expected_search_quality": "necessary",
        "messages": [
            "I have a severe peanut allergy. What Thai dishes should I completely avoid?",
            "Good to know. What does Pad See Ew typically taste like?",
            # Turn 2: general knowledge, no search needed.
            "Is it a common dish to find at Thai restaurants?",
            # Turn 3: general knowledge, no search needed.
            "Can you find me a Thai restaurant in Seattle?",
            # Turn 4: user does NOT say "peanut-free" -- bot must carry constraint silently.
            # Correct search: "Thai restaurants Seattle peanut-free menu"
            # Wrong search:   "Thai restaurants Seattle"  (drops the allergy)
        ],
    }


def get_scenario_no_search_needed() -> Dict[str, Any]:
    """
    Negative test: a coherent conversation that stays within LLM general knowledge.
    All questions build on the same topic (Italian cooking), none require web search.

    expected_search_quality "necessary" means: the bot correctly made zero searches.
    Context retention IS testable: the vegetarian preference from turn 2 should
    carry through to the recipe suggestion in turn 4.
    """
    return {
        "name": "No-Search Needed (General Knowledge)",
        "session_id": _next_session_id("nosearch"),
        "expected_coherence": True,
        "expected_retention": "good",
        "expected_search_quality": "necessary",  # correct behavior = zero searches
        "messages": [
            "What are the main pasta shapes used in Italian cooking and what sauces go with them?",
            "I'm vegetarian -- which of those pasta dishes are easy to make meat-free?",
            "What's the difference between pecorino and parmesan in terms of flavor?",
            "Can you suggest a simple vegetarian pasta recipe using what we've discussed?",
            # Turn 4: bot should use the vegetarian constraint from turn 2
            # and the pasta/cheese knowledge from turns 1-3 -- no search needed.
        ],
    }


def get_all_scenarios() -> List[Dict[str, Any]]:
    return [
        get_scenario_restaurant_research(),
        get_scenario_food_safety(),
        get_scenario_nutrition_and_allergens(),
        get_scenario_no_search_needed(),
    ]


def get_scenario_by_name(name: str) -> Dict[str, Any]:
    lookup = {s["name"]: s for s in get_all_scenarios()}
    # Also support short keys matching the start of the name
    short_lookup = {
        "restaurant": get_scenario_restaurant_research(),
        "safety":     get_scenario_food_safety(),
        "allergen":   get_scenario_nutrition_and_allergens(),
        "nosearch":   get_scenario_no_search_needed(),
    }
    if name in lookup:
        return lookup[name]
    if name in short_lookup:
        return short_lookup[name]
    raise ValueError(
        f"Unknown scenario: '{name}'. "
        f"Available: {list(short_lookup.keys())}"
    )

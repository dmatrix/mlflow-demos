"""
Multi-turn conversational bot with web search.

Demonstrates MLflow session-level evaluation for a tool-augmented agent:
- Web search via Tavily (stateless API)
- Session tracking with mlflow.update_current_trace()
- Session-level judges using {{ conversation }} template
- mlflow.genai.evaluate() with coherence, context retention, and search quality judges
"""

from devconnect.restaurant_research_bot.restaurant_research_agent_cls import RestaurantResearchAgent
from devconnect.restaurant_research_bot.scenarios import (
    get_scenario_restaurant_research,
    get_scenario_food_safety,
    get_scenario_nutrition_and_allergens,
    get_scenario_no_search_needed,
    get_all_scenarios,
    get_scenario_by_name,
)

__all__ = [
    "RestaurantResearchAgent",
    "get_scenario_restaurant_research",
    "get_scenario_food_safety",
    "get_scenario_nutrition_and_allergens",
    "get_scenario_no_search_needed",
    "get_all_scenarios",
    "get_scenario_by_name",
]

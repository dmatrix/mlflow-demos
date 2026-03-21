"""
CLI entry point for the web-search multi-turn bot evaluation.

Usage:
  # OpenAI (default)
  export OPENAI_API_KEY=sk-...
  export TAVILY_API_KEY=tvly-...
  uv run mlflow-restaurant-research-bot

  # Run a specific scenario
  uv run mlflow-restaurant-research-bot --scenario allergen

  # Databricks
  export DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
  export DATABRICKS_TOKEN=dapi...
  export TAVILY_API_KEY=tvly-...
  uv run mlflow-restaurant-research-bot --provider databricks --model databricks-gpt-4o-mini

  # View results
  mlflow ui
"""

import argparse
import mlflow
from dotenv import load_dotenv

from devconnect.config import AgentConfig
from devconnect.mlflow_config import setup_mlflow_tracking
from devconnect.restaurant_research_bot.restaurant_research_agent_cls import RestaurantResearchAgent as WebSearchAgent
from devconnect.restaurant_research_bot.scenarios import get_all_scenarios, get_scenario_by_name

EXPERIMENT_NAME = "restaurant-research-bot"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Multi-turn web-search bot with MLflow session evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "databricks"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (default: gpt-4o-mini for openai, databricks-gpt-4o-mini for databricks)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for session-level judges (default: same as --model)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run a single scenario by short name: restaurant, safety, allergen, nosearch",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print evaluation DataFrame columns and extra details",
    )
    args = parser.parse_args()

    # Model defaults
    if args.model is None:
        args.model = (
            "gpt-5-mini" if args.provider == "openai" else "databricks-gpt-5-mini"
        )
    if args.judge_model is None:
        args.judge_model = (
            "gpt-5-mini"
            if args.provider == "openai"
            else "databricks-gemini-2-5-flash"
        )

    # MLflow make_judge() requires "<provider>:/<model>" format for all providers.
    judge_model_uri = f"openai:/{args.judge_model}"

    print("\n" + "=" * 60)
    print("Multi-Turn Web-Search Bot  |  MLflow Session Evaluation")
    print("=" * 60)
    print(f"\n  Provider:    {args.provider}")
    print(f"  Model:       {args.model}")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")

    # MLflow setup
    setup_mlflow_tracking(experiment_name=EXPERIMENT_NAME, enable_autolog=True)

    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        mlflow_experiment=EXPERIMENT_NAME,
    )

    print("\n  Initialising judges...")
    agent = WebSearchAgent(config=config, judge_model=judge_model_uri, debug=args.debug)

    scenarios = (
        [get_scenario_by_name(args.scenario)]
        if args.scenario
        else get_all_scenarios()
    )

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario['name']}")
        print(f"Session:  {scenario['session_id']}")
        print("=" * 60)

        # Each scenario runs inside its own MLflow run.
        # All handle_message() traces produced inside the with-block share
        # this run_id, which evaluate_session() uses to find them.
        with mlflow.start_run(run_name=scenario["name"]) as run:
            agent.run_conversation(scenario["messages"], scenario["session_id"])
            try:
                results = agent.evaluate_session(
                    scenario["session_id"], run.info.run_id
                )
            except Exception as exc:
                print(f"\n  Evaluation failed: {exc}")
                print("  Ensure MLflow >= 3.8.0 and TAVILY_API_KEY is set.\n")
                continue

        coh  = results["coherence"]
        ctx  = results["context_retention"]
        srch = results["search_quality"]

        print(f"\n{'─' * 40}")
        print(f"  Coherence:         {'PASS' if coh['passed'] else 'FAIL'}  ({coh['feedback_value']})")
        if coh["rationale"]:
            print(f"    {coh['rationale']}")
        print(f"  Context Retention: {str(ctx['feedback_value']).upper()}")
        if ctx["rationale"]:
            print(f"    {ctx['rationale']}")
        print(f"  Search Quality:    {str(srch['feedback_value']).upper()}")
        if srch["rationale"]:
            print(f"    {srch['rationale']}")

    print(f"\n{'=' * 60}")
    print(f"Done. View traces: mlflow ui  →  '{EXPERIMENT_NAME}' experiment")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

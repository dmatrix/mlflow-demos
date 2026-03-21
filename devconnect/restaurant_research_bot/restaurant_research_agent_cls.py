"""
WebSearchAgent: multi-turn conversational bot backed by web search.

Key design:
- Tool-use loop: LLM calls web_search() when it needs current data; the agent
  executes the call, feeds results back, and loops until no more tool calls.
- Session history: accumulated per session_id so the LLM sees full context on
  every turn, enabling it to construct contextual search queries.
- MLflow tracing: each handle_message() call is a CHAT_MODEL trace tagged with
  the session ID; web_search() calls appear as child TOOL spans inside it.
- Evaluation: mlflow.genai.evaluate() with three session-level judges that each
  use the {{ conversation }} template to see all turns as one conversation.
"""

import json
import os
import litellm
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from typing import Dict, List, Literal

from devconnect.config import AgentConfig
from devconnect.providers import get_client
from devconnect.restaurant_research_bot.search_tool import web_search, WEB_SEARCH_TOOL_SCHEMA
from devconnect.restaurant_research_bot.prompts import (
    get_system_prompt,
    get_coherence_judge_instructions,
    get_context_retention_judge_instructions,
    get_search_quality_judge_instructions,
)


class RestaurantResearchAgent:
    """
    Multi-turn conversational bot with web search and MLflow session evaluation.

    Usage:
        config = AgentConfig(model="gpt-4o-mini", provider="openai")
        agent = WebSearchAgent(config, judge_model="gpt-5-mini")

        with mlflow.start_run(run_name="my-scenario") as run:
            agent.run_conversation(messages, session_id="session-001")
            results = agent.evaluate_session("session-001", run.info.run_id)
    """

    def __init__(
        self,
        config: AgentConfig,
        judge_model: str,
        debug: bool = False,
    ):
        """
        Args:
            config: AgentConfig specifying provider, model, experiment name, etc.
            judge_model: Model URI for session-level judges.
                         For OpenAI: "gpt-4o-mini"
                         For Databricks: "openai:/databricks-gemini-2-5-flash"
            debug: Print DataFrame columns and extra evaluation details.
        """
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config
        self.judge_model = judge_model
        self.debug = debug
        self.session_histories: Dict[str, List[dict]] = {}
        self._init_judges()

    def _init_judges(self) -> None:
        """
        Initialise three session-level judges.

        All three use the {{ conversation }} template, which makes MLflow
        automatically aggregate all traces for the session and pass the full
        conversation to each judge rather than evaluating turn by turn.

        litellm (used internally by MLflow judges) has its own credential
        initialisation separate from the OpenAI SDK. Explicitly set the API key
        so litellm can reach the OpenAI endpoint.
        """
        litellm.api_key = os.environ.get("OPENAI_API_KEY", "")
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
        self.search_quality_judge = make_judge(
            name="search_quality",
            model=self.judge_model,
            instructions=get_search_quality_judge_instructions(),
            feedback_value_type=Literal["necessary", "unnecessary", "skipped"],
        )

        print(f"  Coherence judge session-level:      {self.coherence_judge.is_session_level_scorer}")
        print(f"  Context retention judge session-level: {self.context_judge.is_session_level_scorer}")
        print(f"  Search quality judge session-level: {self.search_quality_judge.is_session_level_scorer}")

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="handle_message")
    def handle_message(self, message: str, session_id: str) -> str:
        """
        Process one conversation turn with MLflow tracing and tool-use loop.

        The @mlflow.trace decorator opens a CHAT_MODEL span. The
        mlflow.update_current_trace() call tags it with the session ID so
        MLflow can group all turns for evaluate_session(). web_search() calls
        appear as child TOOL spans inside this span.

        Args:
            message: The user's message for this turn.
            session_id: Stable identifier for this conversation session.

        Returns:
            The assistant's reply as a plain string.
        """
        # Tag this trace so MLflow can group it with other turns in the session.
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id}
        )

        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        history = self.session_histories[session_id]

        # system + full history + new user message
        messages = [{"role": "system", "content": get_system_prompt()}]
        messages += history
        messages.append({"role": "user", "content": message})

        # Tool-use loop: iterate until the LLM stops requesting tool calls.
        while True:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=[WEB_SEARCH_TOOL_SCHEMA],
                tool_choice="auto",
                max_completion_tokens=2048,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                # Append the assistant message that contains the tool_calls field
                # before adding tool results -- required by the OpenAI API.
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = web_search(args["query"])  # child TOOL span

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                # Loop: LLM now has the search results and will form a response.

            else:
                reply = choice.message.content
                break

        # Persist only the user/assistant exchange in session history,
        # not the intermediate tool messages (those are turn-local scaffolding).
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        return reply

    def run_conversation(self, messages: List[str], session_id: str) -> List[str]:
        """
        Run a full multi-turn conversation, printing each turn.

        Args:
            messages: Ordered list of user messages (one per turn).
            session_id: Session identifier shared across all turns.

        Returns:
            List of assistant replies in turn order.
        """
        replies = []
        for i, msg in enumerate(messages):
            print(f"\nTurn {i + 1}/{len(messages)}")
            print(f"  User: {msg}")
            reply = self.handle_message(msg, session_id)
            print(f"  Bot:  {reply}")
            replies.append(reply)
        return replies

    def evaluate_session(self, session_id: str, run_id: str) -> Dict:
        """
        Evaluate the full conversation using mlflow.genai.evaluate().

        Searches for all traces produced inside the given MLflow run, then fans
        them out to the three session-level judges via mlflow.genai.evaluate().
        Each judge receives the complete conversation via {{ conversation }}.

        Args:
            session_id: Session identifier (used for display only here).
            run_id: MLflow run ID from mlflow.start_run(). All handle_message()
                    traces created inside that run share this run_id.

        Returns:
            Dictionary with coherence, context_retention, and search_quality results.
        """
        experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment)
        if experiment is None:
            raise ValueError(
                f"MLflow experiment '{self.config.mlflow_experiment}' not found. "
                "Call mlflow.set_experiment() before running conversations."
            )

        session_traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            filter_string=f"run_id = '{run_id}'",
        )

        if len(session_traces) == 0:
            raise ValueError(
                f"No traces found for run_id '{run_id}'. "
                "Ensure handle_message() was called inside mlflow.start_run()."
            )

        print(f"\nEvaluating session '{session_id}' ({len(session_traces)} traces)...")

        try:
            from IPython.utils.capture import capture_output as _cap
            with _cap():
                eval_results = mlflow.genai.evaluate(
                    data=session_traces,
                    scorers=[
                        self.coherence_judge,
                        self.context_judge,
                        self.search_quality_judge,
                    ],
                )
        except ImportError:
            eval_results = mlflow.genai.evaluate(
                data=session_traces,
                scorers=[
                    self.coherence_judge,
                    self.context_judge,
                    self.search_quality_judge,
                ],
            )

        result_df = eval_results.result_df

        def extract(keyword: str, suffix: str):
            """Find first column matching keyword + suffix substrings."""
            cols = [
                c for c in result_df.columns
                if keyword in c.lower() and suffix in c.lower()
            ]
            if not cols:
                return None
            series = result_df[cols[0]].dropna()
            return series.iloc[0] if len(series) > 0 else None

        # Column names follow the pattern "<judge_name>/value" and
        # "<judge_name>/reason" (per CLAUDE.md). Fall back to "/justification"
        # for forward-compatibility with older MLflow versions.
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
            "search_quality": {
                "feedback_value": extract("search", "/value"),
                "rationale":      extract_reason("search"),
            },
        }

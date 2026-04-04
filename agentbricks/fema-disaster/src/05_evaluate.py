# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install databricks-vectorsearch databricks-agents mlflow==3.10.0 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration from bundle variables
import requests
import pandas as pd
import mlflow

# NOTE: Replace with your actual supervisor endpoint name
SUPERVISOR_ENDPOINT = "<YOUR SUPERVISOR ENDPOINT NAME>"

WORKSPACE_HOST = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

assert SUPERVISOR_ENDPOINT, (
    "supervisor_endpoint is not set. Create the Supervisor Agent in the UI first, "
    "then set the variable: databricks bundle deploy -t dev --var supervisor_endpoint=<endpoint-name>"
)

print(f"Supervisor endpoint: {SUPERVISOR_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Build evaluation dataset
eval_data = pd.DataFrame([
    {
        "request": "How many disasters hit California in 2024?",
        "expected_facts": ["Should include a count or number of disasters", "Should mention California", "Should reference 2024"],
    },
    {
        "request": "What are FEMA's wildfire safety guidelines?",
        "expected_facts": ["Should mention defensible space", "Should reference Zone 1 and Zone 2", "Should mention go-bag or 72 hours"],
    },
    {
        "request": "What was the total federal aid for hurricanes in 2024?",
        "expected_facts": ["Should include a dollar amount", "Should reference hurricanes", "Should reference 2024"],
    },
    {
        "request": "What is the disaster declaration process?",
        "expected_facts": ["Should mention Preliminary Damage Assessment", "Should distinguish Emergency vs Major Disaster Declaration"],
    },
    {
        "request": "How many tornado events occurred in 2024 and what tornado safety procedures does FEMA recommend?",
        "expected_facts": ["Should include tornado count", "Should mention shelter in lowest interior room", "Should reference Enhanced Fujita Scale"],
    },
    {
        "request": "Which state had the highest severity earthquake and what is the earthquake response protocol?",
        "expected_facts": ["Should name a state", "Should mention Drop Cover Hold On", "Should reference ATC-20 tagging"],
    },
])

print(f"Evaluation dataset: {len(eval_data)} queries")
print(f"Target endpoint: endpoints:/{SUPERVISOR_ENDPOINT}")
display(eval_data)

# COMMAND ----------

# DBTITLE 1,Run MLflow GenAI evaluation
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines, Correctness


@mlflow.trace(span_type="AGENT")
def predict_supervisor(request: str) -> str:
    """Query the supervisor and return the final synthesized answer."""
    resp = requests.post(
        f"https://{WORKSPACE_HOST}/serving-endpoints/{SUPERVISOR_ENDPOINT}/invocations",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"input": [{"role": "user", "content": request}]},
    )
    resp.raise_for_status()
    result = resp.json()

    if "output" in result and isinstance(result["output"], list):
        for msg in reversed(result["output"]):
            if msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                if isinstance(content, list):
                    texts = [c["text"] for c in content if c.get("type") == "output_text" and c.get("text")]
                    if texts:
                        return "\n".join(texts)
                elif isinstance(content, str) and content:
                    return content
    return str(result)


# Build evaluation dataset with expectations for the Correctness scorer
genai_eval_data = [
    {
        "inputs": {"request": row["request"]},
        "expectations": {"expected_response": "; ".join(row["expected_facts"])},
    }
    for _, row in eval_data.iterrows()
]

scorers = [
    RelevanceToQuery(),
    Safety(),
    Correctness(),
    Guidelines(
        name="disaster_response_quality",
        guidelines=(
            "Responses about data should include specific numbers or statistics. "
            "Responses about policies should reference specific protocol names. "
            "Combined responses should integrate data findings with policy guidance. "
            "All responses should be actionable and avoid vague generalities."
        ),
    ),
]

with mlflow.start_run(run_name="AgentBricks-Supervisor-Eval-v3"):
    eval_results = mlflow.genai.evaluate(
        data=genai_eval_data,
        predict_fn=predict_supervisor,
        scorers=scorers,
    )

print("Evaluation complete!")
print(f"\nMetrics:")
for metric, value in eval_results.metrics.items():
    print(f"  {metric}: {value}")

print(f"\nView detailed results in the MLflow experiment UI.")
display(eval_results.tables["eval_results"])

# COMMAND ----------

# DBTITLE 1,Individual judge assessments
from databricks.agents.evals import judges

print("Running individual judge assessments...\n")
print("=" * 70)

# Re-run queries to get fresh responses for judge evaluation
for i, row in eval_data.iterrows():
    query = row["request"]
    print(f"\nQuery {i+1}: {query[:60]}...")

    try:
        response = predict_supervisor(query)

        # Relevance
        rel = judges.relevance_to_query(request=query, response=response)
        print(f"  Relevance:  {rel.value} - {rel.rationale[:100]}")

        # Safety
        safe = judges.safety(request=query, response=response)
        print(f"  Safety:     {safe.value} - {safe.rationale[:100]}")

        # Guideline adherence
        guidelines = {
            "disaster_response": [
                "Responses about data should include specific numbers or statistics",
                "Responses about policies should reference specific protocol names",
                "Combined responses should integrate data findings with policy guidance",
                "All responses should be actionable and avoid vague generalities",
            ]
        }
        ga = judges.guideline_adherence(
            request=query, response=response, guidelines=guidelines
        )
        print(f"  Guidelines: {ga[0].value} - {ga[0].rationale[:100]}")
    except Exception as e:
        print(f"  Skipped (error: {e})")

print("\n" + "=" * 70)
print("Individual judge assessments complete.")

# Databricks notebook source
# DBTITLE 1,Configuration from bundle variables
import requests
import pandas as pd


# NOTE: Replace with your actual supervisor endpoint name
SUPERVISOR_ENDPOINT = "<YOUR SUPERVISOR ENDPOINT NAME>"
WORKSPACE_HOST = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

assert SUPERVISOR_ENDPOINT, (
    "supervisor_endpoint is not set. Create the Supervisor Agent in the UI first, "
    "then set the variable: databricks bundle deploy -t dev --var supervisor_endpoint=<endpoint-name>"
)

print(f"Supervisor endpoint: {SUPERVISOR_ENDPOINT}")
print(f"Workspace:           {WORKSPACE_HOST}")

# COMMAND ----------

# DBTITLE 1,Define query function and test data
def query_supervisor(query: str) -> str:
    """Query the AgentBricks Supervisor endpoint and return the final answer."""
    response = requests.post(
        f"https://{WORKSPACE_HOST}/serving-endpoints/{SUPERVISOR_ENDPOINT}/invocations",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"input": [{"role": "user", "content": query}]},
    )
    response.raise_for_status()
    result = response.json()

    # Extract the text from the LAST assistant message in the output trace.
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
    # Fallback for other response shapes
    if "choices" in result and result["choices"]:
        return result["choices"][0]["message"]["content"]
    return str(result)


# Same test queries as the original tutorial
test_queries = [
    # Genie route (structured data)
    "How many disasters hit California in 2024?",
    "What was the total federal aid for hurricane-related disasters in 2024?",
    # Knowledge Assistant route (documents/policy)
    "What are FEMA's evacuation protocols for wildfire zones?",
    "Who is eligible for federal disaster assistance and what types of aid are available?",
    # Both routes (data + policy)
    "How many flood disasters occurred in 2024 and what are FEMA's flood response procedures?",
    "Which states had severity-5 disasters in 2024, and what safety protocols apply to those disaster types?",
]

print(f"Running {len(test_queries)} test queries against the Supervisor endpoint...")
print(f"Endpoint: {SUPERVISOR_ENDPOINT}")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Run all test queries
results_log = []

for i, query in enumerate(test_queries):
    print(f"\nQuery {i+1}: {query}")
    print("-" * 70)

    try:
        response = query_supervisor(query)
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                response = response["choices"][0]["message"]["content"]
            elif "result" in response:
                response = response["result"]
            else:
                response = str(response)
        print(f"Response: {response[:400]}..." if isinstance(response, str) and len(response) > 400 else f"Response: {response}")

        results_log.append({
            "query": query,
            "response": response,
            "status": "success",
        })
    except Exception as e:
        print(f"Error: {e}")
        results_log.append({
            "query": query,
            "response": str(e),
            "status": "error",
        })

    print("=" * 80)

print(f"\nCompleted {len(results_log)} queries.")
print(f"  Successes: {sum(1 for r in results_log if r['status'] == 'success')}")
print(f"  Errors:    {sum(1 for r in results_log if r['status'] == 'error')}")

# COMMAND ----------

# DBTITLE 1,Results summary
results_df = pd.DataFrame([
    {
        "Query": r["query"][:60] + "...",
        "Expected Route": (
            "Genie" if i < 2 else "Knowledge Assistant" if i < 4 else "Both"
        ),
        "Status": r["status"],
        "Response Preview": r["response"][:150] + "..." if len(r["response"]) > 150 else r["response"],
    }
    for i, r in enumerate(results_log)
])

print("\nRouting Summary")
print("=" * 60)
for i, r in enumerate(results_log):
    route_icon = "[DATA]" if i < 2 else "[DOCS]" if i < 4 else "[BOTH]"
    print(f"  {route_icon} {r['query'][:60]}...")

display(spark.createDataFrame(results_df))

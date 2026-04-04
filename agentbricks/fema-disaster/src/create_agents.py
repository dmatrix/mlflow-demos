# Databricks notebook source
"""
Create Genie Space, Vector Search index, and Knowledge Assistant for the
FEMA multi-agent demo.

Runs as a Databricks job task within the DAB bundle. Uses the Databricks SDK
(pre-installed on serverless) with automatic workspace authentication.

Imports core logic from setup_agents.py (deployed alongside this notebook).
"""

# COMMAND ----------

dbutils.widgets.text("catalog", "jules_catalog", "Catalog")
dbutils.widgets.text("schema", "fema_demo", "Schema")
dbutils.widgets.text("table_name", "disaster_data", "Disaster Table")
dbutils.widgets.text("policy_table_name", "policy_docs", "Policy Table")
dbutils.widgets.text("vs_endpoint_name", "fema_vs_endpoint", "VS Endpoint")
dbutils.widgets.text("embedding_model", "databricks-gte-large-en", "Embedding Model")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")
policy_table_name = dbutils.widgets.get("policy_table_name")
vs_endpoint_name = dbutils.widgets.get("vs_endpoint_name")
embedding_model = dbutils.widgets.get("embedding_model")
warehouse_id = dbutils.widgets.get("warehouse_id")

if not warehouse_id:
    raise ValueError("warehouse_id parameter is required for Genie Space creation")

fq_table = f"{catalog}.{schema}.{table_name}"
fq_policy_table = f"{catalog}.{schema}.{policy_table_name}"
vs_index_name = f"{catalog}.{schema}.policy_docs_index"

print(f"Catalog:         {catalog}")
print(f"Schema:          {schema}")
print(f"Disaster table:  {fq_table}")
print(f"Policy table:    {fq_policy_table}")
print(f"VS endpoint:     {vs_endpoint_name}")
print(f"VS index:        {vs_index_name}")
print(f"Embedding model: {embedding_model}")

# COMMAND ----------

import importlib
import sys

def _resolve_scripts_dir():
    nb_path = (
        dbutils.notebook.entry_point
        .getDbutils().notebook().getContext()
        .notebookPath().get()
    )
    if not nb_path.startswith("/Workspace"):
        nb_path = "/Workspace" + nb_path
    return "/".join(nb_path.rstrip("/").split("/")[:-1])

scripts_dir = _resolve_scripts_dir()
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import setup_agents
importlib.reload(setup_agents)

print(f"Loaded setup_agents from: {scripts_dir}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
user_name = w.current_user.me().user_name

print("=" * 60)
print("CREATING AGENTS FOR FEMA MULTI-AGENT DEMO")
print("=" * 60)

# COMMAND ----------

# --- Create Genie Space ---

genie_id = setup_agents.create_genie_space(w, fq_table, warehouse_id)

# COMMAND ----------

# --- Create Vector Search endpoint + Delta Sync index ---

setup_agents.create_vector_search(
    vs_endpoint_name, vs_index_name, fq_policy_table, embedding_model
)

# COMMAND ----------

# --- Create Knowledge Assistant ---

ka_name = setup_agents.create_knowledge_assistant(w, vs_index_name)

# COMMAND ----------

# --- Print Supervisor Agent instructions (manual step) ---

setup_agents.print_supervisor_instructions("FEMA Disaster Data", "FEMA Policy Assistant")

# COMMAND ----------

print(f"\n{'='*60}")
print("AGENT SETUP COMPLETE")
print(f"{'='*60}")
print(f"\nGenie Space: {genie_id}")
print(f"Knowledge Assistant: {ka_name}")
print("\nNext: Create Supervisor Agent manually (see instructions above)")

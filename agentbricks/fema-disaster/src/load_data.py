# Databricks notebook source
"""
Load FEMA disaster data and policy documents into Unity Catalog Delta tables.

Runs as a Databricks job task within the DAB bundle. Imports core logic from
load_fema_data.py (deployed alongside this notebook).
"""

# COMMAND ----------

dbutils.widgets.text("catalog", "jules_catalog", "Catalog")
dbutils.widgets.text("schema", "fema_demo", "Schema")
dbutils.widgets.text("table_name", "disaster_data", "Disaster Table")
dbutils.widgets.text("policy_table_name", "policy_docs", "Policy Table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")
policy_table_name = dbutils.widgets.get("policy_table_name")

print(f"Target: {catalog}.{schema}")
print(f"Disaster table: {table_name}")
print(f"Policy table:   {policy_table_name}")

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

import load_fema_data
importlib.reload(load_fema_data)

print(f"Loaded load_fema_data from: {scripts_dir}")

# COMMAND ----------

# --- Create catalog and schema ---
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"Catalog and schema ready: {catalog}.{schema}")

# COMMAND ----------

# --- Load FEMA disaster data ---
fq_table = load_fema_data.load_fema_disaster_data(
    spark=spark,
    catalog=catalog,
    schema=schema,
    table_name=table_name,
)
print(f"\nDisaster table created: {fq_table}")
count = spark.sql(f"SELECT count(*) as n FROM {fq_table}").first().n
print(f"  {count} rows")

# COMMAND ----------

# --- Write policy documents to Delta table ---
import pandas as pd

POLICY_DOCUMENTS = [
    {
        "doc_id": "evacuation_protocols",
        "doc_uri": "https://fema.gov/ics-300",
        "text": (
            "FEMA Evacuation Protocols (ICS-300): All evacuation orders must follow the Incident Command System. "
            "Zone-based evacuation proceeds from highest-risk zones outward. Mandatory evacuation requires "
            "governor authorization. Evacuation routes must be pre-designated and communicated via Wireless "
            "Emergency Alerts (WEA). Special needs populations require dedicated transport. Shelter capacity "
            "must be verified before issuing orders. Pet-friendly shelters must be available per the PETS Act."
        ),
    },
    {
        "doc_id": "wildfire_safety",
        "doc_uri": "https://fema.gov/wildfire-safety",
        "text": (
            "FEMA Wildfire Safety Guidelines: Create defensible space of 100 feet around structures. "
            "Zone 1 (0-30 ft): remove all dead vegetation and flammable materials. Zone 2 (30-100 ft): "
            "reduce and space vegetation. Have a go-bag ready with essentials for 72 hours. Know two "
            "evacuation routes from your area. Close all windows and doors before evacuating. Wear "
            "protective clothing: long sleeves, cotton/wool, N95 mask. Monitor air quality index (AQI) daily."
        ),
    },
    {
        "doc_id": "disaster_declaration",
        "doc_uri": "https://fema.gov/disaster-declaration",
        "text": (
            "FEMA Disaster Declaration Process: Local government requests state assistance. If overwhelmed, "
            "governor requests federal declaration from the President. FEMA conducts Preliminary Damage "
            "Assessment (PDA). Two declaration types: Emergency Declaration (Category B assistance, $5M cap) "
            "and Major Disaster Declaration (all assistance categories, no cap). Individual Assistance (IA) "
            "and Public Assistance (PA) can be authorized separately. Timeline: PDA within 10 days, "
            "presidential decision within 5 days of governor's request."
        ),
    },
    {
        "doc_id": "aid_eligibility",
        "doc_uri": "https://fema.gov/aid-eligibility",
        "text": (
            "FEMA Individual Assistance Eligibility: Applicants must be U.S. citizens, non-citizen nationals, "
            "or qualified aliens. Primary residence must be in the declared disaster area. Applicants must "
            "register within 60 days of disaster declaration. Types of assistance: Housing Assistance "
            "(temporary rental, home repair, replacement), Other Needs Assistance (medical, dental, funeral, "
            "personal property, transportation). Maximum grant amount is adjusted annually (currently ~$42,500). "
            "SBA disaster loans available for amounts exceeding FEMA maximums."
        ),
    },
    {
        "doc_id": "flood_response",
        "doc_uri": "https://fema.gov/flood-response",
        "text": (
            "FEMA Flood Response Procedures: National Flood Insurance Program (NFIP) covers up to $250,000 "
            "for residential structures and $100,000 for contents. Flash flood warnings require immediate "
            "evacuation to higher ground. Do not walk/drive through floodwaters. 6 inches of moving water "
            "can knock down an adult. Turn Around, Don't Drown campaign. After flooding: document all "
            "damage photographically before cleanup. Boil water advisories typically issued. Mold "
            "remediation must begin within 24-48 hours."
        ),
    },
    {
        "doc_id": "earthquake_response",
        "doc_uri": "https://fema.gov/earthquake-response",
        "text": (
            "FEMA Earthquake Response Protocol: Drop, Cover, and Hold On during shaking. After shaking stops: "
            "check for injuries, check for structural damage, be prepared for aftershocks. Do not re-enter "
            "damaged buildings. Check gas lines and water mains. Earthquake early warning systems (ShakeAlert) "
            "provide seconds to minutes of advance notice. Building codes (IBC seismic provisions) require "
            "structures to resist collapse. Critical infrastructure (hospitals, fire stations) must meet "
            "higher seismic standards. Post-earthquake safety inspections use ATC-20 tagging system: "
            "Green (inspected), Yellow (restricted use), Red (unsafe)."
        ),
    },
    {
        "doc_id": "tornado_safety",
        "doc_uri": "https://fema.gov/tornado-safety",
        "text": (
            "FEMA Tornado Safety Guidelines: Seek shelter in lowest interior room away from windows. "
            "Mobile homes are NOT safe during tornadoes -- evacuate to sturdy buildings. Tornado Watch: "
            "conditions favorable, be prepared. Tornado Warning: tornado detected, take shelter immediately. "
            "Enhanced Fujita Scale (EF0-EF5) rates tornado intensity. Storm shelters must meet FEMA P-361 "
            "standards. Safe rooms can be built in existing homes (FEMA P-320). Community shelters require "
            "minimum 50 square feet per person. Monitor NOAA Weather Radio for alerts."
        ),
    },
]

fq_policy_table = f"{catalog}.{schema}.{policy_table_name}"
policy_df = spark.createDataFrame(pd.DataFrame(POLICY_DOCUMENTS))
policy_df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(fq_policy_table)

spark.sql(f"""COMMENT ON TABLE {fq_policy_table} IS
  'FEMA policy documents for Knowledge Assistant retrieval. Contains evacuation protocols, safety guidelines, declaration processes, aid eligibility, and disaster-specific response procedures.'""")

print(f"\nPolicy table created: {fq_policy_table}")
print(f"  {len(POLICY_DOCUMENTS)} documents written")
print(f"  Change Data Feed enabled (required for Delta Sync Index)")

# COMMAND ----------

print(f"\n{'='*60}")
print("DEMO DATA LOADED SUCCESSFULLY")
print(f"{'='*60}")
print(f"\nTables in {catalog}.{schema}:")
print(f"  {table_name}: {count} rows")
print(f"  {policy_table_name}: {len(POLICY_DOCUMENTS)} rows")
print(f"\nNext: run create_agents to create Genie Space and Knowledge Assistant")

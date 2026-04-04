"""
FEMA Multi-Agent Demo - Agent Setup Library

Creates a Genie Space, Vector Search index, and Knowledge Assistant using
the Databricks SDK. The Supervisor Agent must still be created manually via
the UI (no SDK support yet).

This module is imported by create_agents.py (the thin notebook wrapper).

Usage (CLI):
    python setup_agents.py --catalog jules_catalog --schema fema_demo \
        --vs-endpoint fema_vs_endpoint --warehouse-id <id>
"""

import json
import sys
import time
import uuid

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.knowledgeassistants import (
        FilesSpec,
        KnowledgeAssistant,
        KnowledgeSource,
        IndexSpec,
    )
except ImportError as _import_err:
    if __name__ == "__main__":
        print(
            "Error: databricks-sdk not installed or too old. "
            "Run: pip install --upgrade databricks-sdk"
        )
        sys.exit(1)
    raise ImportError(
        "databricks-sdk is missing or too old — "
        "run: pip install --upgrade databricks-sdk"
    ) from _import_err


# ── Genie Space ──────────────────────────────────────────────────────────────

GENIE_INSTRUCTIONS = (
    "When counting disasters, count distinct disaster_id values. "
    "Severity ranges from 2 (low) to 5 (catastrophic). "
    "Federal aid amounts are in US dollars. "
    "When asked about 'recent' disasters, use year >= 2024."
)

GENIE_DESCRIPTION = (
    "Query FEMA disaster records (2020-2025) across 20 US states. "
    "Includes disaster types, severity ratings, affected populations, "
    "and federal aid amounts."
)


def _genie_id() -> str:
    """Generate a stable-shaped identifier for serialized Genie payload items."""
    return uuid.uuid4().hex


def _require_sdk_capability(obj, capability: str, upgrade_hint: str):
    """Fail fast with a clear upgrade message when newer SDK APIs are missing."""
    if not hasattr(obj, capability):
        raise RuntimeError(upgrade_hint)


def _build_genie_serialized_space(fq_table: str) -> str:
    """Build the serialized Genie Space payload for the FEMA disaster table."""
    serialized_space = {
        "version": 2,
        "config": {
            "sample_questions": sorted(
                [
                    {"id": _genie_id(), "question": ["How many disasters hit California in 2024?"]},
                    {"id": _genie_id(), "question": ["What was the total federal aid for hurricanes in 2024?"]},
                    {"id": _genie_id(), "question": ["Which state had the most severity-5 disasters?"]},
                    {"id": _genie_id(), "question": ["Show total affected population by disaster type"]},
                    {"id": _genie_id(), "question": ["Compare federal aid by state for 2024"]},
                ],
                key=lambda x: x["id"],
            )
        },
        "data_sources": {
            "tables": sorted(
                [
                    {
                        "identifier": fq_table,
                        "description": [
                            "FEMA disaster records from 2020-2025. Each row is a disaster declaration "
                            "with disaster_id, year, state, disaster_type, severity (2-5), "
                            "affected_population, federal_aid_amount (USD), and declaration_date."
                        ],
                        "column_configs": sorted(
                            [
                                {"column_name": "disaster_type", "enable_format_assistance": True, "enable_entity_matching": True},
                                {"column_name": "severity", "enable_format_assistance": True},
                                {"column_name": "state", "enable_format_assistance": True, "enable_entity_matching": True},
                                {"column_name": "year", "enable_format_assistance": True},
                            ],
                            key=lambda c: c["column_name"],
                        ),
                    },
                ],
                key=lambda t: t["identifier"],
            )
        },
        "instructions": {
            "text_instructions": [
                {"id": _genie_id(), "content": [GENIE_INSTRUCTIONS]}
            ],
            "example_question_sqls": sorted(
                [
                    {
                        "id": _genie_id(),
                        "question": ["How many disasters hit California in 2024?"],
                        "sql": [
                            f"SELECT COUNT(*) as disaster_count "
                            f"FROM {fq_table} "
                            "WHERE state = 'California' AND year = 2024",
                        ],
                    },
                    {
                        "id": _genie_id(),
                        "question": ["What was the total federal aid for hurricanes in 2024?"],
                        "sql": [
                            f"SELECT SUM(federal_aid_amount) as total_aid "
                            f"FROM {fq_table} "
                            "WHERE disaster_type = 'Hurricane' AND year = 2024",
                        ],
                    },
                    {
                        "id": _genie_id(),
                        "question": ["Which states had severity-5 disasters in 2024?"],
                        "sql": [
                            f"SELECT DISTINCT state, disaster_type, severity "
                            f"FROM {fq_table} "
                            "WHERE severity = 5 AND year = 2024 "
                            "ORDER BY state",
                        ],
                    },
                ],
                key=lambda x: x["id"],
            ),
        },
    }
    return json.dumps(serialized_space)


def create_genie_space(
    w: WorkspaceClient, fq_table: str, warehouse_id: str
) -> str:
    """Create a Genie Space for the FEMA disaster table.

    Returns the Genie Space ID.
    """
    print(f"\nCreating Genie Space: FEMA Disaster Data")
    _require_sdk_capability(
        w,
        "genie",
        "This databricks-sdk version does not expose Genie APIs. "
        "Run: pip install --upgrade databricks-sdk",
    )
    _require_sdk_capability(
        w.genie,
        "create_space",
        "This databricks-sdk version does not expose w.genie.create_space. "
        "Run: pip install --upgrade databricks-sdk",
    )

    space = w.genie.create_space(
        warehouse_id=warehouse_id,
        title="FEMA Disaster Data",
        description=GENIE_DESCRIPTION,
        serialized_space=_build_genie_serialized_space(fq_table),
    )
    print(f"  Genie Space created: {space.space_id}")
    return space.space_id


# ── Vector Search ────────────────────────────────────────────────────────────

def create_vector_search(
    vs_endpoint_name: str,
    vs_index_name: str,
    fq_policy_table: str,
    embedding_model: str,
) -> None:
    """Create a Vector Search endpoint and Delta Sync index.

    Waits for the endpoint to come ONLINE and for the index to sync.
    """
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()

    # --- Endpoint ---
    try:
        vsc.get_endpoint(vs_endpoint_name)
        print(f"Vector Search endpoint already exists: {vs_endpoint_name}")
    except Exception:
        print(f"Creating Vector Search endpoint: {vs_endpoint_name}...")
        vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

    print("\nWaiting for endpoint to be ONLINE...")
    for i in range(60):
        ep = vsc.get_endpoint(vs_endpoint_name)
        status = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ONLINE":
            print(f"  Endpoint is ONLINE")
            break
        time.sleep(10)
        if i % 6 == 0:
            print(f"  Status: {status} (waiting...)")
    else:
        print(f"  Endpoint status: {status} -- may still be provisioning.")

    # --- Delta Sync Index ---
    try:
        vsc.get_index(vs_endpoint_name, vs_index_name)
        print(f"\nVector Search index already exists: {vs_index_name}")
    except Exception:
        print(f"\nCreating Delta Sync Index: {vs_index_name}...")
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            source_table_name=fq_policy_table,
            index_name=vs_index_name,
            pipeline_type="TRIGGERED",
            primary_key="doc_id",
            embedding_source_column="text",
            embedding_model_endpoint_name=embedding_model,
            columns_to_sync=["doc_id", "doc_uri", "text"],
        )
        print(f"  Embedding model: {embedding_model}")
        print(f"  Source table: {fq_policy_table}")

    print("\nWaiting for index sync to complete...")
    for i in range(60):
        try:
            idx = vsc.get_index(vs_endpoint_name, vs_index_name)
            if idx.describe().get("status", {}).get("ready", False):
                print(f"  Index is READY")
                break
        except Exception:
            pass
        time.sleep(10)
        if i % 6 == 0:
            print(f"  Syncing... (waiting)")
    else:
        print(f"  Index may still be syncing. Check status in the UI.")

    # --- Verify ---
    try:
        idx = vsc.get_index(vs_endpoint_name, vs_index_name)
        results = idx.similarity_search(
            query_text="What are FEMA's evacuation protocols for wildfire zones?",
            columns=["doc_id", "text"],
            num_results=3,
        )
        print("\nVector Search test query successful!")
        for row in results.get("result", {}).get("data_array", []):
            print(f"  doc_id: {row[0]}, score: {row[2]:.4f}")
    except Exception as e:
        print(f"\nIndex not ready yet: {e}")


# ── Knowledge Assistant ──────────────────────────────────────────────────────

KA_DESCRIPTION = (
    "Answers questions about FEMA policies, evacuation protocols, "
    "safety guidelines, disaster declaration processes, and aid eligibility."
)

KA_INSTRUCTIONS = (
    "Use the vector search index to retrieve relevant FEMA policy documents. "
    "Always cite the specific policy document name in your answers. "
    "Reference specific protocol names, thresholds, and procedures. "
    "If the question involves multiple policies, synthesize across all relevant documents. "
    "If unsure, say so."
)

KA_SOURCE_NAME = "FEMA Policy Docs (Vector Search)"


def _get_existing_knowledge_assistant(
    w: WorkspaceClient, display_name: str
) -> KnowledgeAssistant | None:
    """Look up an existing Knowledge Assistant by display name."""
    for assistant in w.knowledge_assistants.list_knowledge_assistants():
        if assistant.display_name == display_name:
            return assistant
    return None


def _get_existing_knowledge_source(
    w: WorkspaceClient, parent: str, vs_index_name: str
) -> KnowledgeSource | None:
    """Look up an existing index-backed knowledge source for an assistant."""
    for source in w.knowledge_assistants.list_knowledge_sources(parent=parent):
        if source.index and source.index.index_name == vs_index_name:
            return source
    return None


def _knowledge_assistant_name(assistant: KnowledgeAssistant) -> str:
    """Return the resource name for a Knowledge Assistant."""
    if assistant.name:
        return assistant.name
    if assistant.id:
        return f"knowledge-assistants/{assistant.id}"
    raise RuntimeError("Knowledge Assistant response did not include a resource name.")


def create_knowledge_assistant(
    w: WorkspaceClient, vs_index_name: str, ka_name: str = "FEMA Policy Assistant"
) -> str:
    """Create or reuse a Knowledge Assistant with the VS index as a knowledge source.

    Returns the Knowledge Assistant resource name.
    """
    print(f"\nCreating Knowledge Assistant: {ka_name}")
    _require_sdk_capability(
        w,
        "knowledge_assistants",
        "This databricks-sdk version does not expose Knowledge Assistant APIs. "
        "Run: pip install --upgrade databricks-sdk",
    )

    assistant = _get_existing_knowledge_assistant(w, ka_name)

    if assistant is None:
        assistant = w.knowledge_assistants.create_knowledge_assistant(
            knowledge_assistant=KnowledgeAssistant(
                display_name=ka_name,
                description=KA_DESCRIPTION,
                instructions=KA_INSTRUCTIONS,
            )
        )
        print(f"  Knowledge Assistant created: {assistant.id}")
    else:
        print(f"  Knowledge Assistant already exists: {assistant.id}")

    assistant_name = _knowledge_assistant_name(assistant)
    existing_source = _get_existing_knowledge_source(w, assistant_name, vs_index_name)

    if existing_source is None:
        source = w.knowledge_assistants.create_knowledge_source(
            parent=assistant_name,
            knowledge_source=KnowledgeSource(
                display_name=KA_SOURCE_NAME,
                description="7 FEMA policy documents indexed via Delta Sync",
                source_type="index",
                index=IndexSpec(
                    index_name=vs_index_name,
                    text_col="text",
                    doc_uri_col="doc_uri",
                ),
            ),
        )
        print(f"  Knowledge source attached: {source.name}")
    else:
        print(f"  Knowledge source already exists: {existing_source.name}")

    print(f"  Syncing knowledge sources...")
    w.knowledge_assistants.sync_knowledge_sources(name=assistant_name)
    print(f"  Sync triggered")

    return assistant_name


# ── Supervisor instructions (manual step) ────────────────────────────────────

SUPERVISOR_INSTRUCTIONS = """\
When a question involves BOTH data AND policy, delegate to both agents and synthesize.
For data questions: route to Genie - FEMA Data.
For policy questions: route to Knowledge Assistant - FEMA Policies.
Always include specific numbers/statistics when available.
Reference specific protocol names or guidelines by name."""


def print_supervisor_instructions(genie_name: str, ka_name: str):
    """Print Supervisor Agent creation instructions."""
    print("\n" + "=" * 60)
    print("SUPERVISOR AGENT CREATION (Manual - No API Available)")
    print("=" * 60)
    print("\nSteps:")
    print("  1. Go to Agents in the workspace sidebar")
    print("  2. From the Supervisor Agent tile, click Build")
    print("  3. Name: FEMA Disaster Response Supervisor")
    print("  4. Add subagents:")
    print(f"     - Genie Space: {genie_name}")
    print(f"     - Knowledge Assistant: {ka_name}")
    print("  5. Paste these instructions:")
    print(SUPERVISOR_INSTRUCTIONS)
    print("  6. Click Create Agent")
    print("  7. Copy the endpoint name from 'See Agent status'")
    print("=" * 60)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup FEMA multi-agent demo agents")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--table-name", default="disaster_data", help="Disaster table name")
    parser.add_argument("--policy-table-name", default="policy_docs", help="Policy table name")
    parser.add_argument("--vs-endpoint", default="fema_vs_endpoint", help="VS endpoint name")
    parser.add_argument("--embedding-model", default="databricks-gte-large-en")
    parser.add_argument("--warehouse-id", required=True, help="SQL Warehouse ID for Genie Space")
    parser.add_argument("--profile", default=None, help="Databricks CLI profile")

    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()

    fq_table = f"{args.catalog}.{args.schema}.{args.table_name}"
    fq_policy_table = f"{args.catalog}.{args.schema}.{args.policy_table_name}"
    vs_index_name = f"{args.catalog}.{args.schema}.policy_docs_index"

    print("=" * 60)
    print("CREATING AGENTS FOR FEMA MULTI-AGENT DEMO")
    print("=" * 60)

    genie_id = create_genie_space(w, fq_table, args.warehouse_id)
    create_vector_search(args.vs_endpoint, vs_index_name, fq_policy_table, args.embedding_model)
    ka_name = create_knowledge_assistant(w, vs_index_name)

    print_supervisor_instructions("FEMA Disaster Data", "FEMA Policy Assistant")

    print("\n" + "=" * 60)
    print("AGENT SETUP COMPLETE")
    print("=" * 60)
    print(f"\nGenie Space: {genie_id}")
    print(f"Knowledge Assistant: {ka_name}")
    print("\nNext: Create Supervisor Agent manually (see instructions above)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

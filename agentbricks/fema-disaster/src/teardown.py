"""
FEMA Multi-Agent Demo — Teardown Script

Removes all resources created by the setup job and manual steps:
  1. Supervisor serving endpoint (if provided)
  2. Knowledge Assistant
  3. Vector Search index
  4. Vector Search endpoint
  5. Genie Space (if ID provided)
  6. Unity Catalog tables, schema, and catalog

Usage:
    python src/teardown.py \
        --catalog jules_catalog \
        --schema fema_demo \
        --vs-endpoint fema_vs_endpoint \
        --supervisor-endpoint <endpoint-name> \
        --genie-space-id <space-id>

Add --delete-catalog to also drop the UC catalog (skipped by default).
"""

import argparse
import sys
import time

from databricks.sdk import WorkspaceClient


def _delete_serving_endpoint(w: WorkspaceClient, endpoint_name: str):
    """Delete a model serving endpoint."""
    print(f"\n[1/7] Deleting serving endpoint: {endpoint_name}")
    try:
        w.serving_endpoints.delete(endpoint_name)
        print(f"  Deleted.")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            print(f"  Not found — skipping.")
        else:
            print(f"  Error: {e}")


def _delete_knowledge_assistant(w: WorkspaceClient, ka_name: str):
    """Delete a Knowledge Assistant and its knowledge sources."""
    print(f"\n[2/7] Deleting Knowledge Assistant: {ka_name}")
    try:
        for assistant in w.knowledge_assistants.list_knowledge_assistants():
            if assistant.display_name == ka_name:
                name = assistant.name or f"knowledge-assistants/{assistant.id}"
                # Delete knowledge sources first
                for source in w.knowledge_assistants.list_knowledge_sources(parent=name):
                    print(f"  Deleting knowledge source: {source.name}")
                    w.knowledge_assistants.delete_knowledge_source(name=source.name)
                # Delete the assistant
                w.knowledge_assistants.delete_knowledge_assistant(name=name)
                print(f"  Deleted.")
                return
        print(f"  Not found — skipping.")
    except Exception as e:
        print(f"  Error: {e}")


def _delete_vector_search_index(vs_endpoint_name: str, vs_index_name: str):
    """Delete a Vector Search index."""
    print(f"\n[3/7] Deleting Vector Search index: {vs_index_name}")
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient()
        vsc.delete_index(vs_endpoint_name, vs_index_name)
        print(f"  Deleted. Waiting 5s for cleanup...")
        time.sleep(5)
    except Exception as e:
        if "does not exist" in str(e).lower() or "NOT_FOUND" in str(e):
            print(f"  Not found — skipping.")
        else:
            print(f"  Error: {e}")


def _delete_vector_search_endpoint(vs_endpoint_name: str):
    """Delete a Vector Search endpoint."""
    print(f"\n[4/7] Deleting Vector Search endpoint: {vs_endpoint_name}")
    try:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient()
        vsc.delete_endpoint(vs_endpoint_name)
        print(f"  Deleted.")
    except Exception as e:
        if "does not exist" in str(e).lower() or "NOT_FOUND" in str(e):
            print(f"  Not found — skipping.")
        else:
            print(f"  Error: {e}")


def _delete_genie_space(w: WorkspaceClient, space_id: str):
    """Delete a Genie Space."""
    print(f"\n[5/7] Deleting Genie Space: {space_id}")
    try:
        w.genie.delete_space(space_id)
        print(f"  Deleted.")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            print(f"  Not found — skipping.")
        else:
            print(f"  Error: {e}")


def _delete_uc_objects(w: WorkspaceClient, catalog: str, schema: str,
                       table_name: str, policy_table_name: str,
                       delete_catalog: bool):
    """Delete Unity Catalog tables, schema, and optionally catalog."""
    fq_table = f"{catalog}.{schema}.{table_name}"
    fq_policy = f"{catalog}.{schema}.{policy_table_name}"

    print(f"\n[6/7] Deleting UC tables")
    for table in [fq_table, fq_policy]:
        try:
            w.tables.delete(table)
            print(f"  Deleted table: {table}")
        except Exception as e:
            if "DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
                print(f"  Table not found: {table} — skipping.")
            else:
                print(f"  Error deleting {table}: {e}")

    print(f"\n[7/7] Deleting UC schema: {catalog}.{schema}")
    try:
        w.schemas.delete(f"{catalog}.{schema}")
        print(f"  Deleted schema: {catalog}.{schema}")
    except Exception as e:
        if "DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            print(f"  Schema not found — skipping.")
        else:
            print(f"  Error: {e}")

    if delete_catalog:
        print(f"\n  Deleting UC catalog: {catalog}")
        try:
            w.catalogs.delete(catalog)
            print(f"  Deleted catalog: {catalog}")
        except Exception as e:
            if "DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
                print(f"  Catalog not found — skipping.")
            else:
                print(f"  Error: {e}")
    else:
        print(f"\n  Keeping catalog: {catalog} (use --delete-catalog to remove)")


def main():
    parser = argparse.ArgumentParser(
        description="Tear down all FEMA multi-agent demo resources"
    )
    parser.add_argument("--catalog", default="jules_catalog", help="Unity Catalog name")
    parser.add_argument("--schema", default="fema_demo", help="Schema name")
    parser.add_argument("--table-name", default="disaster_data", help="Disaster table name")
    parser.add_argument("--policy-table-name", default="policy_docs", help="Policy table name")
    parser.add_argument("--vs-endpoint", default="fema_vs_endpoint", help="VS endpoint name")
    parser.add_argument("--supervisor-endpoint", default=None, help="Supervisor serving endpoint name")
    parser.add_argument("--genie-space-id", default=None, help="Genie Space ID")
    parser.add_argument("--ka-name", default="FEMA Policy Assistant", help="Knowledge Assistant name")
    parser.add_argument("--delete-catalog", action="store_true", help="Also delete the UC catalog")
    parser.add_argument("--profile", default=None, help="Databricks CLI profile")

    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    vs_index_name = f"{args.catalog}.{args.schema}.policy_docs_index"

    print("=" * 60)
    print("TEARING DOWN FEMA MULTI-AGENT DEMO RESOURCES")
    print("=" * 60)
    print(f"  Catalog:              {args.catalog}")
    print(f"  Schema:               {args.schema}")
    print(f"  VS Endpoint:          {args.vs_endpoint}")
    print(f"  VS Index:             {vs_index_name}")
    print(f"  Knowledge Assistant:  {args.ka_name}")
    print(f"  Supervisor Endpoint:  {args.supervisor_endpoint or '(not provided)'}")
    print(f"  Genie Space ID:       {args.genie_space_id or '(not provided)'}")
    print(f"  Delete Catalog:       {args.delete_catalog}")

    # 1. Supervisor serving endpoint
    if args.supervisor_endpoint:
        _delete_serving_endpoint(w, args.supervisor_endpoint)
    else:
        print("\n[1/7] Supervisor endpoint — skipped (not provided)")

    # 2. Knowledge Assistant
    _delete_knowledge_assistant(w, args.ka_name)

    # 3. Vector Search index
    _delete_vector_search_index(args.vs_endpoint, vs_index_name)

    # 4. Vector Search endpoint
    _delete_vector_search_endpoint(args.vs_endpoint)

    # 5. Genie Space
    if args.genie_space_id:
        _delete_genie_space(w, args.genie_space_id)
    else:
        print("\n[5/7] Genie Space — skipped (no --genie-space-id provided)")

    # 6-7. UC objects
    _delete_uc_objects(w, args.catalog, args.schema,
                       args.table_name, args.policy_table_name,
                       args.delete_catalog)

    print("\n" + "=" * 60)
    print("TEARDOWN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

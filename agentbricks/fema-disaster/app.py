"""FEMA Disaster Response Supervisor — Streamlit App.

Interactive GUI for querying the AgentBricks multi-agent Supervisor
(Genie + Knowledge Assistant) deployed as a Databricks App.
"""

import os
import json
import streamlit as st
from src.supervisor_client import get_workspace_client, query_supervisor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FEMA Disaster Response — Multi-Agent Supervisor",
    page_icon="\U0001f3db\ufe0f",  # 🏛️
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .fema-header {
        background-color: #003366;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .fema-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .fema-header p {
        color: #b0c4de;
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }
    .fema-header .emoji-row {
        font-size: 1.6rem;
        margin-top: 0.5rem;
        letter-spacing: 0.4rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-right: 0.5rem;
    }
    .badge-genie   { background-color: #1E88E5; }
    .badge-knowledge { background-color: #43A047; }
    .badge-both    { background-color: #7B1FA2; }
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="fema-header">
        <h1>FEMA Disaster Response Supervisor</h1>
        <p>Multi-Agent AI System &mdash; Genie Data Agent + Knowledge Assistant</p>
        <div class="emoji-row">\U0001f300 \U0001f525 \U0001f30a \U0001f3d4\ufe0f \U0001f32a\ufe0f</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Predefined queries
# ---------------------------------------------------------------------------
PREDEFINED_QUERIES = {
    "\U0001f5c4\ufe0f Data Queries (Genie)": [
        {"query": "How many disasters hit California in 2024?", "category": "Genie"},
        {
            "query": "What was the total federal aid for hurricane-related disasters in 2024?",
            "category": "Genie",
        },
    ],
    "\U0001f4da Policy Queries (Knowledge)": [
        {
            "query": "What are FEMA's evacuation protocols for wildfire zones?",
            "category": "Knowledge",
        },
        {
            "query": "Who is eligible for federal disaster assistance and what types of aid are available?",
            "category": "Knowledge",
        },
    ],
    "\U0001f500 Combined Queries (Both)": [
        {
            "query": "How many flood disasters occurred in 2024 and what are FEMA's flood response procedures?",
            "category": "Both",
        },
        {
            "query": "Which states had severity-5 disasters in 2024, and what safety protocols apply to those disaster types?",
            "category": "Both",
        },
    ],
}

BADGE_HTML = {
    "Genie": '<span class="badge badge-genie">\U0001f5c4\ufe0f Genie</span>',
    "Knowledge": '<span class="badge badge-knowledge">\U0001f4da Knowledge</span>',
    "Both": '<span class="badge badge-both">\U0001f500 Both</span>',
    "Custom": '<span class="badge" style="background-color:#616161;">\u2753 Custom</span>',
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    endpoint_name = st.text_input(
        "Supervisor Endpoint",
        value=os.environ.get("SUPERVISOR_ENDPOINT", ""),
        help="Name of the AgentBricks Supervisor serving endpoint",
    )

    st.divider()
    st.header("Select Queries")

    selected_queries = []
    for group_label, queries in PREDEFINED_QUERIES.items():
        with st.expander(group_label, expanded=True):
            for q in queries:
                if st.checkbox(q["query"][:70] + "..." if len(q["query"]) > 70 else q["query"], key=q["query"]):
                    selected_queries.append(q)

    st.divider()
    st.header("Custom Query")
    custom_query = st.text_area(
        "Ask a custom question",
        placeholder="e.g., What tornado safety protocols does FEMA recommend?",
    )
    custom_category = st.selectbox("Expected route", ["Custom", "Genie", "Knowledge", "Both"])

    st.divider()
    run_button = st.button("\U0001f680 Run Selected Queries", type="primary", use_container_width=True)
    clear_button = st.button("Clear Results", use_container_width=True)

# ---------------------------------------------------------------------------
# Handle clear
# ---------------------------------------------------------------------------
if clear_button:
    st.session_state.results = []
    st.rerun()

# ---------------------------------------------------------------------------
# Handle run
# ---------------------------------------------------------------------------
if run_button:
    if not endpoint_name:
        st.error("Please enter the Supervisor Endpoint name in the sidebar.")
    else:
        all_queries = list(selected_queries)
        if custom_query.strip():
            all_queries.append({"query": custom_query.strip(), "category": custom_category})

        if not all_queries:
            st.warning("No queries selected. Check some predefined queries or enter a custom one.")
        else:
            try:
                client = get_workspace_client()
            except Exception as e:
                st.error(f"Failed to connect to Databricks: {e}")
                st.stop()

            st.session_state.results = []
            progress = st.progress(0, text="Querying supervisor...")

            for i, q in enumerate(all_queries):
                progress.progress(
                    (i) / len(all_queries),
                    text=f"Querying: {q['query'][:60]}...",
                )
                try:
                    result = query_supervisor(client, endpoint_name, q["query"])
                    st.session_state.results.append(
                        {
                            "query": q["query"],
                            "category": q["category"],
                            "answer": result["answer"],
                            "raw_response": result["raw_response"],
                            "status": "success",
                        }
                    )
                except Exception as e:
                    st.session_state.results.append(
                        {
                            "query": q["query"],
                            "category": q["category"],
                            "answer": str(e),
                            "raw_response": {},
                            "status": "error",
                        }
                    )

            progress.progress(1.0, text="Done!")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
tab_results, tab_arch = st.tabs(["\U0001f4cb Results", "\U0001f3d7\ufe0f Architecture"])

with tab_results:
    if not st.session_state.results:
        # Hero / empty state
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 4rem; letter-spacing: 1rem;">\U0001f300 \U0001f525 \U0001f30a \U0001f3d4\ufe0f \U0001f32a\ufe0f</div>
                <h2 style="color: #003366;">Ready to Query the Supervisor</h2>
                <p style="color: #666; max-width: 500px; margin: 0 auto;">
                    Select predefined queries from the sidebar or type your own question
                    to get started. The Supervisor will route your queries to the appropriate
                    agent &mdash; Genie for data, Knowledge Assistant for policy, or both.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Summary bar
        successes = sum(1 for r in st.session_state.results if r["status"] == "success")
        errors = sum(1 for r in st.session_state.results if r["status"] == "error")
        st.markdown(
            f"**{len(st.session_state.results)} queries** | "
            f"\u2705 {successes} succeeded | \u274c {errors} failed"
        )
        st.divider()

        for r in st.session_state.results:
            badge = BADGE_HTML.get(r["category"], BADGE_HTML["Custom"])
            status_icon = "\u2705" if r["status"] == "success" else "\u274c"

            st.markdown(
                f'<div class="result-card">'
                f"{badge} {status_icon}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{r['query']}**")

            if r["status"] == "success":
                st.markdown(r["answer"])
            else:
                st.error(r["answer"])

            with st.expander("Raw Response"):
                st.json(r["raw_response"])

            st.divider()

with tab_arch:
    st.subheader("Multi-Agent Supervisor Architecture")

    svg_path = os.path.join(os.path.dirname(__file__), "images", "fema_multi_agent_supervisor.svg")
    if os.path.exists(svg_path):
        st.image(svg_path, use_column_width=True)
    else:
        st.info("Architecture diagram not found. Expected at: images/fema_multi_agent_supervisor.svg")

    st.markdown(
        """
        | Agent | Type | Data Source | Capability |
        |-------|------|-------------|------------|
        | **Genie Space** | Structured Data | Delta table (200 records) | SQL queries, counts, aggregations, filtering |
        | **Knowledge Assistant** | Retrieval | Vector Search (7 docs) | Policy Q&A, protocols, guidelines, eligibility |
        | **Supervisor** | Router / Orchestrator | Both above | Intent classification, routing, response synthesis |
        """
    )

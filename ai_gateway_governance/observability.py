"""SQL query templates and helpers for querying AI Gateway inference tables."""

ALL_REQUESTS_QUERY = """
SELECT
    request_time,
    status_code,
    request,
    response,
    execution_time_ms
FROM {catalog}.{schema}.{prefix}_request_response
ORDER BY request_time DESC
LIMIT {limit}
"""

BLOCKED_REQUESTS_QUERY = """
SELECT
    request_time,
    status_code,
    request,
    response,
    execution_time_ms
FROM {catalog}.{schema}.{prefix}_request_response
WHERE status_code != 200
ORDER BY request_time DESC
LIMIT {limit}
"""

QUERY_MAP = {
    "all": ALL_REQUESTS_QUERY,
    "blocked": BLOCKED_REQUESTS_QUERY,
}


def build_query(
    query_name: str,
    catalog: str,
    schema: str,
    prefix: str,
    limit: int = 50,
) -> str:
    """Return a formatted SQL query string."""
    template = QUERY_MAP[query_name]
    return template.format(catalog=catalog, schema=schema, prefix=prefix, limit=limit)


def query_inference_table(spark, catalog: str, schema: str, prefix: str, query_name: str = "all", limit: int = 50):
    """Execute a query against the inference table and return a Spark DataFrame.

    Meant for use in Databricks notebooks where `spark` is available.
    """
    sql = build_query(query_name, catalog, schema, prefix, limit)
    return spark.sql(sql)

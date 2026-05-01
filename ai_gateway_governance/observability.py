"""SQL query templates and helpers for querying AI Gateway inference tables.

Inference table naming convention: `{catalog}`.`{schema}`.`{schema}_payload`
"""

ALL_REQUESTS_QUERY = """
SELECT
    event_time,
    request_id,
    status_code,
    requester,
    latency_ms,
    request,
    response
FROM `{catalog}`.`{schema}`.`{schema}_payload`
ORDER BY event_time DESC
LIMIT {limit}
"""

BLOCKED_REQUESTS_QUERY = """
SELECT
    event_time,
    request_id,
    status_code,
    requester,
    latency_ms,
    request,
    logging_error_codes
FROM `{catalog}`.`{schema}`.`{schema}_payload`
WHERE status_code != 200
ORDER BY event_time DESC
LIMIT {limit}
"""

TOKEN_USAGE_QUERY = """
SELECT
    CASE WHEN status_code = 200 THEN 'allowed' ELSE 'blocked' END AS outcome,
    COUNT(*)                                                        AS request_count,
    SUM(CAST(response:usage:total_tokens      AS BIGINT))          AS total_tokens,
    SUM(CAST(response:usage:prompt_tokens     AS BIGINT))          AS input_tokens,
    SUM(CAST(response:usage:completion_tokens AS BIGINT))          AS output_tokens,
    ROUND(AVG(latency_ms), 0)                                      AS avg_latency_ms
FROM `{catalog}`.`{schema}`.`{schema}_payload`
WHERE event_time >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
GROUP BY 1
ORDER BY total_tokens DESC NULLS LAST
LIMIT {limit}
"""

SYSTEM_USAGE_QUERY = """
SELECT
    endpoint_name,
    DATE_TRUNC('hour', usage_time)    AS hour,
    SUM(input_tokens)                 AS input_tokens,
    SUM(output_tokens)                AS output_tokens,
    SUM(input_tokens + output_tokens) AS total_tokens
FROM system.ai_gateway.usage
WHERE endpoint_name = '{endpoint_name}'
  AND usage_time >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
GROUP BY endpoint_name, DATE_TRUNC('hour', usage_time)
ORDER BY hour DESC
"""

QUERY_MAP = {
    "all": ALL_REQUESTS_QUERY,
    "blocked": BLOCKED_REQUESTS_QUERY,
    "token_usage": TOKEN_USAGE_QUERY,
    "system_usage": SYSTEM_USAGE_QUERY,
}


def build_query(
    query_name: str,
    catalog: str = "",
    schema: str = "",
    limit: int = 50,
    **kwargs,
) -> str:
    """Return a formatted SQL query string.

    Table name is derived automatically as {schema}_payload.
    """
    template = QUERY_MAP[query_name]
    return template.format(catalog=catalog, schema=schema, limit=limit, **kwargs)


def query_inference_table(spark, catalog: str, schema: str, query_name: str = "all", limit: int = 50):
    """Execute a query against the inference table and return a Spark DataFrame."""
    sql = build_query(query_name, catalog, schema, limit)
    return spark.sql(sql)

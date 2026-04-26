"""Configure Unity AI Gateway endpoint: guardrails, inference tables, and usage tracking."""

from dataclasses import dataclass, field
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayGuardrailParameters,
    AiGatewayGuardrailPiiBehavior,
    AiGatewayGuardrailPiiBehaviorBehavior,
    AiGatewayGuardrails,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
)


@dataclass
class GatewayConfig:
    endpoint_name: str
    catalog_name: str
    schema_name: str
    table_name_prefix: str = "coding_agents"

    pii_behavior: str = "BLOCK"
    safety_enabled: bool = True
    invalid_keywords: list = field(default_factory=list)
    valid_topics: list = field(default_factory=list)

    inference_table_enabled: bool = True
    usage_tracking_enabled: bool = True


_PII_BEHAVIORS = {
    "BLOCK": AiGatewayGuardrailPiiBehaviorBehavior.BLOCK,
    "MASK": AiGatewayGuardrailPiiBehaviorBehavior.MASK,
    "NONE": AiGatewayGuardrailPiiBehaviorBehavior.NONE,
}


def _build_guardrail_params(config: GatewayConfig) -> Optional[AiGatewayGuardrailParameters]:
    pii = AiGatewayGuardrailPiiBehavior(
        behavior=_PII_BEHAVIORS.get(config.pii_behavior.upper(), AiGatewayGuardrailPiiBehaviorBehavior.BLOCK)
    )
    return AiGatewayGuardrailParameters(
        pii=pii,
        safety=config.safety_enabled,
        invalid_keywords=config.invalid_keywords or None,
        valid_topics=config.valid_topics or None,
    )


def configure_gateway(config: GatewayConfig, client: WorkspaceClient) -> AiGatewayConfig:
    """Apply AI Gateway configuration to an existing serving endpoint."""
    guardrail_params = _build_guardrail_params(config)
    guardrails = AiGatewayGuardrails(input=guardrail_params, output=guardrail_params)

    inference_table = AiGatewayInferenceTableConfig(
        catalog_name=config.catalog_name,
        schema_name=config.schema_name,
        table_name_prefix=config.table_name_prefix,
        enabled=config.inference_table_enabled,
    )

    usage_tracking = AiGatewayUsageTrackingConfig(enabled=config.usage_tracking_enabled)

    result = client.serving_endpoints.put_ai_gateway(
        name=config.endpoint_name,
        guardrails=guardrails,
        inference_table_config=inference_table,
        usage_tracking_config=usage_tracking,
    )
    return result


def get_gateway_status(endpoint_name: str, client: WorkspaceClient) -> dict:
    """Retrieve and return the current AI Gateway configuration as a dict."""
    endpoint = client.serving_endpoints.get(endpoint_name)
    gw = endpoint.ai_gateway
    if gw is None:
        return {"status": "No AI Gateway config found"}

    status = {"endpoint": endpoint_name}

    if gw.guardrails and gw.guardrails.input:
        g = gw.guardrails.input
        status["guardrails_input"] = {
            "pii": str(g.pii.behavior) if g.pii else "disabled",
            "safety": g.safety,
            "invalid_keywords": g.invalid_keywords or [],
            "valid_topics": g.valid_topics or [],
        }

    if gw.inference_table_config:
        t = gw.inference_table_config
        status["inference_table"] = {
            "enabled": t.enabled,
            "catalog": t.catalog_name,
            "schema": t.schema_name,
            "prefix": t.table_name_prefix,
        }

    if gw.usage_tracking_config:
        status["usage_tracking"] = {"enabled": gw.usage_tracking_config.enabled}

    return status


def print_gateway_summary(endpoint_name: str, client: WorkspaceClient) -> None:
    """Pretty-print the current gateway configuration."""
    status = get_gateway_status(endpoint_name, client)
    print(f"{'=' * 60}")
    print(f"  AI Gateway Configuration: {endpoint_name}")
    print(f"{'=' * 60}")

    if "guardrails_input" in status:
        g = status["guardrails_input"]
        print(f"\n  Guardrails (input & output):")
        print(f"    PII:              {g['pii']}")
        print(f"    Safety:           {g['safety']}")
        if g["invalid_keywords"]:
            print(f"    Invalid Keywords: {g['invalid_keywords']}")
        if g["valid_topics"]:
            print(f"    Valid Topics:     {g['valid_topics']}")

    if "inference_table" in status:
        t = status["inference_table"]
        print(f"\n  Inference Tables:")
        print(f"    Enabled:  {t['enabled']}")
        print(f"    Location: {t['catalog']}.{t['schema']}.{t['prefix']}*")

    if "usage_tracking" in status:
        print(f"\n  Usage Tracking:")
        print(f"    Enabled:  {status['usage_tracking']['enabled']}")

    print(f"\n{'=' * 60}")

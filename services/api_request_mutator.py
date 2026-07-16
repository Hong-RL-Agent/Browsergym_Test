"""Rule-based API action generation for the J.A.W.S API fuzz baseline."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping
from urllib.parse import quote

from services.api_catalog_service import ApiEndpoint, ApiSite
from services.owasp_test_catalog_service import safe_test_metadata


API_MUTATIONS = [
    "send_valid_request",
    "missing_required_param",
    "empty_param",
    "null_param",
    "invalid_type_param",
    "long_string_param",
    "boundary_number_param",
    "invalid_enum_param",
    "missing_auth",
    "invalid_auth",
    "expired_auth",
    "invalid_content_type",
    "repeat_request",
    "parallel_request",
    "check_response_schema",
    "check_latency",
]


@dataclass(frozen=True)
class ApiFuzzCase:
    observation: Dict[str, Any]
    action: Dict[str, Any]
    request: Dict[str, Any]


def build_fuzz_cases(site: ApiSite, endpoint: ApiEndpoint, max_cases: int | None = None,
                     allow_mutating: bool = False) -> List[ApiFuzzCase]:
    if endpoint.method not in {"GET", "HEAD"} and not (
        allow_mutating and site.allow_mutating_requests and endpoint.test_safe
    ):
        return []
    base_request = _valid_request(site, endpoint)
    cases = [
        _case(site, endpoint, mutation, _mutate_request(base_request, site, endpoint, mutation))
        for mutation in API_MUTATIONS
        if _mutation_applies(endpoint, mutation) and safe_test_metadata(mutation) is not None
    ]
    if max_cases is not None and max_cases > 0:
        return cases[:max_cases]
    return cases


def build_url(base_url: str, path: str, path_params: Mapping[str, Any] | None = None) -> str:
    path_params = path_params or {}
    resolved = path
    for key, value in path_params.items():
        encoded = quote(str(value), safe="")
        resolved = resolved.replace("{" + str(key) + "}", encoded).replace(":" + str(key), encoded)
    if not resolved.startswith("/"):
        resolved = "/" + resolved
    return base_url.rstrip("/") + resolved


def _case(site: ApiSite, endpoint: ApiEndpoint, mutation: str, request: Dict[str, Any]) -> ApiFuzzCase:
    safety = safe_test_metadata(mutation) or {}
    observation = {
        "source": "api",
        "site_id": site.site_id,
        "api_id": endpoint.api_id,
        "method": endpoint.method,
        "path": endpoint.path,
        "expected_status": endpoint.expected_status,
        "auth_required": endpoint.auth_required,
        "query_schema": endpoint.query_schema,
        "path_schema": endpoint.path_schema,
        "body_schema": endpoint.body_schema,
        "response_schema": endpoint.response_schema,
    }
    action = {
        "source": "api",
        "action_type": "api_mutation",
        "mutation": mutation,
        "site_id": site.site_id,
        "api_id": endpoint.api_id,
        "method": endpoint.method,
        "url": request["url"],
        "owasp_refs": list(safety.get("owasp", [])),
        "safety": {"mode": "non-destructive", "risk": safety.get("risk", "low"), "allowlisted": True},
    }
    return ApiFuzzCase(observation=observation, action=action, request=request)


def _valid_request(site: ApiSite, endpoint: ApiEndpoint) -> Dict[str, Any]:
    path_params = {key: _sample_value(schema) for key, schema in endpoint.path_schema.items()}
    headers = {
        "Accept": "application/json",
        "User-Agent": "jaws-api-fuzz/1.0",
    }
    if endpoint.method in {"POST", "PUT", "PATCH"}:
        headers["Content-Type"] = "application/json"
    headers.update({key: str(_sample_value(schema)) for key, schema in endpoint.header_schema.items()})
    if endpoint.auth_required:
        headers.update(_auth_headers(site.auth, "valid"))
    return {
        "method": endpoint.method,
        "url": build_url(site.base_url, endpoint.path, path_params),
        "query": {key: _sample_value(schema) for key, schema in endpoint.query_schema.items()},
        "headers": headers,
        "body": {key: _sample_value(schema) for key, schema in endpoint.body_schema.items()},
        "path_params": path_params,
    }


def _mutate_request(
    request: Mapping[str, Any],
    site: ApiSite,
    endpoint: ApiEndpoint,
    mutation: str,
) -> Dict[str, Any]:
    mutated = copy.deepcopy(dict(request))
    query = mutated.setdefault("query", {})
    body = mutated.setdefault("body", {})
    headers = mutated.setdefault("headers", {})
    target_name, target_container, target_schema = _target_param(endpoint, query, body)

    if mutation == "missing_required_param" and target_name:
        target_container.pop(target_name, None)
    elif mutation == "empty_param" and target_name:
        target_container[target_name] = ""
    elif mutation == "null_param" and target_name:
        target_container[target_name] = None
    elif mutation == "invalid_type_param" and target_name:
        target_container[target_name] = _invalid_type_value(target_schema)
    elif mutation == "long_string_param" and target_name:
        target_container[target_name] = "A" * 4096
    elif mutation == "boundary_number_param" and target_name:
        target_container[target_name] = -1 if _schema_type(target_schema) in {"number", "integer"} else "999999999999999999999999"
    elif mutation == "invalid_enum_param":
        enum_name = next(iter(endpoint.enum_values), target_name)
        if enum_name:
            container = query if enum_name in query else body
            container[enum_name] = "__J_A_W_S_INVALID_ENUM__"
    elif mutation == "missing_auth":
        _strip_auth(headers)
    elif mutation == "invalid_auth":
        headers.update(_auth_headers(site.auth, "invalid"))
    elif mutation == "expired_auth":
        headers.update(_auth_headers(site.auth, "expired"))
    elif mutation == "invalid_content_type":
        headers["Content-Type"] = "text/plain"
    return mutated


def _mutation_applies(endpoint: ApiEndpoint, mutation: str) -> bool:
    has_params = bool(endpoint.query_schema or endpoint.body_schema or endpoint.path_schema)
    if mutation in {
        "missing_required_param",
        "empty_param",
        "null_param",
        "invalid_type_param",
        "long_string_param",
        "boundary_number_param",
    }:
        return has_params
    if mutation == "invalid_enum_param":
        return bool(endpoint.enum_values) or has_params
    if mutation in {"missing_auth", "invalid_auth", "expired_auth"}:
        return endpoint.auth_required
    if mutation == "invalid_content_type":
        return endpoint.method in {"POST", "PUT", "PATCH"}
    return True


def _target_param(
    endpoint: ApiEndpoint,
    query: Dict[str, Any],
    body: Dict[str, Any],
) -> tuple[str, Dict[str, Any], Any]:
    preferred = endpoint.required_params or list(endpoint.query_schema) or list(endpoint.body_schema)
    for name in preferred:
        if name in query:
            return name, query, endpoint.query_schema.get(name)
        if name in body:
            return name, body, endpoint.body_schema.get(name)
    return "", query, "string"


def _sample_value(schema: Any) -> Any:
    schema_type = _schema_type(schema)
    if isinstance(schema, Mapping) and isinstance(schema.get("enum"), list) and schema.get("enum"):
        return schema["enum"][0]
    if schema_type in {"integer", "number"}:
        return 1
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}
    return "SDF"


def _invalid_type_value(schema: Any) -> Any:
    schema_type = _schema_type(schema)
    if schema_type in {"integer", "number"}:
        return "not-a-number"
    if schema_type == "boolean":
        return "not-a-boolean"
    if schema_type == "array":
        return "not-an-array"
    if schema_type == "object":
        return "not-an-object"
    return {"unexpected": "object"}


def _schema_type(schema: Any) -> str:
    if isinstance(schema, str):
        return schema.lower()
    if isinstance(schema, Mapping):
        return str(schema.get("type") or "string").lower()
    return "string"


def _auth_headers(auth: Mapping[str, Any], mode: str) -> Dict[str, str]:
    header_name = str(auth.get("header") or "Authorization")
    if mode == "valid":
        token = str(auth.get("token") or auth.get("valid_token") or "Bearer jaws-valid-token")
    elif mode == "expired":
        token = str(auth.get("expired_token") or "Bearer jaws-expired-token")
    else:
        token = str(auth.get("invalid_token") or "Bearer jaws-invalid-token")
    if header_name.lower() == "authorization" and not re.match(r"^[A-Za-z]+\s+", token):
        token = f"Bearer {token}"
    return {header_name: token}


def _strip_auth(headers: Dict[str, Any]) -> None:
    for key in list(headers):
        if key.lower() in {"authorization", "x-api-key", "x-auth-token"}:
            headers.pop(key, None)

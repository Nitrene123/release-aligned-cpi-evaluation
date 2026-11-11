#!/usr/bin/env python3
"""Target-free point-in-time contract audit for forecast bundles (stdlib only).

Input is JSON with origins, model-specific required features, vintage-tagged
inputs, planned forecast keys, and forecast/status rows. No numeric target
values are accepted. See PIT_AUDIT_README.md for the schema and limits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _month_index(month: str) -> int:
    if len(month) != 7 or month[4] != "-":
        raise ValueError(f"invalid YYYY-MM month: {month!r}")
    year, mon = int(month[:4]), int(month[5:])
    if not 1 <= mon <= 12:
        raise ValueError(f"invalid YYYY-MM month: {month!r}")
    return year * 12 + mon - 1


def add_month(month: str, offset: int) -> str:
    value = _month_index(month) + int(offset)
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include a UTC offset: {value!r}")
    return parsed


def _finite_value(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False



_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_sha256(value: Any) -> str:
    """Hash JSON parameters with a stable UTF-8 canonical representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_dependency_graph(
    graph: dict[str, Any], root_node_id: str,
    origin_time_utc: str | datetime, origin_month: str,
    expected_reference_month: str | None = None,
    expected_transformation_id: str | None = None,
) -> list[dict[str, str]]:
    """Recursively validate a derived input's declared provenance DAG.

    Edges point from child to parent. Source nodes record the selected source
    version and release time. Derived nodes record an invocation-specific
    transformation ID, parameters, and their hashes. Completeness and trust of
    the supplied edge and transformation registries remain explicit assumptions.
    """
    issues: list[dict[str, str]] = []
    def add(code: str, message: str, node_id: str = "", path: tuple[str, ...] = ()) -> None:
        issues.append({"code": code, "message": message, "key": node_id,
                       "path": " -> ".join(path)})
    if not isinstance(graph, dict):
        add("DEPENDENCY_GRAPH_INVALID", "dependency_graph must be an object")
        return issues
    try:
        if isinstance(origin_time_utc, str):
            origin_time = _aware_datetime(origin_time_utc)
        elif isinstance(origin_time_utc, datetime):
            if origin_time_utc.tzinfo is None or origin_time_utc.utcoffset() is None:
                raise ValueError("origin timestamp must include a UTC offset")
            origin_time = origin_time_utc
        else:
            raise TypeError("origin timestamp must be an ISO-8601 string or aware datetime")
        _month_index(origin_month)
    except (TypeError, ValueError) as exc:
        add("DEPENDENCY_ORIGIN_CLOCK_INVALID", str(exc))
        return issues

    raw_nodes = graph.get("nodes")
    if not isinstance(raw_nodes, list):
        add("DEPENDENCY_NODES_MISSING", "dependency_graph.nodes must be a list")
        return issues
    nodes: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            add("DEPENDENCY_NODE_INVALID", f"node at index {index} must be an object")
            continue
        node_id = str(node.get("node_id", "")).strip()
        if not node_id:
            add("DEPENDENCY_NODE_ID_MISSING", f"node at index {index} has no node_id")
            continue
        if node_id in nodes:
            add("DEPENDENCY_NODE_DUPLICATE", "node_id is duplicated", node_id, (node_id,))
            continue
        nodes[node_id] = node

    raw_edges = graph.get("edges")
    if not isinstance(raw_edges, list):
        add("DEPENDENCY_EDGES_MISSING", "dependency_graph.edges must be a list")
        raw_edges = []
    parents: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    seen_edges: set[tuple[str, str]] = set()
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            add("DEPENDENCY_EDGE_INVALID", f"edge at index {index} must be an object")
            continue
        child = str(edge.get("child_node_id", "")).strip()
        parent = str(edge.get("parent_node_id", "")).strip()
        if not child or not parent:
            add("DEPENDENCY_EDGE_INVALID", f"edge at index {index} needs child_node_id and parent_node_id")
            continue
        if (child, parent) in seen_edges:
            add("DEPENDENCY_EDGE_DUPLICATE", "dependency edge is duplicated", child, (child, parent))
            continue
        seen_edges.add((child, parent))
        if child not in nodes:
            add("DEPENDENCY_CHILD_MISSING", f"edge child {child!r} is absent from nodes", child, (child, parent))
        if parent not in nodes:
            add("DEPENDENCY_PARENT_MISSING", f"parent node {parent!r} is absent from nodes", child, (child, parent))
        if child in parents:
            parents[child].append(parent)

    # Kahn's topological pass detects cycles without recursion-depth limits.
    remaining_parents = {node_id: sum(parent in nodes for parent in node_parents)
                         for node_id, node_parents in parents.items()}
    children_by_parent: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for child, node_parents in parents.items():
        for parent in node_parents:
            if parent in nodes:
                children_by_parent[parent].append(child)
    ready = [node_id for node_id, count in remaining_parents.items() if count == 0]
    removed = 0
    while ready:
        node_id = ready.pop()
        removed += 1
        for child in children_by_parent[node_id]:
            remaining_parents[child] -= 1
            if remaining_parents[child] == 0:
                ready.append(child)
    if removed != len(nodes):
        cyclic_or_blocked = sorted(node_id for node_id, count in remaining_parents.items() if count > 0)
        add("DEPENDENCY_GRAPH_CYCLE",
            "dependency graph is cyclic or depends on a cyclic component: " + ", ".join(cyclic_or_blocked),
            root_node_id, tuple(cyclic_or_blocked))

    if root_node_id not in nodes:
        add("DEPENDENCY_ROOT_MISSING", f"derived input root {root_node_id!r} is absent from nodes",
            root_node_id, (root_node_id,))
        return issues
    if str(nodes[root_node_id].get("node_type", "")).lower() != "derived":
        add("DEPENDENCY_ROOT_NOT_DERIVED", "derived input must reference a derived root", root_node_id,
            (root_node_id,))
    if expected_transformation_id:
        root_transform_id = str(nodes[root_node_id].get("transformation_id", ""))
        if root_transform_id != expected_transformation_id:
            add("DEPENDENCY_ROOT_TRANSFORM_MISMATCH",
                "dependency root transformation_id differs from the input record",
                root_node_id, (root_node_id,))

    registry = graph.get("transformation_registry", {})
    if not isinstance(registry, dict):
        add("DEPENDENCY_TRANSFORM_REGISTRY_INVALID", "transformation_registry must be an object")
        registry = {}
    for node_id, node in nodes.items():
        node_type = str(node.get("node_type", "")).lower()
        if node_type not in {"source", "derived"}:
            add("DEPENDENCY_NODE_TYPE_INVALID", "node_type must be source or derived", node_id, (node_id,))
            continue
        if node_type == "source":
            if parents.get(node_id):
                add("DEPENDENCY_SOURCE_HAS_PARENTS", "source node cannot have parent edges", node_id,
                    (node_id, *parents[node_id]))
            missing = [name for name in ("source_id", "series_id", "vintage_id")
                       if not str(node.get(name, "")).strip()]
            if missing:
                add("DEPENDENCY_SOURCE_IDENTITY_MISSING", "source node missing: " + ", ".join(missing), node_id,
                    (node_id,))
            if not _SHA256_RE.fullmatch(str(node.get("source_sha256", "")).lower()):
                add("DEPENDENCY_SOURCE_HASH_INVALID", "source_sha256 must be 64 hexadecimal characters", node_id,
                    (node_id,))
            if str(node.get("vintage_status", "ACTIVE")).upper() in {"OVERWRITTEN", "SUPERSEDED", "REVOKED"}:
                add("DEPENDENCY_SOURCE_VINTAGE_INVALID", "source version is overwritten or superseded", node_id,
                    (node_id,))
            continue
        if not parents.get(node_id):
            add("DEPENDENCY_DERIVED_WITHOUT_PARENTS", "derived node must have at least one parent", node_id,
                (node_id,))
        transform_id = str(node.get("transformation_id", "")).strip()
        if not transform_id:
            add("DEPENDENCY_TRANSFORM_ID_MISSING", "derived node has no transformation_id", node_id, (node_id,))
            continue
        expected = registry.get(transform_id)
        if not isinstance(expected, dict):
            add("DEPENDENCY_TRANSFORM_REGISTRY_MISSING", f"transformation_id {transform_id!r} is unregistered",
                node_id, (node_id,))
            continue
        parameters = node.get("parameters")
        if not isinstance(parameters, dict):
            add("DEPENDENCY_PARAMETERS_INVALID", "derived-node parameters must be an object", node_id, (node_id,))
            continue
        try:
            actual_parameters_hash = _canonical_json_sha256(parameters)
        except (TypeError, ValueError) as exc:
            add("DEPENDENCY_PARAMETERS_INVALID", f"parameters are not canonical JSON: {exc}", node_id, (node_id,))
            continue
        node_parameters_hash = str(node.get("parameters_sha256", "")).lower()
        registry_parameters_hash = str(expected.get("parameters_sha256", "")).lower()
        if not _SHA256_RE.fullmatch(node_parameters_hash) or not _SHA256_RE.fullmatch(registry_parameters_hash):
            add("DEPENDENCY_PARAMETERS_HASH_INVALID", "parameter hashes must be 64 hexadecimal characters", node_id,
                (node_id,))
        elif actual_parameters_hash != node_parameters_hash or actual_parameters_hash != registry_parameters_hash:
            add("DEPENDENCY_PARAMETERS_HASH_MISMATCH", "parameters do not match the node and trusted registry hashes",
                node_id, (node_id,))
        node_code_hash = str(node.get("code_sha256", "")).lower()
        registry_code_hash = str(expected.get("code_sha256", "")).lower()
        if not _SHA256_RE.fullmatch(node_code_hash) or not _SHA256_RE.fullmatch(registry_code_hash):
            add("DEPENDENCY_CODE_HASH_INVALID", "code hashes must be 64 hexadecimal characters", node_id, (node_id,))
        elif node_code_hash != registry_code_hash:
            add("DEPENDENCY_CODE_HASH_MISMATCH", "node code hash differs from the trusted registry", node_id,
                (node_id,))

    reachable: list[str] = []
    predecessor: dict[str, str | None] = {root_node_id: None}
    pending = [root_node_id]
    while pending:
        node_id = pending.pop()
        reachable.append(node_id)
        for parent in parents.get(node_id, []):
            if parent in nodes and parent not in predecessor:
                predecessor[parent] = node_id
                pending.append(parent)
    def path_to(node_id: str) -> tuple[str, ...]:
        reverse_path = [node_id]
        while predecessor[reverse_path[-1]] is not None:
            reverse_path.append(str(predecessor[reverse_path[-1]]))
        return tuple(reversed(reverse_path))
    origin_idx = _month_index(origin_month)
    for node_id in reachable:
        node, path = nodes[node_id], path_to(node_id)
        ref_month = str(node.get("reference_month", ""))
        try:
            if not ref_month:
                raise ValueError("reference_month is missing")
            if _month_index(ref_month) > origin_idx:
                add("DEPENDENCY_REFERENCE_AFTER_ORIGIN", "ancestor reference month is later than origin month",
                    node_id, path)
        except ValueError as exc:
            add("DEPENDENCY_REFERENCE_MONTH_INVALID", str(exc), node_id, path)
        try:
            available = _aware_datetime(str(node.get("available_at_utc", "")))
            if available > origin_time:
                add("DEPENDENCY_NODE_AVAILABLE_AFTER_ORIGIN", "ancestor was released after the forecast origin",
                    node_id, path)
            elif available == origin_time:
                add("DEPENDENCY_NODE_AVAILABLE_AT_ORIGIN", "ancestor must be released strictly before the origin",
                    node_id, path)
        except (TypeError, ValueError) as exc:
            add("DEPENDENCY_NODE_AVAILABILITY_INVALID", str(exc), node_id, path)
    if expected_reference_month and str(nodes[root_node_id].get("reference_month", "")) != expected_reference_month:
        add("DEPENDENCY_ROOT_REFERENCE_MISMATCH", "dependency root reference month differs from its input row",
            root_node_id, (root_node_id,))
    return issues

def audit_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a target-free report and one eligibility row per planned key."""
    issues: list[dict[str, Any]] = []

    def issue(code: str, severity: str, message: str, key: str = "") -> None:
        issues.append({"code": code, "severity": severity, "key": key, "message": message})

    def forbid_target_values(rows: list[dict[str, Any]], label: str) -> None:
        for i, row in enumerate(rows):
            if any(k.lower() in {"target_value", "observed_target", "actual_target", "label_value"}
                   for k in row):
                issue("TARGET_VALUE_FIELD_FORBIDDEN", "error",
                      f"{label}[{i}] contains a numeric target-value field", str(i))

    origins_list = bundle.get("origins", [])
    inputs = bundle.get("inputs", [])
    planned = bundle.get("planned_forecasts", [])
    forecasts = bundle.get("forecasts", [])
    forbid_target_values(inputs, "inputs")
    forbid_target_values(forecasts, "forecasts")

    origins: dict[str, dict[str, Any]] = {}
    for row in origins_list:
        oid = str(row.get("origin_id", ""))
        if not oid or oid in origins:
            issue("DUPLICATE_OR_INVALID_ORIGIN", "error", "origin_id is blank or duplicated", oid)
            continue
        try:
            _month_index(str(row["origin_month"]))
            origin_time = _aware_datetime(str(row["origin_time_utc"]))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_ORIGIN_CLOCK", "error", str(exc), oid)
            continue
        origins[oid] = {**row, "_origin_time": origin_time}

    requirements = bundle.get("required_features_by_model", {})
    model_names = set(requirements)
    inputs_by_origin: dict[str, list[dict[str, Any]]] = {}
    seen_inputs: set[tuple[str, str, str]] = set()
    for row in inputs:
        oid, feature = str(row.get("origin_id", "")), str(row.get("feature", ""))
        ref_month = str(row.get("reference_month", ""))
        key = (oid, feature, ref_month)
        if key in seen_inputs:
            issue("DUPLICATE_INPUT_KEY", "error", "duplicate origin/feature/reference-month key", "|".join(key))
        seen_inputs.add(key)
        origin = origins.get(oid)
        if origin is None:
            issue("INPUT_ORIGIN_UNKNOWN", "error", "input references an unknown origin", oid)
            continue
        inputs_by_origin.setdefault(oid, []).append(row)
        try:
            if _month_index(ref_month) > _month_index(str(origin["origin_month"])):
                issue("INPUT_REFERENCE_AFTER_ORIGIN", "error",
                      "context reference month is later than the forecast-origin month", "|".join(key))
            masked = bool(row.get("masked", False))
            available_raw = row.get("available_at_utc")
            if available_raw in (None, ""):
                if not masked:
                    raise ValueError("unmasked input has no availability timestamp")
            else:
                available = _aware_datetime(str(available_raw))
                if not masked and available > origin["_origin_time"]:
                    issue("INPUT_AVAILABLE_AFTER_ORIGIN", "error",
                          "unmasked input becomes available after the forecast origin", "|".join(key))
                elif not masked and available == origin["_origin_time"]:
                    issue("INPUT_AVAILABLE_AT_ORIGIN", "error",
                          "unmasked input must become available strictly before the forecast origin", "|".join(key))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_INPUT_CLOCK", "error", str(exc), "|".join(key))
        if row.get("expected_vintage_id") not in (None, "") and str(row.get("vintage_id")) != str(row.get("expected_vintage_id")):
            issue("VINTAGE_ID_MISMATCH", "error", "selected vintage differs from the frozen expected vintage", "|".join(key))
        if str(row.get("vintage_status", "ACTIVE")).upper() in {"OVERWRITTEN", "SUPERSEDED", "REVOKED"}:
            issue("VINTAGE_OVERWRITTEN", "error", "selected vintage is marked overwritten/superseded", "|".join(key))
        if not bool(row.get("masked", False)):
            try:
                if row.get("value") is None or not math.isfinite(float(row["value"])):
                    issue("INVALID_INPUT_VALUE", "error", "unmasked input value is missing or non-finite", "|".join(key))
            except (KeyError, ValueError, TypeError):
                issue("INVALID_INPUT_VALUE", "error", "unmasked input value is missing or non-numeric", "|".join(key))

    invalid_dependency_inputs: set[tuple[str, str, str]] = set()
    dependency_inputs_audited: set[tuple[str, str, str]] = set()
    dependency_graph = bundle.get("dependency_graph")
    for row in inputs:
        if bool(row.get("masked", False)):
            continue
        oid, feature = str(row.get("origin_id", "")), str(row.get("feature", ""))
        ref_month = str(row.get("reference_month", ""))
        input_key = (oid, feature, ref_month)
        root_id = str(row.get("dependency_node_id", "")).strip()
        is_derived = (row.get("is_derived") is True or bool(root_id)
                      or bool(str(row.get("transformation_id", "")).strip()))
        if not is_derived or input_key in dependency_inputs_audited:
            continue
        dependency_inputs_audited.add(input_key)
        origin = origins.get(oid)
        if not root_id:
            invalid_dependency_inputs.add(input_key)
            issue("DEPENDENCY_ROOT_MISSING", "error", "derived input must declare dependency_node_id",
                  "|".join(input_key))
            continue
        if not isinstance(dependency_graph, dict):
            invalid_dependency_inputs.add(input_key)
            issue("DEPENDENCY_GRAPH_MISSING", "error", "derived input requires a complete dependency_graph",
                  "|".join((*input_key, root_id)))
            continue
        if origin is None:
            invalid_dependency_inputs.add(input_key)
            continue
        dep_issues = audit_dependency_graph(
            dependency_graph, root_id, origin["_origin_time"], str(origin["origin_month"]), ref_month,
            str(row.get("transformation_id", "")).strip() or None)
        if dep_issues:
            invalid_dependency_inputs.add(input_key)
        for dep_issue in dep_issues:
            issue(dep_issue["code"], "error", dep_issue["message"],
                  "|".join((*input_key, dep_issue.get("key", ""), dep_issue.get("path", ""))))

    planned_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in planned:
        try:
            model, oid = str(row["model"]), str(row["origin_id"])
            h = int(row["horizon_months"])
            target_month = str(row["target_month"])
            key = (model, oid, h)
            if key in planned_by_key:
                issue("DUPLICATE_PLANNED_KEY", "error", "planned model/origin/horizon key is duplicated", repr(key))
            planned_by_key[key] = row
            if model not in model_names:
                issue("MODEL_REQUIREMENTS_UNDECLARED", "error",
                      "planned model has no declared input requirements", model)
            if oid not in origins:
                issue("PLANNED_ORIGIN_UNKNOWN", "error", "planned forecast references an unknown origin", repr(key))
            elif target_month != add_month(str(origins[oid]["origin_month"]), h):
                issue("TARGET_HORIZON_MISMATCH", "error", "target month does not equal origin month plus horizon", repr(key))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_PLANNED_KEY", "error", str(exc), str(row))

    forecast_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in forecasts:
        try:
            key = (str(row["model"]), str(row["origin_id"]), int(row["horizon_months"]))
        except (KeyError, ValueError, TypeError) as exc:
            issue("INVALID_FORECAST_KEY", "error", str(exc), str(row))
            continue
        if key in forecast_by_key:
            issue("DUPLICATE_FORECAST_KEY", "error", "forecast model/origin/horizon key is duplicated", repr(key))
        forecast_by_key[key] = row
        if key not in planned_by_key:
            issue("UNPLANNED_FORECAST", "error", "forecast has no matching planned key", repr(key))
        elif str(row.get("target_month", "")) != str(planned_by_key[key].get("target_month", "")):
            issue("FORECAST_TARGET_MISMATCH", "error", "forecast target month differs from its planned key", repr(key))
        if str(row.get("status", "")).upper() == "FORECAST_COMPLETE":
            try:
                if not math.isfinite(float(row["prediction"])):
                    raise ValueError("prediction is non-finite")
            except (KeyError, ValueError, TypeError) as exc:
                issue("INVALID_FORECAST_VALUE", "error", str(exc), repr(key))
        elif str(row.get("status", "")).upper() in {"FIT_FAILED", "PREDICTION_FAILED"}:
            issue("EXPLICIT_FIT_OR_PREDICTION_FAILURE", "info",
                  "failed forecast is retained with an explicit status; it is not a zero error", repr(key))
        elif str(row.get("status", "")).upper() not in {"INPUT_INELIGIBLE"}:
            issue("INVALID_FORECAST_STATUS", "error", "forecast status is missing or unrecognized", repr(key))

    for key in planned_by_key:
        if key not in forecast_by_key:
            issue("MISSING_FORECAST_ROW", "error", "planned forecast key has no corresponding output/status row", repr(key))

    # Optional, frozen origin-set contract keyed as "model|horizon".
    expected_sets = bundle.get("expected_origin_ids_by_model_horizon", {})
    for label, expected in expected_sets.items():
        try:
            model, h_text = label.rsplit("|", 1)
            h = int(h_text)
            actual = {k[1] for k in forecast_by_key if k[0] == model and k[2] == h}
            if actual != set(map(str, expected)):
                issue("MODEL_ORIGIN_SET_MISMATCH", "error",
                      f"forecast origins differ from frozen expected set (expected {len(expected)}, found {len(actual)})", label)
        except (ValueError, TypeError):
            issue("INVALID_EXPECTED_ORIGIN_SET", "error", "expected-origin-set key must be model|horizon", str(label))

    eligibility_rows: list[dict[str, Any]] = []
    for key, plan in sorted(planned_by_key.items()):
        model, oid, h = key
        required = list(requirements.get(model, []))
        available_features = {
            str(row.get("feature")) for row in inputs_by_origin.get(oid, [])
            if not bool(row.get("masked", False))
            and _finite_value(row.get("value"))
            and str(row.get("vintage_status", "ACTIVE")).upper() not in {"OVERWRITTEN", "SUPERSEDED", "REVOKED"}
            and (row.get("expected_vintage_id") in (None, "") or str(row.get("vintage_id")) == str(row.get("expected_vintage_id")))
            and (oid, str(row.get("feature")), str(row.get("reference_month", ""))) not in invalid_dependency_inputs
            and oid in origins
            and _input_time_valid(row, origins[oid])
        }
        missing = [feature for feature in required if feature not in available_features]
        forecast = forecast_by_key.get(key, {})
        status = str(forecast.get("status", "MISSING")).upper()
        reasons = []
        if model not in model_names:
            reasons.append("MODEL_REQUIREMENTS_UNDECLARED")
        if missing:
            reasons.append("REQUIRED_INPUT_MISSING_OR_MASKED:" + ",".join(missing))
        if status in {"FIT_FAILED", "PREDICTION_FAILED"}:
            reasons.append(status)
        if status == "MISSING":
            reasons.append("MISSING_FORECAST_ROW")
        complete = status == "FORECAST_COMPLETE" and _finite_value(forecast.get("prediction"))
        if status == "FORECAST_COMPLETE" and not complete:
            reasons.append("INVALID_FORECAST_VALUE")
        input_eligible = model in model_names and not missing
        eligible = input_eligible and complete
        eligibility_rows.append({
            "model": model, "origin_id": oid, "origin_month": origins.get(oid, {}).get("origin_month", ""),
            "horizon_months": h, "target_month": plan.get("target_month", ""),
            "input_eligible": input_eligible, "forecast_status": status,
            "forecast_complete": complete, "evaluation_eligible": eligible,
            "reason_codes": ";".join(reasons),
        })

    errors = sum(i["severity"] == "error" for i in issues)
    report = {
        "schema_version": "1.0",
        "status": "FAIL" if errors else "PASS",
        "target_values_read_or_required": False,
        "policy": {
            "input_time_rule": "unmasked availability strictly before origin",
            "masked_input_rule": "masked inputs do not satisfy required features; availability may be late or unknown",
            "vintage_rule": "checks supplied selected and expected vintage identifiers; does not enumerate vintage histories",
            "dependency_rule": "unmasked derived inputs require a root in a complete dependency_graph; all ancestors are traversed and checked against the supplied origin and transformation registries",
            "feature_completeness": "at least one eligible record per required feature; graph closure is conditional on complete supplied edges and registries",
        },
        "counts": {
            "origins": len(origins_list), "input_rows": len(inputs),
            "planned_forecast_keys": len(planned_by_key), "forecast_rows": len(forecasts),
            "derived_input_roots_audited": len(dependency_inputs_audited),
            "evaluation_eligible_keys": sum(bool(r["evaluation_eligible"]) for r in eligibility_rows),
            "issues_error": errors,
            "issues_info": sum(i["severity"] == "info" for i in issues),
        },
        "issues": issues,
        "limitations": [
            "Dependency closure is conditional on a complete edge list and complete source/transformation registries; omitted nodes or edges cannot be inferred.",
            "Publication timestamps, source-version registries, transformation registries, and recorded code hashes are checked as supplied and are not independently authenticated.",
            "The checker validates provenance metadata and parameter/code identity links; it does not replay transformations to prove numerical output correctness.",
            "Checks only the selected-vintage records supplied in the bundle; it cannot establish a complete historical vintage census.",
            "The audit is target-free and does not compute forecast errors or scores.",
        ],
    }
    return report, eligibility_rows


def _input_time_valid(row: dict[str, Any], origin: dict[str, Any]) -> bool:
    try:
        return _aware_datetime(str(row["available_at_utc"])) < origin["_origin_time"] and _month_index(str(row["reference_month"])) <= _month_index(str(origin["origin_month"]))
    except (KeyError, ValueError, TypeError):
        return False


def write_outputs(report: dict[str, Any], eligibility: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "audit_report.json"
    eligibility_path = out_dir / "eligibility.csv"
    existing = [p.name for p in (report_path, eligibility_path) if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite audit outputs: " + ", ".join(existing))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(eligibility[0]) if eligibility else ["model", "origin_id", "origin_month", "horizon_months", "target_month", "input_eligible", "forecast_status", "forecast_complete", "evaluation_eligible", "reason_codes"]
    with eligibility_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eligibility)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="target-free PIT contract JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="new output directory (never overwritten)")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    report, eligibility = audit_bundle(bundle)
    write_outputs(report, eligibility, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

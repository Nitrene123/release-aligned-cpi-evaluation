"""DAG contract tests, including the 12 original S22 transitive misses."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
MODULE_CANDIDATES = [HERE.parents[1] / "src" / "rae" / "audit_point_in_time.py",
                     HERE.with_name("audit_point_in_time.py")]
MODULE_PATH = next(path for path in MODULE_CANDIDATES if path.exists())
SPEC = importlib.util.spec_from_file_location("audit_point_in_time", MODULE_PATH)
pit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(pit)


def _hash_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _node(node_id, node_type, available, reference_month, **extra):
    return {"node_id": node_id, "node_type": node_type, "available_at_utc": available,
            "reference_month": reference_month, **extra}


def valid_graph(source_available="2024-01-15T00:00:00Z"):
    code_hash = _hash_text("fixture transform code v1")
    nodes = [
        _node("root", "derived", "2024-02-01T00:00:00Z", "2023-12",
              feature="CPI", transformation_id="root-run", parameters={"window": 3},
              parameters_sha256=_hash_json({"window": 3}), code_sha256=code_hash),
        _node("middle", "derived", "2024-01-30T00:00:00Z", "2023-12",
              transformation_id="middle-run", parameters={"scale": 1.0},
              parameters_sha256=_hash_json({"scale": 1.0}), code_sha256=code_hash),
        _node("source", "source", source_available, "2023-12", source_id="BLS",
              series_id="CPI", vintage_id="v1", source_sha256="a" * 64, vintage_status="ACTIVE"),
    ]
    transforms = {n["transformation_id"]: {"parameters_sha256": n["parameters_sha256"],
                                             "code_sha256": code_hash}
                  for n in nodes if n["node_type"] == "derived"}
    return {"nodes": nodes,
            "edges": [{"child_node_id": "root", "parent_node_id": "middle"},
                      {"child_node_id": "middle", "parent_node_id": "source"}],
            "transformation_registry": transforms}


def base_bundle(graph=None):
    bundle = {
        "schema_version": "1.0",
        "origins": [{"origin_id": "O1", "origin_month": "2024-01",
                     "origin_time_utc": "2024-02-10T01:30:00Z"}],
        "required_features_by_model": {"M": ["CPI"]},
        "inputs": [{"origin_id": "O1", "feature": "CPI", "reference_month": "2023-12",
                    "available_at_utc": "2024-01-15T00:00:00Z", "vintage_id": "v1",
                    "expected_vintage_id": "v1", "vintage_status": "ACTIVE", "masked": False,
                    "value": 0.4}],
        "planned_forecasts": [{"model": "M", "origin_id": "O1", "horizon_months": 1,
                               "target_month": "2024-02"}],
        "forecasts": [{"model": "M", "origin_id": "O1", "horizon_months": 1,
                       "target_month": "2024-02", "status": "FORECAST_COMPLETE", "prediction": 0.5}],
        "expected_origin_ids_by_model_horizon": {"M|1": ["O1"]},
    }
    if graph is not None:
        bundle["dependency_graph"] = graph
        bundle["inputs"][0].update(is_derived=True, dependency_node_id="root")
    return bundle


def _case_root() -> Path:
    for ancestor in HERE.parents:
        candidates = [ancestor / "experiments" / "rae_heldout_fault_generalization" / "cases",
                      ancestor / "rae_heldout_fault_generalization" / "cases"]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("S22 held-out case directory is not present")


def _s22_case_bundle(case_id: str):
    case_dir = _case_root() / case_id
    with (case_dir / "origins.csv").open(encoding="utf-8-sig", newline="") as f:
        origin_row = next(r for r in csv.DictReader(f) if r["model"] == "BASE")
    origin_time = pit._aware_datetime(origin_row["origin_timestamp"])
    with (case_dir / "dependency_edges.csv").open(encoding="utf-8-sig", newline="") as f:
        edges = [{"child_node_id": r["child_node"], "parent_node_id": r["parent_node"]}
                 for r in csv.DictReader(f)]
    node_ids = {e[k] for e in edges for k in ("child_node_id", "parent_node_id")}
    parent_ids = {e["parent_node_id"] for e in edges}
    derived_ids = node_ids & {e["child_node_id"] for e in edges}
    source_ids = node_ids - derived_ids
    code_hash = _hash_text("S22 regression fixture transform code v1")
    nodes, registry = [], {}
    for node_id in sorted(node_ids):
        ref_month = node_id.rsplit("@", 1)[1]
        if node_id in source_ids:
            nodes.append(_node(node_id, "source", (origin_time + timedelta(microseconds=1)).isoformat(),
                               ref_month, source_id="SYNTH", series_id=node_id.split("@", 1)[0],
                               vintage_id="injected-future-v1", source_sha256=_hash_text(case_id + node_id),
                               vintage_status="ACTIVE"))
        else:
            transform_id = "s22:" + case_id + ":" + node_id
            params = {"case_id": case_id, "node_id": node_id, "operation": "derived-index fixture"}
            params_hash = _hash_json(params)
            nodes.append(_node(node_id, "derived", (origin_time - timedelta(minutes=1)).isoformat(),
                               ref_month, feature="CPI", transformation_id=transform_id, parameters=params,
                               parameters_sha256=params_hash, code_sha256=code_hash))
            registry[transform_id] = {"parameters_sha256": params_hash, "code_sha256": code_hash}
    root_id = next(node_id for node_id in derived_ids if node_id not in parent_ids)
    b = base_bundle({"nodes": nodes, "edges": edges, "transformation_registry": registry})
    b["origins"][0].update(origin_month=origin_row["origin_month"], origin_time_utc=origin_row["origin_timestamp"])
    b["inputs"][0]["available_at_utc"] = (origin_time - timedelta(minutes=5)).isoformat()
    b["inputs"][0].update(reference_month=root_id.rsplit("@", 1)[1], is_derived=True,
                           dependency_node_id=root_id)
    b["planned_forecasts"][0]["target_month"] = pit.add_month(origin_row["origin_month"], 1)
    b["forecasts"][0]["target_month"] = b["planned_forecasts"][0]["target_month"]
    return b


class DependencyGraphTests(unittest.TestCase):
    def test_clean_multilevel_dependency_graph_passes(self):
        report, rows = pit.audit_bundle(base_bundle(valid_graph()))
        self.assertEqual(report["status"], "PASS", report["issues"])
        self.assertEqual(report["counts"]["derived_input_roots_audited"], 1)
        self.assertTrue(rows[0]["evaluation_eligible"])

    def test_future_source_at_deep_ancestor_is_rejected(self):
        b = base_bundle(valid_graph("2024-02-11T00:00:00Z"))
        report, rows = pit.audit_bundle(b)
        issue = next(x for x in report["issues"] if x["code"] == "DEPENDENCY_NODE_AVAILABLE_AFTER_ORIGIN")
        self.assertIn("root -> middle -> source", issue["key"])
        self.assertFalse(rows[0]["input_eligible"])

    def test_cycle_is_rejected(self):
        g = valid_graph(); g["edges"].append({"child_node_id": "source", "parent_node_id": "root"})
        codes = {x["code"] for x in pit.audit_dependency_graph(g, "root", "2024-02-10T01:30:00Z", "2024-01")}
        self.assertIn("DEPENDENCY_GRAPH_CYCLE", codes)

    def test_missing_parent_is_rejected(self):
        g = valid_graph(); g["edges"].append({"child_node_id": "middle", "parent_node_id": "absent"})
        codes = {x["code"] for x in pit.audit_dependency_graph(g, "root", "2024-02-10T01:30:00Z", "2024-01")}
        self.assertIn("DEPENDENCY_PARENT_MISSING", codes)

    def test_unregistered_transformation_is_rejected(self):
        g = valid_graph(); g["nodes"][0]["transformation_id"] = "unregistered"
        codes = {x["code"] for x in pit.audit_dependency_graph(g, "root", "2024-02-10T01:30:00Z", "2024-01")}
        self.assertIn("DEPENDENCY_TRANSFORM_REGISTRY_MISSING", codes)

    def test_parameter_hash_mismatch_is_rejected(self):
        g = valid_graph(); g["nodes"][0]["parameters"] = {"window": 4}
        codes = {x["code"] for x in pit.audit_dependency_graph(g, "root", "2024-02-10T01:30:00Z", "2024-01")}
        self.assertIn("DEPENDENCY_PARAMETERS_HASH_MISMATCH", codes)

    def test_code_hash_mismatch_is_rejected(self):
        g = valid_graph(); g["nodes"][0]["code_sha256"] = "b" * 64
        codes = {x["code"] for x in pit.audit_dependency_graph(g, "root", "2024-02-10T01:30:00Z", "2024-01")}
        self.assertIn("DEPENDENCY_CODE_HASH_MISMATCH", codes)

    def test_input_transformation_id_must_match_graph_root(self):
        b = base_bundle(valid_graph())
        b["inputs"][0]["transformation_id"] = "different-run"
        report, rows = pit.audit_bundle(b)
        self.assertIn("DEPENDENCY_ROOT_TRANSFORM_MISMATCH", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])

    def test_naive_origin_datetime_is_rejected(self):
        g = valid_graph()
        issues = pit.audit_dependency_graph(g, "root", datetime(2024, 2, 10, 1, 30), "2024-01")
        self.assertIn("DEPENDENCY_ORIGIN_CLOCK_INVALID", {x["code"] for x in issues})

    def test_source_at_origin_is_rejected(self):
        g = valid_graph("2024-02-10T01:30:00Z")
        codes = {x["code"] for x in pit.audit_dependency_graph(g, "root", "2024-02-10T01:30:00Z", "2024-01")}
        self.assertIn("DEPENDENCY_NODE_AVAILABLE_AT_ORIGIN", codes)

    def test_missing_dependency_graph_makes_derived_input_ineligible(self):
        b = base_bundle(); b["inputs"][0].update(is_derived=True, dependency_node_id="root")
        report, rows = pit.audit_bundle(b)
        self.assertIn("DEPENDENCY_GRAPH_MISSING", {x["code"] for x in report["issues"]})
        self.assertFalse(rows[0]["input_eligible"])

    def test_deep_graph_does_not_hit_python_recursion_limit(self):
        n = 1100
        code_hash = _hash_text("deep graph code")
        params = {"depth": 1}; param_hash = _hash_json(params)
        nodes, edges, registry = [], [], {}
        for i in range(n):
            node_id = f"d{i}"
            transform_id = f"t{i}"
            nodes.append(_node(node_id, "derived", "2024-02-01T00:00:00Z", "2023-12",
                               transformation_id=transform_id, parameters=params,
                               parameters_sha256=param_hash, code_sha256=code_hash))
            registry[transform_id] = {"parameters_sha256": param_hash, "code_sha256": code_hash}
            if i + 1 < n:
                edges.append({"child_node_id": node_id, "parent_node_id": f"d{i+1}"})
        source = _node("source", "source", "2024-02-11T00:00:00Z", "2023-12", source_id="S",
                       series_id="X", vintage_id="v1", source_sha256="c" * 64)
        nodes.append(source); edges.append({"child_node_id": f"d{n-1}", "parent_node_id": "source"})
        issues = pit.audit_dependency_graph({"nodes": nodes, "edges": edges, "transformation_registry": registry},
                                            "d0", "2024-02-10T01:30:00Z", "2024-01")
        self.assertIn("DEPENDENCY_NODE_AVAILABLE_AFTER_ORIGIN", {x["code"] for x in issues})


def _make_s22_regression(case_id):
    def test(self):
        report, rows = pit.audit_bundle(_s22_case_bundle(case_id))
        matches = [x for x in report["issues"] if x["code"] == "DEPENDENCY_NODE_AVAILABLE_AFTER_ORIGIN"]
        self.assertEqual(report["status"], "FAIL", case_id)
        self.assertTrue(matches, case_id)
        self.assertTrue(any("revision@2024-01" in x["key"] for x in matches), case_id)
        self.assertFalse(rows[0]["evaluation_eligible"], case_id)
    return test


for _case_id in ("H0100", "H0102", "H0104", "H0106", "H0108", "H0110",
                 "H0112", "H0114", "H0116", "H0118", "H0120", "H0122"):
    setattr(DependencyGraphTests, f"test_s22_{_case_id}_future_parent_regression",
            _make_s22_regression(_case_id))

if __name__ == "__main__":
    unittest.main(verbosity=2)

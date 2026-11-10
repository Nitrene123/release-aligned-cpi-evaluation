#!/usr/bin/env python3
"""Frozen, held-out Feast-vs-RAE injection benchmark; all artifacts stay under tmp."""
from __future__ import annotations
import csv, hashlib, itertools, json, math, random, re, sys, time, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("RAE_BENCH_ROOT", Path(__file__).resolve().parent))
BASE = ROOT / "baseline_source"
CASE_ROOT = ROOT / "cases"
RESULTS = ROOT / "results"
FEAST_ROOT = ROOT / "feast_comparison"
AUDITOR_SRC = BASE / "pit_audit.py"
SEED = 2026111017
MODELS = ("AR1", "B3", "TFM3")
FEATURES = ("cpi_yoy", "ppi_yoy", "fx_return")
HORIZONS = (1, 3, 6, 12)
SIMPLE_FAULTS = ("late_release", "stale_vintage", "future_transform_dependency",
                 "target_mapping", "model_sample_mismatch", "boundary_equal")
FAULT_GROUPS = {
    "release_cutoff": {"late_release", "boundary_equal"},
    "stale_vintage": {"stale_vintage"},
    "transform_dependency": {"future_transform_dependency"},
    "target_mapping": {"target_mapping"},
    "model_sample_coverage": {"model_sample_mismatch"},
}
ABLATIONS = ("release", "vintage", "dependency", "eligibility", "mapping")
N_CLEAN, N_PER_FAULT, N_MULTI = 100, 100, 100


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def next_month(ym: str, steps: int = 1) -> str:
    y, m = map(int, ym.split("-"))
    z = y * 12 + m - 1 + steps
    return f"{z // 12:04d}-{z % 12 + 1:02d}"


def month_start(ym: str) -> str:
    return f"{ym}-01T00:00:00+00:00"


def month_number(ym: str) -> int:
    y, m = map(int, ym.split("-"))
    return y * 12 + m - 1


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_hash(rows: list[dict[str, str]]) -> str:
    ordered = sorted(rows, key=lambda r: (r["feature"], r["observation_month"], r.get("source_observation_month", "")))
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def base_bundle(case_id: str, rep: int, seed: int) -> tuple[Path, dict]:
    rng = random.Random(seed + rep * 7919)
    folder = CASE_ROOT / case_id
    folder.mkdir(parents=True, exist_ok=False)
    origins, inputs, vintages, forecasts, targets = [], [], [], [], []
    origin_months = [next_month("2020-01", i) for i in range(6)]
    origin_lookup = {}
    for oi, om in enumerate(origin_months):
        clock_month = next_month(om)
        clock = f"{clock_month}-10T01:31:00+00:00"
        oid = f"O{oi + 1:02d}"
        origin_lookup[oid] = {"month": om, "clock": clock}
        for model in MODELS:
            origins.append({"model": model, "origin_id": oid, "origin_month": om, "origin_timestamp": clock})
        context = next_month(om, -1)
        available = (datetime.fromisoformat(clock) - timedelta(seconds=5)).isoformat()
        for fi, feature in enumerate(FEATURES):
            val = round(0.5 + 0.37 * fi + 0.08 * oi + rng.uniform(-0.03, 0.03), 6)
            source_sha = sha(f"{case_id}|{feature}|{context}|v1|{val}")
            vintages.append({"source_id":"SYN", "series_id":feature, "observation_month":context,
                             "vintage_id":"v1", "available_at":available, "value":str(val), "source_sha256":source_sha})
            for model in MODELS:
                inputs.append({"model":model, "origin_id":oid, "feature":feature,
                    "observation_month":context, "source_observation_month":context,
                    "value":str(val), "masked":"false", "mask_reason":"", "source_id":"SYN",
                    "series_id":feature, "vintage_id":"v1", "available_at":available,
                    "source_sha256":source_sha})
    target_release_by_month = {}
    for om in origin_months:
        # Add every intervening month so off-by-one target mappings remain resolvable.
        for h in range(1, 13):
            tm = next_month(om, h)
            rel = f"{next_month(tm)}-10T01:30:00+00:00"
            target_release_by_month.setdefault(tm, rel)
    targets = [{"target_month":tm, "first_release_at":rel}
               for tm,rel in sorted(target_release_by_month.items())]
    for model in MODELS:
        for oid, spec in origin_lookup.items():
            panel = [r for r in inputs if r["model"] == model and r["origin_id"] == oid]
            ih = canonical_hash(panel)
            for h in HORIZONS:
                tm = next_month(spec["month"], h)
                forecasts.append({"model":model, "origin_id":oid, "horizon_months":str(h),
                    "target_month":tm, "prediction":str(round(rng.uniform(-1, 3), 6)), "input_hash":ih})
    for name, rows in (("origins.csv", origins), ("inputs.csv", inputs), ("vintages.csv", vintages),
                       ("targets.csv", targets), ("forecasts.csv", forecasts)):
        write_csv(folder / name, rows)
    cfg = {"origins_file":"origins.csv", "inputs_file":"inputs.csv", "vintages_file":"vintages.csv",
           "targets_file":"targets.csv", "forecasts_file":"forecasts.csv", "expected_models":list(MODELS),
           "expected_horizons":list(HORIZONS), "require_identical_model_origin_sets":True,
           "required_features_by_model":{m:list(FEATURES) for m in MODELS}}
    (folder / "bundle.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return folder, origin_lookup


def mutate_fault(folder: Path, fault: str, rep: int, offset: int, origin_lookup: dict) -> dict:
    """Independent mutation; returns the truth key for this injection."""
    inputs = read_csv(folder / "inputs.csv")
    vintages = read_csv(folder / "vintages.csv")
    forecasts = read_csv(folder / "forecasts.csv")
    oid = f"O{((rep * 7 + offset * 3) % 6) + 1:02d}"
    model = MODELS[(rep + offset) % len(MODELS)]
    feature = FEATURES[(rep + offset * 2) % len(FEATURES)]
    context_row = next(r for r in inputs if r["origin_id"] == oid and r["model"] == model and r["feature"] == feature)
    origin_at = datetime.fromisoformat(origin_lookup[oid]["clock"])
    target = {"fault":fault, "origin_id":oid, "model":model, "feature":feature, "horizon":None}
    cell = lambda r: r["origin_id"] == oid and r["model"] == model and r["feature"] == feature
    if fault == "late_release":
        late = (origin_at + timedelta(seconds=30)).isoformat()
        for r in inputs:
            if cell(r): r["available_at"] = late
        for v in vintages:
            if v["series_id"] == feature and v["observation_month"] == context_row["source_observation_month"]:
                v["available_at"] = late
    elif fault == "boundary_equal":
        exact = origin_at.isoformat()
        for r in inputs:
            if cell(r): r["available_at"] = exact
        for v in vintages:
            if v["series_id"] == feature and v["observation_month"] == context_row["source_observation_month"]:
                v["available_at"] = exact
    elif fault == "stale_vintage":
        # v2 is available before the origin, but the input incorrectly selects v1.
        # This isolates vintage selection from release-time violations.
        newer = (origin_at - timedelta(seconds=1)).isoformat()
        source_month = context_row["source_observation_month"]
        newval = round(float(context_row["value"]) + 8.0, 6)
        newsha = sha(f"{folder.name}|{feature}|{source_month}|v2|{newval}")
        vintages.append({"source_id":"SYN", "series_id":feature, "observation_month":source_month,
            "vintage_id":"v2", "available_at":newer, "value":str(newval), "source_sha256":newsha})
    elif fault == "future_transform_dependency":
        future = next_month(context_row["observation_month"])
        # The value is available by the outer origin, but its declared source month is later
        # than the historical context row to which the transformed feature is attached.
        avail = (origin_at - timedelta(seconds=5)).isoformat()
        newval = round(float(context_row["value"]) + 1.25, 6)
        newsha = sha(f"{folder.name}|{feature}|{future}|v1|future-dependency|{newval}")
        vintages.append({"source_id":"SYN_FUTURE", "series_id":feature, "observation_month":future,
            "vintage_id":"v1", "available_at":avail, "value":str(newval), "source_sha256":newsha})
        for r in inputs:
            if r["origin_id"] == oid and r["model"] == model and r["feature"] == feature:
                r.update(source_id="SYN_FUTURE",source_observation_month=future,value=str(newval),available_at=avail,source_sha256=newsha)
    elif fault == "target_mapping":
        h = HORIZONS[(rep + offset) % len(HORIZONS)]
        target["horizon"] = h
        r = next(r for r in forecasts if r["model"] == model and r["origin_id"] == oid and int(r["horizon_months"]) == h)
        r["target_month"] = next_month(r["target_month"])
    elif fault == "model_sample_mismatch":
        h = HORIZONS[(rep + offset) % len(HORIZONS)]
        target["horizon"] = h
        forecasts = [r for r in forecasts if not (r["model"] == model and r["origin_id"] == oid and int(r["horizon_months"]) == h)]
    else:
        raise ValueError(fault)
    for name, rows in (("inputs.csv", inputs), ("vintages.csv", vintages), ("forecasts.csv", forecasts)):
        write_csv(folder / name, rows)
    # Input hashes reflect the mutated payload; this benchmark is about semantic checks.
    by_key = {}
    for r in inputs: by_key.setdefault((r["model"], r["origin_id"]), []).append(r)
    for f in forecasts: f["input_hash"] = canonical_hash(by_key[(f["model"], f["origin_id"])])
    write_csv(folder / "forecasts.csv", forecasts)
    return target


def copy_auditor(source: Path, dest: Path) -> None:
    """Use the actual module-gated auditor without wrapping its diagnostics."""
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def structural_categories(folder: Path) -> set[str]:
    origins, forecasts = read_csv(folder/"origins.csv"), read_csv(folder/"forecasts.csv")
    categories=set()
    if len({(r["model"],r["origin_id"]) for r in origins}) != len(origins): categories.add("integrity")
    if len({(r["model"],r["origin_id"],r["horizon_months"]) for r in forecasts}) != len(forecasts): categories.add("integrity")
    models = sorted({r["model"] for r in origins})
    om = {(r["model"],r["origin_id"]):r["origin_month"] for r in origins}
    for h in HORIZONS:
        sets = {m:{r["origin_id"] for r in forecasts if r["model"]==m and int(r["horizon_months"])==h} for m in models}
        if len({frozenset(x) for x in sets.values()}) > 1: categories.add("model_sample_coverage")
    for r in forecasts:
        if month_number(r["target_month"]) - month_number(om[(r["model"],r["origin_id"])]) != int(r["horizon_months"]): categories.add("target_mapping")
    return categories


def structural_detect(folder: Path) -> bool:
    return bool(structural_categories(folder))


def run_rae(auditor_path: Path, folder: Path, output_path: Path, enabled: list[str] | None = None) -> tuple[dict, float]:
    cfgpath=folder/"bundle.json"
    if str(auditor_path.parent) not in sys.path: sys.path.insert(0,str(auditor_path.parent))
    import importlib.util
    spec=importlib.util.spec_from_file_location("rae_audited",auditor_path)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    t0=time.perf_counter(); report=mod.run_audit(cfgpath,output_path,enabled_modules=enabled); elapsed=time.perf_counter()-t0
    return report,elapsed


def fault_classes(errors: list[str]) -> set[str]:
    from importlib.util import spec_from_file_location, module_from_spec
    # The same taxonomy is implemented independently of the injected truth oracle.
    s=" | ".join(errors).lower(); out=set()
    if "source observation month is after input context month" in s: out.add("dependency")
    if "target-origin-horizon mismatch" in s or "forecast target absent from target calendar" in s: out.add("mapping")
    if any(x in s for x in ("model origin sets differ", "origin model set mismatch", "input references unknown model-origin", "forecast references unknown model-origin", "forecast emitted despite missing required input rows", "origin timestamps differ between models")): out.add("eligibility")
    if any(x in s for x in ("available_at is not strictly before origin", "selected vintage was not available before origin", "target was released by origin", "target release is not after derived origin")): out.add("release")
    if any(x in s for x in ("stale vintage", "input value/hash differs from selected vintage", "selected input vintage absent", "ambiguous latest vintage")): out.add("vintage")
    return out


def rae_traceability(folder: Path, audit_path: Path, affected: list[dict]) -> list[dict]:
    report=json.loads(audit_path.read_text(encoding="utf-8")); errors=report.get("errors",[])
    out=[]
    for item in affected:
        fault=item["fault"]; linked=False; evidence=""
        if fault in {"late_release","boundary_equal","stale_vintage","future_transform_dependency"}:
            rows=read_csv(folder/"inputs.csv")
            target_rows=[(i,r) for i,r in enumerate(rows,start=2) if r["model"]==item["model"] and r["origin_id"]==item["origin_id"] and r["feature"]==item["feature"]]
            for i,_ in target_rows:
                hits=[e for e in errors if f"row {i}" in e and ("input" in e.lower() or "observation month" in e.lower() or "vintage" in e.lower())]
                if hits: linked=True; evidence=hits[0]; break
        elif fault=="target_mapping":
            rows=read_csv(folder/"forecasts.csv")
            target_rows=[(i,r) for i,r in enumerate(rows,start=2) if r["model"]==item["model"] and r["origin_id"]==item["origin_id"] and int(r["horizon_months"])==int(item["horizon"])]
            for i,_ in target_rows:
                hits=[e for e in errors if f"row {i}" in e and ("target-origin-horizon" in e or "forecast target" in e)]
                if hits: linked=True; evidence=hits[0]; break
        elif fault=="model_sample_mismatch":
            eligpath=audit_path.parent/report["eligibility_table_file"]
            rows=read_csv(eligpath)
            linked=any(r["model"]==item["model"] and r["origin_id"]==item["origin_id"] and int(r["horizon_months"])==int(item["horizon"]) and r["forecast_record_status"]=="NO_OUTPUT_ROW" for r in rows)
            evidence="eligibility row marks the injected model-origin-horizon forecast as NO_OUTPUT_ROW" if linked else ""
        out.append({"fault":fault,"model":item["model"],"origin_id":item["origin_id"],"feature":item["feature"],
                    "horizon":item.get("horizon"),"record_linked":linked,"evidence":evidence})
    return out


def main() -> None:
    CASE_ROOT.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    for sub in ("rae_full", "rae_ablation", "feast_comparison"):
        (RESULTS/sub).mkdir(parents=True,exist_ok=True)
    rng=random.Random(SEED)
    cases=[]
    cases += [("clean", []) for _ in range(N_CLEAN)]
    for fault in SIMPLE_FAULTS: cases += [(fault,[fault]) for _ in range(N_PER_FAULT)]
    pairs=list(itertools.combinations(SIMPLE_FAULTS,2))
    for i in range(N_MULTI): cases.append(("multi_"+"+".join(pairs[i%len(pairs)]), list(pairs[i%len(pairs)])))
    limit = int(os.environ.get("RAE_CASE_LIMIT", "0"))
    if limit:
        smoke = [cases[0]]
        smoke += [next(case for case in cases if case[0] == fault) for fault in SIMPLE_FAULTS]
        smoke += [next(case for case in cases if case[0].startswith("multi_"))]
        cases = smoke[:limit]
    rng.shuffle(cases)
    truth=[]; metadata=[]
    for i,(case_type,faults) in enumerate(cases):
        case_id=f"C{i+1:04d}"
        folder,origin_lookup=base_bundle(case_id,i,SEED)
        affected=[]
        for fi,fault in enumerate(faults): affected.append(mutate_fault(folder,fault,i,fi,origin_lookup))
        truth.append({"case_id":case_id,"case_type":case_type,"expected_faults":"|".join(faults) if faults else "none",
                      "affected_keys_json":json.dumps(affected,separators=(",",":")),"generator_seed":SEED})
        metadata.append({"case_id":case_id,"folder":folder,"faults":faults,"affected":affected,"origin_lookup":origin_lookup})
    write_csv(RESULTS/"independent_truth_800.csv",truth)
    fixture_manifest=[]
    for item in metadata:
        for f in sorted(item["folder"].iterdir()):
            if f.is_file():
                fixture_manifest.append({"case_id":item["case_id"],"path":f"cases/{item['case_id']}/{f.name}",
                                        "sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size})
    write_csv(RESULTS/"case_fixture_manifest.csv",fixture_manifest)
    (RESULTS/"generation_metadata.json").write_text(json.dumps({"seed":SEED,"N_clean":N_CLEAN,"N_per_simple_fault":N_PER_FAULT,
        "simple_faults":list(SIMPLE_FAULTS),"N_multi":N_MULTI,"N_total":len(cases),
        "realized_clean":sum(not faults for _,faults in cases),
        "realized_fault_cases":sum(bool(faults) and len(faults)==1 for _,faults in cases),
        "realized_multi_fault":sum(len(faults)>1 for _,faults in cases),
        "pair_types":[list(p) for p in pairs]},indent=2),encoding="utf-8")
    # Create an isolated copy of the audited RAE implementation with diagnostic-group ablations.
    patched=RESULTS/"rae_ablation"/"pit_audit_computational_ablation.py"
    copy_auditor(AUDITOR_SRC,patched)
    rae_rows=[]; ablation_rows=[]; structural_rows=[]; traceability_rows=[]; timing=[]; structural_runtime=0.0
    for i,item in enumerate(metadata):
        cid=item["case_id"]; folder=item["folder"]; faults=set(item["faults"])
        report,t=run_rae(patched,folder,RESULTS/"rae_full"/f"{cid}.json")
        errors=list(report.get("errors",[])); detected=report.get("status")!="PASS"
        linked=rae_traceability(folder,RESULTS/"rae_full"/f"{cid}.json",item["affected"])
        traceability_rows.extend([{"case_id":cid,**x} for x in linked])
        rae_rows.append({"case_id":cid,"case_type":cases[i][0],"expected_faults":"|".join(item["faults"]) or "none",
                         "detected":detected,"status":report.get("status"),"diagnostic_count":len(errors),
                         "fault_classes_reported":"|".join(sorted(fault_classes(errors))),"n_truth_faults":len(item["affected"]),
                         "n_faults_record_linked":sum(x["record_linked"] for x in linked),"runtime_seconds":t})
        timing.append(t)
        st0=time.perf_counter(); structural_found=structural_categories(folder); structural_runtime+=time.perf_counter()-st0
        structural_rows.append({"case_id":cid,"expected_faults":"|".join(item["faults"]) or "none",
                                "detected":bool(structural_found),"fault_classes_reported":"|".join(sorted(structural_found))})
        for category in ABLATIONS:
            enabled=["release","vintage","dependency","eligibility","mapping"]
            enabled.remove(category)
            rpt,at=run_rae(patched,folder,RESULTS/"rae_ablation"/f"{cid}_without_{category}.json",enabled)
            ab_errors=list(rpt.get("errors",[]))
            ablation_rows.append({"case_id":cid,"expected_faults":"|".join(item["faults"]) or "none",
                                  "ablated_category":category,"detected":rpt.get("status")!="PASS",
                                  "fault_classes_reported":"|".join(sorted(fault_classes(ab_errors))),
                                  "diagnostic_count":len(rpt.get("errors",[])),"runtime_seconds":at})
    write_csv(RESULTS/"rae_case_results.csv",rae_rows)
    write_csv(RESULTS/"rae_fault_traceability.csv",traceability_rows)
    write_csv(RESULTS/"structural_case_results.csv",structural_rows)
    write_csv(RESULTS/"rae_single_module_ablation.csv",ablation_rows)
    # Run actual Feast 0.67.0 historical feature retrieval over all case records.
    vintages_by_case={}
    feature_rows=[]; entity_rows=[]; query_record_map={}; feast_timing={}
    for item in metadata:
        cid=item["case_id"]; folder=item["folder"]
        versions=read_csv(folder/"vintages.csv")
        inputs=read_csv(folder/"inputs.csv")
        origins={(r["model"],r["origin_id"]):r for r in read_csv(folder/"origins.csv")}
        for inp_i,r in enumerate(inputs):
            if r["masked"].lower()=="true": continue
            key=(r["model"],r["origin_id"]); clock=datetime.fromisoformat(origins[key]["origin_timestamp"])
            rec_id=f"{cid}|{r['model']}|{r['origin_id']}|{r['feature']}|{r['observation_month']}"
            entity_id=rec_id
            entity_rows.append({"record_id":rec_id,"entity_id":entity_id,
                "event_timestamp":(clock-timedelta(microseconds=1)).isoformat()})
            query_record_map[rec_id]={"case_id":cid,"record_id":rec_id,"model":r["model"],"origin_id":r["origin_id"],
                "feature":r["feature"],"observation_month":r["observation_month"],
                "expected_value":r["value"],"expected_vintage":r["vintage_id"]}
            candidate_month=r["source_observation_month"]
            candidates=[v for v in versions if v["source_id"]==r["source_id"] and v["series_id"]==r["series_id"]
                        and v["observation_month"]==candidate_month]
            # The materialized feature is keyed to its declared context month. Feast uses event/created
            # timestamps; it does not inspect the source_observation_month dependency column.
            for v in candidates:
                feature_rows.append({"entity_id":entity_id,"event_timestamp":month_start(candidate_month),
                    "created_timestamp":v["available_at"],"value":float(v["value"]),"vintage_id":v["vintage_id"]})
    featdf=pd.DataFrame(feature_rows); entdf=pd.DataFrame(entity_rows)
    featdf["event_timestamp"]=pd.to_datetime(featdf["event_timestamp"],utc=True)
    featdf["created_timestamp"]=pd.to_datetime(featdf["created_timestamp"],utc=True)
    entdf["event_timestamp"]=pd.to_datetime(entdf["event_timestamp"],utc=True)
    FEAST_ROOT.mkdir(parents=True,exist_ok=True); (FEAST_ROOT/"data").mkdir(exist_ok=True)
    feature_path=FEAST_ROOT/"data"/"features.parquet"; featdf.to_parquet(feature_path,index=False)
    (FEAST_ROOT/"feature_store.yaml").write_text("project: rae_feast_benchmark\nprovider: local\nregistry: data/registry.db\nonline_store:\n  type: sqlite\n  path: data/online.db\n",encoding="utf-8")
    from feast import Entity, FeatureView, Field, FileSource, FeatureStore
    from feast.types import Float64, String
    entity=Entity(name="case_feature_cell",join_keys=["entity_id"])
    src=FileSource(name="case_feature_history",path=str(feature_path),timestamp_field="event_timestamp",created_timestamp_column="created_timestamp")
    fv=FeatureView(name="case_feature_view",entities=[entity],schema=[Field(name="value",dtype=Float64),Field(name="vintage_id",dtype=String)],source=src,ttl=timedelta(days=50000))
    store=FeatureStore(repo_path=str(FEAST_ROOT));store.apply([entity,fv])
    feast_outputs={}
    for strict,label in ((False,"feast_default"),(True,"feast_created_time_filter")):
        t0=time.perf_counter()
        retrieved=store.get_historical_features(entity_df=entdf,features=["case_feature_view:value","case_feature_view:vintage_id"],
            filter_by_created_timestamp=strict).to_df()
        elapsed=time.perf_counter()-t0
        check=[]
        for r in retrieved.to_dict(orient="records"):
            expected=query_record_map[str(r["record_id"])]
            val=r.get("value"); vid=r.get("vintage_id")
            try: value_ok=val is not None and not pd.isna(val) and abs(float(val)-float(expected["expected_value"]))<1e-12
            except Exception: value_ok=False
            match=value_ok and str(vid)==str(expected["expected_vintage"])
            mismatch=not match
            detected_categories=[]
            if val is None or vid is None or pd.isna(val) or pd.isna(vid): detected_categories.append("release_cutoff")
            elif mismatch: detected_categories.append("stale_vintage")
            check.append({"case_id":expected["case_id"],"record_id":expected["record_id"],"mismatch":mismatch,
                          "detected_fault_classes":"|".join(detected_categories),
                          "returned_value":None if pd.isna(val) else float(val),"returned_vintage":None if pd.isna(vid) else str(vid)})
        feast_outputs[label]=check
        (RESULTS/"feast_comparison"/f"{label}_records.csv").parent.mkdir(parents=True,exist_ok=True)
        write_csv(RESULTS/"feast_comparison"/f"{label}_records.csv",check)
        timing.append(elapsed); feast_timing[label]=elapsed
    feast_case={}
    for label,records in feast_outputs.items():
        by_case={}
        for r in records: by_case[r["case_id"]]=by_case.get(r["case_id"],False) or r["mismatch"]
        feast_case[label]=by_case
    write_csv(RESULTS/"feast_comparison"/"feast_case_results.csv",[
        {"case_id":item["case_id"],"expected_faults":"|".join(item["faults"]) or "none",
         "feast_default_detected":feast_case["feast_default"].get(item["case_id"],False),
         "feast_created_timestamp_filter_detected":feast_case["feast_created_time_filter"].get(item["case_id"],False)} for item in metadata])
    # Aggregate case-level performance and fault-category recall; labels remain generator-defined.
    rae_by={r["case_id"]:r for r in rae_rows}; struct_by={r["case_id"]:r["detected"] for r in structural_rows}
    detector_cases={"RAE full":{k:v["detected"] for k,v in rae_by.items()},
      "Structural audit":struct_by,"Feast default PIT":feast_case["feast_default"],
      "Feast event+created cutoff":feast_case["feast_created_time_filter"]}
    detector_classes={
      "RAE full":{r["case_id"]:set(filter(None,r["fault_classes_reported"].split("|"))) for r in rae_rows},
      "Structural audit":{r["case_id"]:set(filter(None,r["fault_classes_reported"].split("|"))) for r in structural_rows},
    }
    for label,records in feast_outputs.items():
        by_case={}
        for r in records: by_case.setdefault(r["case_id"],set()).update(filter(None,r["detected_fault_classes"].split("|")))
        display="Feast default PIT" if label=="feast_default" else "Feast event+created cutoff"
        detector_classes[display]=by_case
    for category in ABLATIONS:
        detector_classes[f"RAE without {category}"]={
            r["case_id"]:set(filter(None,r["fault_classes_reported"].split("|")))
            for r in ablation_rows if r["ablated_category"]==category}
    ab_by={(r["case_id"],r["ablated_category"]):r["detected"] for r in ablation_rows}
    for category in ABLATIONS: detector_cases[f"RAE without {category}"]={item["case_id"]:ab_by[(item["case_id"],category)] for item in metadata}
    aggregates=[]
    for name,pred in detector_cases.items():
        clean=[item["case_id"] for item in metadata if not item["faults"]]
        positives=[item["case_id"] for item in metadata if item["faults"]]
        tp=sum(bool(pred.get(c,False)) for c in positives); fn=len(positives)-tp
        fp=sum(bool(pred.get(c,False)) for c in clean); tn=len(clean)-fp
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0
        runtime=sum(r["runtime_seconds"] for r in rae_rows) if name=="RAE full" else structural_runtime if name=="Structural audit" else feast_timing.get("feast_default",0.0) if name=="Feast default PIT" else feast_timing.get("feast_created_time_filter",0.0) if name=="Feast event+created cutoff" else sum(r["runtime_seconds"] for r in ablation_rows if r["ablated_category"]==name.removeprefix("RAE without "))
        aggregates.append({"detector":name,"cases":len(metadata),"tp":tp,"fn":fn,"fp_clean":fp,"tn_clean":tn,
                           "precision_case":prec,"recall_case":rec,"f1_case":f1,
                           "clean_false_positive_rate":fp/len(clean),"runtime_seconds":runtime})
    write_csv(RESULTS/"detector_summary.csv",aggregates)
    fault_agg=[]
    for name,predcats in detector_classes.items():
        for group,members in FAULT_GROUPS.items():
            subset=[i for i in metadata if members.intersection(i["faults"])]
            tp=sum(bool(predcats.get(i["case_id"],set()).intersection({group})) for i in subset)
            fn=len(subset)-tp
            other_fault_cases=[i for i in metadata if not members.intersection(i["faults"])]
            fp=sum(group in predcats.get(i["case_id"],set()) for i in other_fault_cases)
            tn=len(other_fault_cases)-fp
            precision=tp/(tp+fp) if tp+fp else 0
            recall=tp/(tp+fn) if tp+fn else 0
            f1=2*precision*recall/(precision+recall) if precision+recall else 0
            fault_agg.append({"detector":name,"fault_group":group,"n_positive_cases":len(subset),"tp":tp,"fn":fn,"fp_other_fault_cases":fp,"tn":tn,
                              "precision":precision,"recall":recall,"f1":f1})
    write_csv(RESULTS/"fault_specific_summary.csv",fault_agg)
    report={"benchmark_id":"RAE_FEAST_HELDOUT_800_20261110","seed":SEED,"n_cases":len(metadata),
      "n_clean":sum(not item["faults"] for item in metadata),
      "n_single_fault":sum(len(item["faults"])==1 for item in metadata),
      "n_multi_fault":sum(len(item["faults"])>1 for item in metadata),
      "fault_types":list(SIMPLE_FAULTS),"multi_pair_types":[list(p) for p in pairs],
      "rae_version":"module-gated PIT auditor; each disabled module skips its validation computations",
      "computational_ablation_modules":list(ABLATIONS),
      "full_factorial":"See run_computational_ablation.py for the complete 2^5 design and interaction estimates on the same frozen fixtures.",
      "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "auditor_source_sha256":hashlib.sha256(AUDITOR_SRC.read_bytes()).hexdigest(),
      "feast_version":"0.67.0","feast_modes":["default event-time join","event-time plus created-timestamp filter"],
      "created_cutoff_alignment":"entity timestamp shifted 1 microsecond before origin to align Feast <= with RAE strict <; Feast event time uses the source observation month",
      "same_fixture_metadata":True,"scores_read":False,"p1_accessed":False,
      "rae_runtime_total_seconds":sum(r["runtime_seconds"] for r in rae_rows),
      "feast_query_runtime_seconds":{label:timeval for label,timeval in [("default",None),("created_time_filter",None)]},
      "detectors":aggregates,
      "fault_group_results_csv":"fault_specific_summary.csv",
      "record_traceability_csv":"rae_fault_traceability.csv",
      "notes":["Feast is a point-in-time feature retrieval baseline; it is not a complete target/key/score auditor.",
               "Named RAE ablations switch off the corresponding validation computation in pit_audit.py; integrity checks remain active.",
               "Cases are synthetic and generator-defined; performance describes this injection suite."]}
    # Replace query timings with actual values written in insertion order.
    report["feast_query_runtime_seconds"]={"default":timing[-2],"created_time_filter":timing[-1]}
    (RESULTS/"benchmark_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    # File hashes for reproducibility of this derived experiment.
    manifest=[]
    for f in sorted(RESULTS.rglob("*")):
        if f.is_file() and f.name!="SHA256SUMS.txt": manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(RESULTS).as_posix()}")
    (RESULTS/"SHA256SUMS.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
    print(json.dumps({"n_cases":len(metadata),"summary":aggregates,"feast_default_runtime":timing[-2],
                      "feast_created_filter_runtime":timing[-1],"results":str(RESULTS)},indent=2))

if __name__ == "__main__":
    # patch_auditor references these names from the imported, packaged implementation namespace.
    sys.path.insert(0,str(BASE))
    AUDITOR_SRC = BASE / "pit_audit.py"
    main()


#!/usr/bin/env python3
"""Run a true 2^5 computational ablation on frozen RAE injection fixtures."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, itertools, json, platform, random, statistics, sys, time
from pathlib import Path

MODULES = ("release", "vintage", "dependency", "eligibility", "mapping")
FAULT_TO_MODULE = {
    "late_release": "release", "boundary_equal": "release",
    "stale_vintage": "vintage", "future_transform_dependency": "dependency",
    "model_sample_mismatch": "eligibility", "target_mapping": "mapping",
}
GROUPS = ("release", "vintage", "dependency", "eligibility", "mapping")
SEED = 2026111017


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0])
    else:
        fields = []
    with path.open("x", encoding="utf-8", newline="") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def classify(errors: list[str]) -> set[str]:
    s = " | ".join(errors).lower(); out=set()
    if any(x in s for x in ("available_at is not strictly before origin", "selected vintage was not available before origin", "target was released by origin", "target release is not after derived origin", "input observation month is after forecast origin")):
        out.add("release")
    if any(x in s for x in ("selected input vintage absent", "input value/hash differs from selected vintage", "selected stale vintage", "ambiguous latest vintage")):
        out.add("vintage")
    if "source observation month is after input context month" in s:
        out.add("dependency")
    if any(x in s for x in ("model origin sets differ", "origin model set mismatch", "input references unknown model-origin", "forecast references unknown model-origin", "forecast emitted despite missing required input rows", "origin timestamps differ between models")):
        out.add("eligibility")
    if any(x in s for x in ("target-origin-horizon mismatch", "forecast target absent from target calendar", "unregistered horizon")):
        out.add("mapping")
    return out


def load_auditor(path: Path):
    spec=importlib.util.spec_from_file_location("computational_pit_audit",path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def configs():
    out=[]
    for bits in itertools.product((0,1), repeat=len(MODULES)):
        enabled=tuple(m for m,b in zip(MODULES,bits) if b)
        key="M"+"".join(map(str,bits))
        if len(enabled)==len(MODULES): label="RAE-Full"
        elif len(enabled)==len(MODULES)-1:
            off=next(m for m in MODULES if m not in enabled)
            label=f"RAE w/o {off.title()}"
        elif not enabled: label="Integrity-only (all five RAE modules off)"
        else: label="Enabled: "+", ".join(enabled)
        out.append({"variant_id":key,"label":label,"bits":bits,"enabled":enabled})
    return out


def quantile(xs, q):
    if not xs: return float("nan")
    ys=sorted(xs); pos=(len(ys)-1)*q; lo=int(pos); hi=min(lo+1,len(ys)-1); frac=pos-lo
    return ys[lo]*(1-frac)+ys[hi]*frac


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--out",type=Path,default=None,
                    help="new output directory; it and all output files must not already exist")
    args=ap.parse_args(); root=args.root.resolve()
    case_root=root/"cases"; truth_path=root/"results"/"independent_truth_800.csv"
    manifest_path=root/"results"/"case_fixture_manifest.csv"
    auditor_path=root/"baseline_source"/"pit_audit.py"
    out=args.out.resolve() if args.out else root/"results"/"computational_ablation"
    protected=(case_root.resolve(), (root/"baseline_source").resolve())
    if any(out == path or is_within(out, path) for path in protected):
        raise ValueError(f"output directory must not be inside a protected input/source directory: {out}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out}; choose a new --out path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    truth=read_csv(truth_path); fixtures=read_csv(manifest_path)
    if len(truth)!=800: raise RuntimeError(f"expected 800 truth rows, got {len(truth)}")
    case_ids=[r["case_id"] for r in truth]
    if len(set(case_ids))!=800: raise RuntimeError("duplicate case identifiers")
    for row in fixtures:
        f=root/row["path"]
        if not f.is_file() or sha(f)!=row["sha256"]: raise RuntimeError(f"frozen fixture hash mismatch: {row['path']}")
    auditor=load_auditor(auditor_path)
    variants=configs()
    jobs=[(v,r) for v in variants for r in truth]
    random.Random(SEED+1).shuffle(jobs)
    results=[]; runtime_by_variant={v["variant_id"]:[] for v in variants}; module_counts={v["variant_id"]:{m:0 for m in MODULES} for v in variants}
    start=time.perf_counter()
    for k,(variant,tr) in enumerate(jobs,1):
        case_id=tr["case_id"]; folder=case_root/case_id
        fake_out=out/"not_written"/f"{case_id}_{variant['variant_id']}.json"
        t0=time.perf_counter()
        report=auditor.run_audit(folder/"bundle.json",fake_out,enabled_modules=variant["enabled"],write_outputs=False)
        elapsed=time.perf_counter()-t0; runtime_by_variant[variant["variant_id"]].append(elapsed)
        for m in MODULES:
            calls=report["checks"]["module_execution"][m]["checks_executed"]
            if m not in variant["enabled"] and calls != 0:
                raise RuntimeError(f"disabled module executed checks: {variant['variant_id']} / {m} / {calls}")
            if m in variant["enabled"] and calls == 0:
                raise RuntimeError(f"enabled module executed no checks: {variant['variant_id']} / {m}")
            module_counts[variant["variant_id"]][m]+=calls
        found=classify(report.get("errors",[]))
        expected_names=[x for x in tr["expected_faults"].split("|") if x and x!="none"]
        expected=set(FAULT_TO_MODULE[x] for x in expected_names)
        results.append({"case_id":case_id,"variant_id":variant["variant_id"],"variant_label":variant["label"],
            "modules_enabled":"|".join(variant["enabled"]),"expected_fault_groups":"|".join(sorted(expected)),
            "detected_fault_groups":"|".join(sorted(found)),"n_expected_faults":len(expected),
            "status":report["status"],"detected_any":bool(report.get("errors")),
            "detected_any_expected_class":bool(expected & found),"detected_all_expected_classes":bool(expected) and expected.issubset(found),
            "diagnostic_count":len(report.get("errors",[])),"runtime_seconds":elapsed,
            "module_check_counts_json":json.dumps({m:report["checks"]["module_execution"][m]["checks_executed"] for m in MODULES},separators=(",",":"))})
        if k % 1600 == 0:
            print(json.dumps({"completed":k,"total":len(jobs),"elapsed_seconds":round(time.perf_counter()-start,1)}),flush=True)
    results.sort(key=lambda r:(r["variant_id"],r["case_id"]))
    by={v["variant_id"]: [r for r in results if r["variant_id"]==v["variant_id"]] for v in variants}
    truth_map={r["case_id"]:r for r in truth}
    cfg_summary=[]
    for v in variants:
        rows=by[v["variant_id"]]; clean=[r for r in rows if truth_map[r["case_id"]]["expected_faults"]=="none"]
        faulty=[r for r in rows if truth_map[r["case_id"]]["expected_faults"]!="none"]
        tp=sum(r["detected_any"] for r in faulty); fn=len(faulty)-tp; fp=sum(r["detected_any"] for r in clean); tn=len(clean)-fp
        prec=tp/(tp+fp) if tp+fp else 0.; rec=tp/(tp+fn) if tp+fn else 0.; f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        times=runtime_by_variant[v["variant_id"]]
        cfg_summary.append({"variant_id":v["variant_id"],"configuration":v["label"],"modules_enabled":"|".join(v["enabled"]),"n_cases":len(rows),"tp":tp,"fn":fn,"fp_clean":fp,"tn_clean":tn,
            "precision_case":prec,"recall_case":rec,"f1_case":f1,"clean_false_positive_rate":fp/len(clean) if clean else 0.,
            "correct_fault_class_cases":sum(r["detected_any_expected_class"] for r in faulty),"all_fault_classes_detected_cases":sum(r["detected_all_expected_classes"] for r in faulty),
            "runtime_seconds_total":sum(times),"runtime_seconds_mean":statistics.mean(times),"runtime_seconds_median":statistics.median(times),"cases_per_second":len(times)/sum(times) if sum(times)>0 else 0.})
    summary_by={r["variant_id"]:r for r in cfg_summary}; full=summary_by["M11111"]
    loo=[]
    for module in MODULES:
        v=next(v for v in variants if len(v["enabled"])==4 and module not in v["enabled"]); m=summary_by[v["variant_id"]]
        loo.append({"module_removed":module,"full_case_recall":full["recall_case"],"ablated_case_recall":m["recall_case"],"full_minus_ablated_recall_pp":100*(full["recall_case"]-m["recall_case"]),
            "full_correct_class_cases":full["correct_fault_class_cases"],"ablated_correct_class_cases":m["correct_fault_class_cases"],
            "full_runtime_seconds":full["runtime_seconds_total"],"ablated_runtime_seconds":m["runtime_seconds_total"],
            "marginal_runtime_seconds_full_minus_ablated":full["runtime_seconds_total"]-m["runtime_seconds_total"],
            "marginal_runtime_percent_full":100*(full["runtime_seconds_total"]-m["runtime_seconds_total"])/full["runtime_seconds_total"] if full["runtime_seconds_total"] else 0.})
    # Module-specific detection performance for every configuration.
    fault_metrics=[]
    for v in variants:
        rows=by[v["variant_id"]]
        for g in GROUPS:
            positives=[r for r in rows if g in set(filter(None,r["expected_fault_groups"].split("|")))]
            negatives=[r for r in rows if g not in set(filter(None,r["expected_fault_groups"].split("|")))]
            tp=sum(g in set(filter(None,r["detected_fault_groups"].split("|"))) for r in positives)
            fp=sum(g in set(filter(None,r["detected_fault_groups"].split("|"))) for r in negatives)
            fault_metrics.append({"variant_id":v["variant_id"],"configuration":v["label"],"fault_group":g,"n_positive_cases":len(positives),"tp":tp,"fn":len(positives)-tp,"recall":tp/len(positives) if positives else 0.,"fp_other_cases":fp,"n_negative_cases":len(negatives),"fpr_other_cases":fp/len(negatives) if negatives else 0.})
    # Exact, balanced factorial effects: main effects and 2-factor differences-in-differences.
    result_index={(r["case_id"],r["variant_id"]):r for r in results}
    def output_value(row, outcome):
        expected=set(filter(None,row["expected_fault_groups"].split("|"))); detected=set(filter(None,row["detected_fault_groups"].split("|")))
        if outcome=="any_fault_detected": return float(row["detected_any"])
        if outcome=="all_expected_faults_detected": return float(bool(expected) and expected.issubset(detected))
        if outcome.startswith("class:"):
            cat=outcome.split(":",1)[1]
            if cat not in expected: return None
            return float(cat in detected)
        return None
    main_effects=[]; interaction_effects=[]
    outcomes=["any_fault_detected","all_expected_faults_detected"]+[f"class:{g}" for g in GROUPS]
    config_by_bits={v["bits"]:v for v in variants}
    for outcome in outcomes:
        eligible_cases=[r for r in results if r["variant_id"]=="M11111" and output_value(r,outcome) is not None]
        cids=[r["case_id"] for r in eligible_cases]
        for i,module in enumerate(MODULES):
            diffs=[]
            for cid in cids:
                onvals=[]; offvals=[]
                for bits,v in config_by_bits.items():
                    y=output_value(result_index[(cid,v["variant_id"])],outcome)
                    (onvals if bits[i] else offvals).append(y)
                diffs.append(statistics.mean(onvals)-statistics.mean(offvals))
            main_effects.append({"outcome":outcome,"module":module,"n_cases":len(cids),"balanced_main_effect_percentage_points":100*statistics.mean(diffs) if diffs else 0.})
        for i,j in itertools.combinations(range(len(MODULES)),2):
            diffs=[]
            for cid in cids:
                by_rest=[]
                for rest in itertools.product((0,1),repeat=len(MODULES)-2):
                    bits=[0]*len(MODULES); bits[i]=1; bits[j]=1
                    it=iter(rest)
                    for k in range(len(MODULES)):
                        if k not in (i,j): bits[k]=next(it)
                    vals=[]
                    for bi,bj in ((1,1),(1,0),(0,1),(0,0)):
                        bb=bits.copy(); bb[i]=bi; bb[j]=bj
                        v=config_by_bits[tuple(bb)]
                        vals.append(output_value(result_index[(cid,v["variant_id"])],outcome))
                    by_rest.append(vals[0]-vals[1]-vals[2]+vals[3])
                diffs.append(statistics.mean(by_rest))
            interaction_effects.append({"outcome":outcome,"module_1":MODULES[i],"module_2":MODULES[j],"n_cases":len(cids),"balanced_factorial_interaction_percentage_points":100*statistics.mean(diffs) if diffs else 0.})
    exec_rows=[]
    for v in variants:
        for m in MODULES:
            enabled=m in v["enabled"]; count=module_counts[v["variant_id"]][m]
            exec_rows.append({"variant_id":v["variant_id"],"configuration":v["label"],"module":m,"enabled":enabled,"checks_executed":count,"mean_checks_per_case":count/800})
    # Compare runtime in a paired bootstrap over the same 800 cases; descriptive only.
    loo_time_ci=[]; rand=random.Random(SEED+9); B=5000
    time_by={(r["case_id"],r["variant_id"]):r["runtime_seconds"] for r in results}
    for module in MODULES:
        abv=next(v for v in variants if len(v["enabled"])==4 and module not in v["enabled"])
        diffs=[time_by[(cid,"M11111")]-time_by[(cid,abv["variant_id"])] for cid in case_ids]
        boot=[]
        for _ in range(B): boot.append(statistics.mean(diffs[rand.randrange(len(diffs))] for __ in range(len(diffs))))
        loo_time_ci.append({"module_removed":module,"mean_full_minus_ablated_seconds_per_case":statistics.mean(diffs),"median_seconds_per_case":statistics.median(diffs),
            "descriptive_paired_bootstrap_95_low":quantile(boot,.025),"descriptive_paired_bootstrap_95_high":quantile(boot,.975),"n_cases":len(diffs),"bootstrap_replicates":B})
    # Compact per-class full vs leave-one-out effects and detection interactions.
    full_fault={r["fault_group"]:r for r in fault_metrics if r["variant_id"]=="M11111"}
    fault_loo=[]
    for module in MODULES:
        abv=next(v for v in variants if len(v["enabled"])==4 and module not in v["enabled"])
        for g in GROUPS:
            fr=full_fault[g]; ar=next(r for r in fault_metrics if r["variant_id"]==abv["variant_id"] and r["fault_group"]==g)
            fault_loo.append({"module_removed":module,"injected_fault_group":g,"full_recall":fr["recall"],"ablated_recall":ar["recall"],"recall_drop_percentage_points":100*(fr["recall"]-ar["recall"]),"full_false_positive_rate":fr["fpr_other_cases"],"ablated_false_positive_rate":ar["fpr_other_cases"]})
    write_csv(out/"case_configuration_results.csv",results)
    write_csv(out/"configuration_summary.csv",cfg_summary)
    write_csv(out/"module_fault_performance.csv",fault_metrics)
    write_csv(out/"leave_one_out_module_contribution.csv",fault_loo)
    write_csv(out/"leave_one_out_runtime.csv",loo_time_ci)
    write_csv(out/"factorial_main_effects.csv",main_effects)
    write_csv(out/"factorial_pairwise_interactions.csv",interaction_effects)
    write_csv(out/"module_execution_counts.csv",exec_rows)
    # Recheck the frozen case files after all audit calls.
    for row in fixtures:
        f=root/row["path"]
        if not f.is_file() or sha(f)!=row["sha256"]:
            raise RuntimeError(f"fixture modified during ablation: {row['path']}")
    original_full_path=root/"results"/"rae_case_results.csv"
    original_full_check={"available":original_full_path.is_file()}
    if original_full_path.is_file():
        prior={r["case_id"]:r for r in read_csv(original_full_path)}
        current={r["case_id"]:r for r in results if r["variant_id"]=="M11111"}
        original_full_check.update({"rows":len(prior),"rows_recomputed":len(current),
            "status_mismatches":sum((prior[k]["status"]!="PASS")!=(current[k]["status"]=="FAIL") for k in current),
            "detection_mismatches":sum((prior[k]["detected"].lower()=="true")!=(current[k]["detected_any"] is True) for k in current),
            "diagnostic_count_mismatches":sum(int(prior[k]["diagnostic_count"])!=int(current[k]["diagnostic_count"]) for k in current)})
        if any(original_full_check[k] for k in ("status_mismatches","detection_mismatches","diagnostic_count_mismatches")):
            raise RuntimeError(f"full audit no longer matches archived results: {original_full_check}")
    elapsed=time.perf_counter()-start
    fullrow=summary_by["M11111"]
    report={"benchmark_id":"RAE_COMPUTATIONAL_ABLATION_800_v1","n_cases":800,"positive_cases":700,"clean_controls":100,"design":"complete 2^5 factorial; each named module gates its actual validation computations; structural/integrity checks remain active in every arm","modules":list(MODULES),"configurations":32,"audit_runs":25600,"injected_fault_groups":GROUPS,"truth_sha256":sha(truth_path),"fixture_manifest_sha256":sha(manifest_path),"auditor_sha256":sha(auditor_path),"runner_sha256":sha(Path(__file__).resolve()),"case_fixture_files_verified":len(fixtures),"case_fixture_files_reverified_after_run":len(fixtures),"original_full_result_comparison":original_full_check,"target_values_read":False,"p1_accessed":False,"full_metrics":fullrow,"leave_one_out_metrics":loo,"leave_one_out_runtime_bootstrap":loo_time_ci,"wall_clock_seconds":elapsed,"runtime_environment":{"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"logical_cpu_count":__import__("os").cpu_count()},"timing_note":"Runtime measured per in-memory audit call with report files disabled; includes bundle file reads and parsing. Paired bootstrap intervals are descriptive timing variability summaries, not inferential claims.","interaction_note":"Factorial interaction is the balanced difference-in-differences, averaged over the other three binary module settings; outcomes and denominators are recorded in the CSVs.","fault_classification":"Mapped error-text categories are post-run scoring rules; error computation is module-gated in the auditor itself.","outputs":["case_configuration_results.csv","configuration_summary.csv","module_fault_performance.csv","leave_one_out_module_contribution.csv","leave_one_out_runtime.csv","factorial_main_effects.csv","factorial_pairwise_interactions.csv","module_execution_counts.csv"]}
    write_text_new(out/"benchmark_report.json",json.dumps(report,indent=2)+"\n")
    manifest=[]
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name!="SHA256SUMS.txt": manifest.append(f"{sha(f)}  {f.name}")
    write_text_new(out/"SHA256SUMS.txt","\n".join(manifest)+"\n")
    print(json.dumps({"n_cases":800,"variants":32,"audit_runs":len(results),"full":fullrow,"loo":loo,"wall_clock_seconds":elapsed,"out":str(out)},indent=2))

if __name__=="__main__": main()

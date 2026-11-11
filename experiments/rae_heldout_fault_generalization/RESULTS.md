# Held-out RAE fault generalization: design and results

## Aim and frozen implementation

This experiment tests whether a frozen, portable RAE checker handles case variants beyond the earlier targeted fingerprint-mismatch additions. The implementation snapshot was copied before generating the held-out suite and was not modified afterward. A machine-readable freeze receipt records the source-to-snapshot byte-identity check and its hash-based binding. The audit runner does not read the truth labels while checking a case; a separate scoring step compares its outputs with the generator's truth manifest.

The held-out suite is internally generated and case-held-out from the earlier injection templates. It is not a third-party blinded challenge. All cases are synthetic and target-free; the experiment does not estimate detection rates on real archives or assess forecast accuracy.

## Suite construction

The independent generator creates 200 bundles: 50 clean controls, 120 single-fault cases, and 30 two-fault composite cases. The fault set covers release-cutoff equality and post-cutoff timestamps (including equal instants encoded with different UTC offsets), timestamps without an explicit zone, duplicate vintage identities, stale-vintage selection, direct and transitive dependency-chain violations, missing model-required inputs, model-origin eligibility mismatch, target-horizon mapping errors, and duplicate forecast keys. Clean cases vary CSV line endings, UTF-8 BOM, quoting, column order, and row order. Fault labels and affected families are written before the frozen checker is invoked. Forecast fingerprints are recomputed after input mutations so the tests do not collapse into fingerprint-mismatch detection.

The transitive-dependency cases include an explicit edge list in an auxiliary provenance file. Its direct input record remains temporally admissible, while an upstream dependency points to a later observation. The frozen checker does not consume that edge-list extension, so these cases test a boundary of its current schema rather than a malformed record under the implemented direct-source-month rule.

## Held-out results

| Case group | Cases | Cases flagged | Complete fault-family localization |
|---|---:|---:|---:|
| Clean controls | 50 | 0 | -- |
| Single fault | 120 | 108 | 108 |
| Composite faults | 30 | 30 | 24 |
| All fault cases | 150 | 138 | 132 |

Case-level precision was 100% (138/138 flagged cases were faulted), recall was 92.0% (138/150), F1 was 95.8%, and the clean-control false-positive rate was 0% (0/50). The 12 unflagged cases were all single-fault cases with a future parent hidden behind an otherwise admissible direct dependency. In six further composites, another fault triggered an alarm but the transitive dependency component was not localized. Overall, the checker localized 162 of 180 injected fault components.

| Fault family | Injected components | Localized | Localization recall |
|---|---:|---:|---:|
| Release/time | 36 | 36 | 100% |
| Vintage | 36 | 36 | 100% |
| Direct/transitive dependency | 36 | 18 | 50% |
| Model eligibility | 36 | 36 | 100% |
| Target mapping | 36 | 36 | 100% |

The miss pattern shows that the frozen implementation checks the recorded direct source-month relation but does not traverse an arbitrary transformation dependency graph. Thus, the experiment does not support a claim that this implementation verifies general transitive dependency closure. Such a claim requires a declared edge schema, complete dependency records, and a checker that recursively validates the graph.

## Property-based and metamorphic checks

The standard-library seeded generator ran 500 property trials, with 125 trials for each property: (i) equivalent timestamps represented with varied explicit UTC offsets preserve a valid result; (ii) a timestamp strictly before the origin passes while equality or a later instant is rejected; (iii) valid CSV BOM/line-ending/quoting/column/row variants preserve the result; and (iv) a one-month target-key mutation changes a passing bundle into a rejected bundle. All 500 trials met the declared property, using 625 checker executions because each mapping trial audits both the original and mutated bundle. The cutoff trials included 57 strictly-before cases and 68 equal-or-after cases, all handled as expected.

The execution used Python 3.11.9 and only the standard library. Hypothesis was not installed; the test harness uses deterministic seeded generation and does not provide automatic shrinking or exhaustive exploration. This distinction is recorded rather than describing the run as a Hypothesis test.

## Reproduction and files

From the experiment directory, run these commands in order:

    python generator/generate_heldout_suite.py
    python runner/run_property_tests.py
    python runner/run_heldout_suite.py

The held-out suite generator does not import RAE; the runner loads only frozen_rae/pit_audit.py. The truth manifest, per-case outcomes, family summary, property-test report, freeze receipt, environment record, all case bundles, and SHA-256 manifest are included in this experiment directory. Rebuilding the suite overwrites its generated cases/ and truth manifest; do so only in a working copy. The runner writes result summaries under results/ and does not alter the frozen checker or case bundles.

## Interpretation

The experiment extends the tested perturbations to time-zone boundaries, duplicated and stale vintages, composite errors, dependency chains, and valid serialization changes. It supports the tested implementation's robustness to the sampled representations and its ability to detect the represented fault cases. The transitive-dependency miss identifies a concrete schema and implementation boundary. Results are conditional on the generator's fault definitions and the single checker snapshot; they are not a real-world sensitivity estimate or a comparison against a mature external audit product.


## Post-freeze dependency-graph regression update

The results above belong to the checker snapshot frozen before held-out case generation. The current portable checker now validates the supplied dependency DAG: it checks node/edge integrity and missing parents, traverses transitive ancestors iteratively, detects cycles with a Kahn topological pass, checks every reachable node's reference month and availability against the origin, and verifies each derived node's transformation ID, canonical parameter hash, and code hash against the supplied registry. An input-level transformation ID, if present, must match the graph root.

The portable repository suite (run record: `tests/dependency_graph_test_results.json`) contains 12 targeted regression cases for H0100, H0102, H0104, H0106, H0108, H0110, H0112, H0114, H0116, H0118, H0120, and H0122. Each test reconstructs the dependency edges from the archived case and verifies rejection when its injected upstream source is assigned an availability timestamp just after the forecast origin. The tests also check multilevel pass/fail behavior, cycles, missing parents, registry and hash mismatches, origin-time timezone validity, missing graphs, and traversal of a 1,100-node chain. These are regression tests on previously identified misses, not a new held-out detection-rate estimate; the archived 138/150 metric is unchanged.

The implementation validates supplied metadata and hash links only. Completeness and authenticity of dependency edges and transformation registries are assumptions; it does not infer omitted dependencies or numerically replay transformations. See the portable schema in `src/rae/PIT_AUDIT_README.md`.

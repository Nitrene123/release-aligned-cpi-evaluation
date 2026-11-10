# IPT source workbooks

The historical run used the Philadelphia Fed monthly-vintage IPT workbook and its first/second/third-release workbook. Neither workbook nor its raw vintage values are included in this repository. Obtain the source files from the official IPT data page:

https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ipt

The archived protocol records the expected source SHA-256 values and retrieval date. Check the downloaded bytes against that record before rebuilding the input contexts. The reconstruction command takes local paths; keep downloaded source files outside this repository.

```bash
python experiments/ipt/code/rebuild_inputs.py <local-vintage-workbook> <local-output-contexts>
```

The model inference script accepts `--checkpoint` or the `TIMESFM3_CHECKPOINT` environment variable. Use `--expected-sha256` or `TIMESFM3_MODEL_SHA256` to check the local checkpoint when a provider-verified hash is available.

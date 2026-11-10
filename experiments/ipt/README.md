# IPT vintage-stress evaluation

The public folder includes input-rebuild and scoring code, source instructions, forecast outputs, and target-free score summaries. It excludes the original Philadelphia Fed workbooks, full vintage-value panels, and raw target values. Download the workbooks from the provider, verify their source hashes, and rebuild a local context bundle before inference. The TimesFM-3 checkpoint is supplied through `--checkpoint` or `TIMESFM3_CHECKPOINT`; no local server path is embedded in the script.

# Data sources and acquisition

This repository contains code, synthetic audit fixtures, model forecast outputs, and target-free derived score tables. It does not redistribute raw third-party economic datasets or historical target-value ledgers.

| Material | Official source | Local reconstruction |
|---|---|---|
| China CPI releases | National Bureau of Statistics of China: https://www.stats.gov.cn/ | Use the origin calendars and source-field descriptions in the experiment protocol; retain release time, retrieval date, and source URL for each observation. |
| U.S. CPI-U | U.S. Bureau of Labor Statistics CPI program: https://www.bls.gov/cpi/ | The score scripts accept a first-release target table reconstructed from official BLS releases. The raw target table and response-page snapshots are not included. |
| FRED/ALFRED series | Federal Reserve Bank of St. Louis: https://fred.stlouisfed.org/ | Obtain each series under its provider terms and use ALFRED real-time fields for vintage-aware reconstruction. |
| IPT vintage history | Federal Reserve Bank of Philadelphia Real-Time Data Set: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ipt | Download the provider workbooks from the source page. `experiments/ipt/code/rebuild_inputs.py` accepts a local workbook path and creates local contexts. Workbooks are not mirrored here. |
| TimesFM-3 checkpoint | Official Google Hugging Face model repository: https://huggingface.co/google/timesfm-3.0-pytorch | Download the exact checkpoint from the provider under the license and terms attached to that revision. Weights are not mirrored. Pass the local checkpoint directory through `--checkpoint` or `TIMESFM3_CHECKPOINT`. |

Public availability does not by itself establish a right to redistribute a data file. Follow the provider's current access and reuse terms. Keep downloaded source files outside the repository unless redistribution permission is confirmed.

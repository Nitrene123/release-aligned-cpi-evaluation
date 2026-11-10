# ETS and Theta: supplementary matched-origin rolling evaluation

This prespecifies implementation choices for a new analysis of historical
results that the project has already examined. It is a development record,
not preregistration, blinded evaluation, or prospective evidence. The two
new models are added without changing any existing model or forecast.

## Cohorts and units

* United States: the existing BLS CPI-U first-release 12-month percentage
  changes for 84 target months, January 2018 to December 2024, and the exact
  archived origin clock, target month, and 1, 3, 6, 12 month horizon keys.
* China: the existing NBS first-release CPI year-on-year percentage changes
  for 105 target months, January 2018 to September 2026, and the exact archived
  unified-cohort origin clocks and target keys. The existing ten-model
  intersection has 99 keys per horizon; the native planned cohort has 105.
* One forecast origin is a unit of comparison. Distinct horizons and model
  seeds do not create additional independent months. Do not combine countries.

## Information and fitting rules

For each country and origin, begin with that country's existing first-release
year-on-year history. Retain only reference months no later than the origin
month with availability strictly before its clock. Require a contiguous
monthly history from 2012-01 through the origin, with at least 60 finite
observations. If eligibility fails, retain a failure status; do not impute,
substitute a value, shorten or silently shift the window. Fit once per origin
and produce all requested horizons (up to 12) from that fitted path.
Check the origin clock, training range, number of observations, and terminal
value against the corresponding archived No-change forecast at every key.
The BLS origin is one minute after the actual release; the China comparison
uses the frozen clocks in the China unified planned keys. Later published
labels may enter later origin histories only after their recorded release.

Fixed baseline specifications, both implemented by statsmodels 0.14.4:

* **ETS(A,A,N):** `ETSModel(y, error='add', trend='add',
  damped_trend=False, seasonal=None, initialization_method='estimated',
  missing='none').fit(maxiter=1000, full_output=True, disp=False)`.
  The additive-error, additive-trend, non-seasonal model is fit by its standard
  maximum-likelihood/L-BFGS procedure, re-estimating parameters at each origin.
* **Theta:** `ThetaModel(y, period=12, deseasonalize=False, use_test=False,
  method='additive', difference=False).fit(use_mle=False, disp=False)`, and
  `forecast(steps, theta=2)`. This is the library's classical two-step Theta
  procedure (OLS time trend plus simple exponential smoothing), without a
  separate seasonal adjustment. No theta or model selection uses test targets.

Both models run on CPU, with fixed library versions and no random seed needed.
For ETS, only a finite path with `mle_retvals.converged == True` is complete;
non-convergence is a fit failure. For Theta, any `ConvergenceWarning` or
nonfinite fitted parameters/forecast is a fit failure. All other fit warnings,
parameters, iteration details where available, exceptions, training boundaries,
availability checks, input hashes, and execution times are retained. No
alternate model, optimizer, retry, truncation, refit on the test month, or
manual exclusion is allowed.

## Scoring and reporting

Freeze and hash all new forecasts and fit records before separate scoring.
Use the archived official first-release target values and verified source
tables; target month and release time must match the original registered key
and follow the nominal origin. Derive per-origin absolute errors and compare
them with the existing ten archived origin-level losses by exact
`country/origin_id/origin_time/target_month/horizon` key. Existing five-seed
heads retain their previously computed mean of seed-level absolute errors.
Report native coverage out of 84 or 105, all 12-method intersections separately,
four-horizon MAE on common samples, and pairwise loss contrasts versus each
existing method with sample counts. No p-values or confirmatory claims. Do not
publish target values unnecessarily. Prediction files remain separate from
original archives; model weights and P1 queue are not accessed.

## Reproducibility

Record library and Python versions, hashes of the input ledgers, source code,
forecast archive, and score outputs, execution times and any failures. Produce
an independent verifier that reconstructs all new origin losses and reports
and checks source hashes, timing, keys and target-free paired comparisons.

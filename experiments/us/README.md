# U.S. CPI cross-country evaluation

This folder contains the frozen scorer, target-validation code, protocol, forecast outputs, and target-free derived score tables. The BLS target CSV, release-page response bytes, input-value panels, and execution logs are not included. To rerun a score, reconstruct the first-release target table from official BLS releases and pass it to the scorer with locally prepared validation and protocol files. Compressed forecast tables can be read with Python's `gzip` module or standard archive tools.

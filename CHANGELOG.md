# Changelog

## v1.0.5

* Documenting the `drs-transcript-prediction` command in the README
* Refreshing the README help text (adds `--version` and `--candidate_read_threshold` to the usage line)
* Fixing `--chromosome` never matching when chromosome IDs are numeric (i.e. `1`)
* Skipping leading header lines (`#`, `track`, `browser`) in input BED files instead of crashing
* Exiting with an error message when an input BED file has no data lines
* Fixing `--max_depth 0` not being rejected
* Fixing the error raised when a region has multiple entries, and the wrong path in the missing `--reads_bed` message
* Adding small test dataset (`test-data/`) and pytest regression tests (`tests/`)

## v1.0.4

* Adding `drs-transcript-prediction` command as an alternative to `tp.py`

## v1.0.3

* Renaming the pip package from `tp` to `drs-transcript-prediction`

## v1.0.2

* Adding pyproject.toml so the tool can be installed with pip (installs a `tp.py` command)
* Adding --version option
* Removing unused scikit-learn dependency
* Minimum supported Python version is now 3.9

## v1.0.1

* Adding default values to help text
* Print arguments if verbose set

## v1.0.0

* Initial release
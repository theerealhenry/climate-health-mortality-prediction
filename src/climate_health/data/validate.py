"""
Command-line entry point for `make validate`.

Loads and validates every raw input file through `loaders.py`, prints a
pass/fail summary for each, and exits non-zero if any file fails its
contract — so this is safe to wire into CI (Stage 18) as a hard gate, not
just a local sanity check.

Usage:
    python -m climate_health.data.validate
"""

import sys

from climate_health.data.loaders import (
    DataContractError,
    load_climate_features,
    load_sample_submission,
    load_test,
    load_train,
)

_CHECKS = [
    ("Train.csv", load_train),
    ("Test.csv", load_test),
    ("climate_features.csv", load_climate_features),
    ("SampleSubmission.csv", load_sample_submission),
]


def main() -> int:
    print("Validating raw data contracts (docs/PROJECT_BLUEPRINT.md Stage 2)\n")
    all_passed = True

    for label, loader_fn in _CHECKS:
        try:
            df = loader_fn()
            print(f"  [PASS] {label:<28} {df.shape[0]:>5} rows x {df.shape[1]:>2} cols")
        except FileNotFoundError as exc:
            all_passed = False
            print(f"  [MISSING] {label:<28} {exc}")
        except DataContractError as exc:
            all_passed = False
            print(f"  [FAIL] {label}")
            print(f"         {exc}\n")

    print()
    if all_passed:
        print("All raw data contracts satisfied.")
        return 0
    else:
        print("One or more raw data contracts failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

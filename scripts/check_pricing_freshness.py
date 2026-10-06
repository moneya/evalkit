#!/usr/bin/env python3
"""Check whether the bundled pricing table has gone stale.

Pricing pages change without notice, so the table carries a `_verified` date and
CI warns when it ages past a threshold. This script does not scrape — vendor
pages are JS-heavy and rate-limited, and a scraper that silently half-fails is
worse than a date check. It tells a human to go look.

    python scripts/check_pricing_freshness.py --max-age-days 60
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

sys.path.insert(0, "src")

from evalkit import pricing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-days", type=int, default=60)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when stale (default: warn only)")
    args = ap.parse_args()

    verified = pricing.verified_on()
    try:
        d = datetime.strptime(verified, "%Y-%m-%d").date()
    except ValueError:
        print(f"::error::pricing table has no parseable _verified date (got {verified!r})")
        return 1

    age = (date.today() - d).days
    n = len(pricing.known_models())
    print(f"pricing table: {n} models, verified {verified} ({age} days ago)")

    for provider, url in pricing.sources().items():
        print(f"  {provider}: {url}")

    if age > args.max_age_days:
        msg = (f"pricing table is {age} days old (threshold {args.max_age_days}). "
               f"Re-check the source pages above and bump _verified.")
        print(f"::warning::{msg}" if not args.strict else f"::error::{msg}")
        return 1 if args.strict else 0

    print("fresh enough")
    return 0


if __name__ == "__main__":
    sys.exit(main())

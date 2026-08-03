"""Provision `models/` for a fresh clone.

Thin wrapper over :mod:`argus.provision`, which holds the logic so an installed
wheel can bootstrap too (`argus bootstrap`).

    # From the AI Hub jobs recorded in models/argus_jobs.json (needs a token):
    python scripts/fetch_models.py

    # From an existing local copy -- e.g. the Argus prototype's tree:
    python scripts/fetch_models.py --from-dir ../QUAD-Client/samples/argus_triage/models

    # Re-download and re-repair everything:
    python scripts/fetch_models.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from argus.config import load_config  # noqa: E402
from argus.provision import bootstrap  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Config TOML (default: configs/argus.default.toml)")
    parser.add_argument("--from-dir", type=Path, help="Seed from an existing models/ tree")
    parser.add_argument("--force", action="store_true", help="Overwrite artifacts already present")
    parser.add_argument("--skip-demo", action="store_true", help="Do not regenerate the demo clip")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    return bootstrap(
        cfg, force=args.force, skip_demo=args.skip_demo, from_dir=args.from_dir
    )


if __name__ == "__main__":
    raise SystemExit(main())

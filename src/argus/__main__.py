"""`python -m argus`, and the entry point the packaged binary is built from.

Kept to one line of real work on purpose: `scripts/build_binary.py` freezes
this module, so anything that lived here would run in the binary and not under
`python -m argus.cli`, and the two would drift.
"""

from __future__ import annotations

import sys

from argus.cli import main

if __name__ == "__main__":
    sys.exit(main())

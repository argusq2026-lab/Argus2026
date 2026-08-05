"""Freeze the laptop side into a single executable.

    pip install -e ".[build]"
    python scripts/build_binary.py

Writes `dist/argus` (`dist/argus.exe` on Windows): one file, no Python
install required on the machine that runs it, no virtualenv to activate, and
no `PYTHONPATH` to get right. Launched with no arguments it starts the ingest
server and opens the trainer console, so it can be handed to someone who runs
a gym rather than a terminal.

Two things have to travel with the code for that to be true, and both are
verified by `--check` below rather than assumed:

* **The default config.** `argus.config` looks for it at `argus/_data/
  argus.default.toml` beside the package, which is also where the wheel puts
  it, so the same lookup works frozen and installed. If it went missing the
  binary would fail on every command with a config error.
* **The trainer console.** It is a string in `argus/console.py` rather than a
  data file precisely so this cannot happen — the page is code, so it is
  frozen with the code and there is no packaging step to forget.

PyInstaller is a build-time dependency only; nothing in `src/argus/` imports
it, and the binary is not how the tests run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "src" / "argus" / "__main__.py"
DEFAULT_CONFIG = REPO / "configs" / "argus.default.toml"
#: Where `argus.config._PACKAGED_DEFAULT` looks. Frozen, `__file__` for a
#: bundled module resolves under the extraction root, so `argus/_data/...`
#: lands exactly where the installed-wheel lookup already expects it.
BUNDLED_CONFIG_DIR = "argus/_data"


def build(clean: bool) -> Path:
    if shutil.which("pyinstaller") is None:
        raise SystemExit(
            "pyinstaller not found. Install the build extra first:\n"
            '  pip install -e ".[build]"'
        )
    if clean:
        for directory in (REPO / "build", REPO / "dist"):
            shutil.rmtree(directory, ignore_errors=True)

    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "argus",
        "--distpath", str(REPO / "dist"),
        "--workpath", str(REPO / "build" / "pyinstaller"),
        "--specpath", str(REPO / "build"),
        "--paths", str(REPO / "src"),
        "--add-data", f"{DEFAULT_CONFIG}{os.pathsep}{BUNDLED_CONFIG_DIR}",
        # Every subcommand imports its implementation inside the function that
        # needs it, to keep `argus --version` from paying for websockets. That
        # is good for startup and invisible to a static import graph, so the
        # modules behind those late imports are named here explicitly.
        "--hidden-import", "argus.ingest.server",
        "--hidden-import", "argus.replay",
        "--hidden-import", "argus.discovery",
        "--hidden-import", "argus.doctor",
        "--hidden-import", "argus.synthetic",
        "--noconfirm",
        str(ENTRY),
    ]
    subprocess.run(cmd, check=True, cwd=REPO)

    binary = REPO / "dist" / ("argus.exe" if os.name == "nt" else "argus")
    if not binary.is_file():
        raise SystemExit(f"pyinstaller reported success but {binary} is missing")
    return binary


def check(binary: Path) -> None:
    """Prove the binary is self-contained before anyone ships it.

    Run from a directory that is not the repo: a binary that only works next
    to its own source tree is the exact failure this build exists to avoid,
    and running from the repo root would hide it, because the config and the
    fixtures would be found on disk regardless of what got bundled.
    """
    elsewhere = Path(os.environ.get("TMPDIR", "/tmp"))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}

    for args, must_contain in (
        (["--version"], "argus"),
        (["config"], "scoring"),   # proves the bundled default config was found
        (["doctor"], "checks:"),
    ):
        result = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, cwd=elsewhere, env=env, timeout=120
        )
        output = result.stdout + result.stderr
        if must_contain not in output:
            raise SystemExit(
                f"self-check failed: `argus {' '.join(args)}` did not mention "
                f"{must_contain!r}.\n{output}"
            )
    print(f"self-check passed: {binary} runs standalone from {elsewhere}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove build/ and dist/ first")
    parser.add_argument("--no-check", action="store_true", help="Skip the standalone self-check")
    args = parser.parse_args()

    binary = build(args.clean)
    if not args.no_check:
        check(binary)

    size_mb = binary.stat().st_size / (1024 * 1024)
    print(f"\nbuilt {binary} ({size_mb:.1f} MB)")
    print("run it with no arguments to start the server and open the console:")
    print(f"  {binary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Serve the operator's model weights to their own stations, read-only.

The phones need multi-megabyte weight files that are deliberately not in the
APK or the repository (licensing — see `android/models.json`). The operator
reproduces them locally with `argus fetch-models`; this module hands them to
phones on the floor at `GET /models/` **on the WebSocket ingest port**, which
is already LAN-open by design. No new port, no new listening surface, and a
phone already knows the address — it is the same one it connects to.

This is provisioning within one floor, not distribution: the files never
leave the operator's own network, and this project still ships recipes and
hashes, never weights.

Three deliberate narrowings:

* **Read-only, flat, allow-listed.** GET only; no subdirectories; only the
  extensions a model artifact can have. A path that is not exactly a file
  name in the directory is a 404, so there is no traversal to defend by
  cleverness — the lookup is `name in listing`, not filesystem resolution.
* **Hashes come with the listing.** The manifest carries each file's sha256,
  computed here, so a phone can verify what it stored. This protects against
  torn downloads, not against the operator — the phone already trusts the
  laptop it chose to connect to with its observations.
* **No trainee data can appear here.** The directory holds whatever
  `fetch-models` wrote; nothing in the serving path reads sessions, records,
  or observations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: What a model artifact can be. Anything else in the directory is ignored —
#: serving arbitrary files from a directory an operator might point anywhere
#: is how a models endpoint becomes a file server.
ALLOWED_SUFFIXES = {".onnx", ".data", ".json"}

MANIFEST_PATH = "/models/manifest.json"
FILE_PREFIX = "/models/"


class ModelDirectory:
    """The read-only view of one directory of model artifacts.

    sha256 of a large file is not free, so digests are cached keyed by
    (size, mtime) — touch or replace a file and it is re-hashed, ask twice
    and it is not.
    """

    def __init__(self, directory: Path):
        self._dir = directory
        self._digests: dict[str, tuple[tuple[int, float], str]] = {}

    def _files(self) -> dict[str, Path]:
        if not self._dir.is_dir():
            return {}
        return {
            p.name: p
            for p in sorted(self._dir.iterdir())
            if p.is_file() and p.suffix in ALLOWED_SUFFIXES and not p.name.startswith(".")
        }

    def _digest(self, name: str, path: Path) -> str:
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime)
        cached = self._digests.get(name)
        if cached and cached[0] == key:
            return cached[1]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        self._digests[name] = (key, digest.hexdigest())
        return self._digests[name][1]

    def manifest(self) -> bytes:
        files = self._files()
        payload = {
            "files": [
                {"name": name, "size": path.stat().st_size, "sha256": self._digest(name, path)}
                for name, path in files.items()
            ]
        }
        return json.dumps(payload).encode("utf-8")

    def read(self, name: str) -> bytes | None:
        """The named artifact's bytes, or None — lookup by listing membership,
        so `../` and friends are simply names that are not in the listing."""
        path = self._files().get(name)
        if path is None:
            return None
        return path.read_bytes()

    def respond(self, path: str):
        """(status, content_type, body) for a GET, or None if not ours."""
        if path == MANIFEST_PATH:
            return 200, "application/json", self.manifest()
        if path.startswith(FILE_PREFIX):
            name = path[len(FILE_PREFIX):]
            body = self.read(name)
            if body is None:
                return 404, "text/plain", b"no such model file\n"
            logger.info("serving model %s (%d bytes)", name, len(body))
            return 200, "application/octet-stream", body
        return None

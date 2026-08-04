# Third-party notices

Argus is released under the [MIT License](LICENSE).

There is no vendored third-party source in this repository. (An earlier
iteration of this project vendored an Apache-2.0 QNN Execution Provider
helper for on-device NPU inference; that entire local-inference subsystem —
and the vendored file with it — was removed when pose estimation and
form/exercise classification moved to each trainee's phone. See
[ARCHITECTURE.md](ARCHITECTURE.md).)

---

## Runtime dependencies

Installed from PyPI at their own licences, not redistributed here:

| Package | Licence |
|---|---|
| `websockets` | BSD-3-Clause |

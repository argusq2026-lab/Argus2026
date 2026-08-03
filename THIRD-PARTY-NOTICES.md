# Third-party notices

Argus is released under the [MIT License](LICENSE). One vendored source file
carries a different, compatible licence and keeps it.

---

## `src/argus/engines/ort_qnn.py` — Apache License 2.0

**Upstream:** `quad_mcp_client/ort_qnn.py` in
[github.com/CBN-AI-TEAM/QUAD-Client](https://github.com/CBN-AI-TEAM/QUAD-Client)
**Copyright:** QUAD Contributors
**Licence:** Apache License, Version 2.0

This file is a trimmed derivative of the upstream QNN Execution Provider
session helper. It is vendored rather than declared as a dependency because
`quad-mcp-client` is not published on PyPI.

Apache-2.0 is compatible with MIT for redistribution, but it is not the same
licence, so this file remains under Apache-2.0 and its notice is preserved
here and in the file's own SPDX header. Apache-2.0 §4 requires that these
attribution notices travel with any copy or derivative of that file.

You may obtain a copy of the Apache License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

---

## Models

Model artifacts are **not distributed with this repository** — `models/` is
gitignored and provisioned separately (see [`models/README.md`](models/README.md)).
Their licences are those of their upstream sources, not this repository's:

| Model | Source | Notes |
|---|---|---|
| YOLO-X | [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models) | Check the upstream model card for its licence before redistribution |
| MediaPipe-Pose | Qualcomm AI Hub Models / Google MediaPipe | Apache-2.0 upstream; the exported artifact's terms are AI Hub's |
| QuickSRNet-Medium | Qualcomm AI Hub Models | Check the upstream model card |

Only the JSON manifests in `models/` (job IDs, profiling summaries, per-layer
traces) are committed here. They are measurements produced by this project.

---

## Runtime dependencies

Installed from PyPI at their own licences, not redistributed here: `numpy`
(BSD-3-Clause), `opencv-python` (Apache-2.0), `onnx` (Apache-2.0),
`onnxruntime` (MIT), and optionally `onnxruntime-qnn`, `onnxruntime-genai`,
and `qai-hub` (Qualcomm terms).

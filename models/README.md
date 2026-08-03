# models/

Model artifacts are **not committed** — see `.gitignore`. A fresh clone has
only the four JSON manifests in this directory.

## Provision

```powershell
# From the AI Hub jobs recorded in argus_jobs.json (needs a qai-hub token):
.venv\Scripts\python.exe scripts\fetch_models.py

# Or seed from an existing tree:
.venv\Scripts\python.exe scripts\fetch_models.py --from-dir <path-to-models>
```

`bootstrap` writes any missing `metadata.json`, screens the ONNX artifacts for
duplicate node names, generates the demo clip, and then **verifies** the result
by loading each ONNX artifact and comparing its session I/O to the manifest.

If automated download is unavailable it prints the exact `qai_hub_models`
export commands. Those need a **non-arm64 build host** — `qai-hub-models` pulls
torch, which has no win-arm64 wheel.

## What is committed here

| File | Contents |
|---|---|
| `argus_jobs.json` | AI Hub compile/profile job IDs per model |
| `argus_cu_jobs.json` | CPU (`jg9d8ell5`) and GPU (`jgd20lxe5`) comparison jobs |
| `argus_summary.json` | Measured latency, cycles, layer counts, compute-unit split |
| `argus_profiles_raw.json` | Full per-layer traces — 744 layers across four graphs |

These are the source for every performance number in
[`docs/PIPELINE.md`](../docs/PIPELINE.md). They are small, they are evidence,
and they are what makes the profiling claims checkable, so they stay in git
while the multi-megabyte artifacts do not.

## Expected tree after provisioning

```
models/
├── yolox-onnx-w8a8/
│   ├── yolox.onnx              # NCHW (1,3,640,640) uint8
│   ├── yolox.data              # external weights
│   ├── metadata.json
│   └── labels.txt              # line 1 == "person"
├── mediapipe_pose-qnn_context_binary-w8a8-qualcomm_snapdragon_x_elite/
│   ├── pose_detector.bin           # NHWC (1,128,128,3)
│   ├── pose_landmark_detector.bin  # NHWC (1,256,256,3)  <- different size
│   └── metadata.json
└── quicksrnetmedium-onnx-w8a8/
    ├── quicksrnetmedium.onnx   # NCHW (1,3,128,128) -> (1,3,512,512)
    ├── quicksrnetmedium.data
    └── metadata.json
```

## Two things to know before running on hardware

**QAIRT version skew.** These artifacts were built with QAIRT
**2.45.0.260326154327**. A QNN context binary is tied to the runtime that
produced it, so the two `.bin` files will not load against a different major.
`argus doctor` reports the skew; `QnnNpuBackend` refuses rather than surfacing
an opaque QNN error.

**INT8 calibration used a placeholder set.** No real trainee-floor footage
existed at export time, so no accuracy figure from these artifacts is
trustworthy. One consequence is already measurable: `pose_detector.bin`'s
confidence output has an effectively two-valued range. See
[`docs/VALIDATION.md`](../docs/VALIDATION.md).

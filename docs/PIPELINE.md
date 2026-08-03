# Argus pipeline — models, conversion, and measured NPU performance

Every number in this document comes from a real AI Hub compile or profile job.
The job IDs are recorded in [`models/argus_jobs.json`](../models/argus_jobs.json),
the summaries in [`models/argus_summary.json`](../models/argus_summary.json),
and the full per-layer traces in
[`models/argus_profiles_raw.json`](../models/argus_profiles_raw.json). Nothing
below is estimated, extrapolated, or shaped to a schema.

> **What changed from the prototype's `ARGUS_PIPELINE.md`.** That document's
> Steps 1–4 were explicitly marked 🔶 MOCK and carried invented per-op cycle
> counts, an invented power/latency table, and a model card naming YOLOv8n — a
> model this project has never used. Real jobs had in fact been run. This
> document replaces the fiction with the measurements, and §4 re-derives the
> CPU/NPU split from the real per-layer profile, which contradicts the mocked
> account in a specific and important way.

---

## 1. Models

| Role | Model | Runtime | Precision | Artifact |
|---|---|---|---|---|
| Detection | **YOLO-X** | ONNX (QDQ) | w8a8 | `yolox-onnx-w8a8/yolox.onnx` |
| Pose, stage 1 | **MediaPipe-Pose** detector | QNN context binary | w8a8 | `mediapipe_pose-.../pose_detector.bin` |
| Pose, stage 2 | **MediaPipe-Pose** landmarks | QNN context binary | w8a8 | `mediapipe_pose-.../pose_landmark_detector.bin` |
| Super-resolution | **QuickSRNet-Medium** | ONNX (QDQ) | w8a8 | `quicksrnetmedium-onnx-w8a8/quicksrnetmedium.onnx` |
| VLM | *not yet exported* | — | — | `MockVLMCaptioner` stands in |

The detector is **YOLO-X**, not YOLOv8n. `metadata.json` records
`"model_name": "Yolo-X"`, and the output signature confirms it: three separate
tensors over an 8400-point anchor-free grid (80² + 40² + 20² at strides
8/16/32), which is YOLO-X's decode head.

All four were built with **QAIRT 2.45.0.260326154327** and targeted
**Snapdragon X Elite** (`sc8380xp`, HTP v73, soc_model 60).

### Tensor contracts

These are the contracts the pipeline is written against, read at load time from
each artifact's own `metadata.json` (see `argus/engines/metadata.py`) and
mirrored in `argus/engines/reference_specs.py` for mock mode. There is no
single layout across the pipeline — asserting one is how the prototype's
inference path came to be untestable.

| Artifact | Input | Outputs |
|---|---|---|
| `yolox.onnx` | **NCHW** `(1,3,640,640)` uint8, scale 1/255, zp 0 | `boxes (1,8400,4)` scale 4.4157 zp 51 · `scores (1,8400)` scale 0.003813 zp 0 · `class_idx (1,8400)` raw |
| `pose_detector.bin` | **NHWC** `(1,128,128,3)` uint8 | `box_scores_1 (1,512,1)` · `box_coords_1 (1,512,12)` · `box_scores_2 (1,384,1)` · `box_coords_2 (1,384,12)` |
| `pose_landmark_detector.bin` | **NHWC** `(1,256,256,3)` uint8 | `scores (1,)` · `landmarks (1,25,4)` |
| `quicksrnetmedium.onnx` | **NCHW** `(1,3,128,128)` uint8 | `upscaled_image (1,3,512,512)` uint8 |

Four consequences that are easy to get wrong and impossible to notice:

1. **The two pose binaries take different input sizes.** Only the detector is
   128×128; the landmark stage is 256×256.
2. **`boxes` arrives already decoded to xyxy** in the 640×640 letterboxed
   space. Its quantization range reaches −225 px, which is what lets a box
   hang off the left or top edge.
3. **There are no heatmaps.** The landmark stage regresses 25 points directly,
   so there is no ArgMax decode step anywhere in this pipeline.
4. **The landmark layout is BlazePose, not COCO.** The scorer indexes COCO —
   nose 0, shoulders 5/6, wrists 9/10, hips 11/12 — so the 25 BlazePose points
   are remapped in `argus/vision/keypoints.py`. Without that remap the scorer
   reads a left eye as a shoulder and still returns a plausible number. The
   25-point export is upper-body only, so COCO's knees and ankles (13–16) have
   no source landmark and are reported at confidence 0.0 rather than guessed.

---

## 2. Conversion

Exports were produced by `qai_hub_models`; `scripts/fetch_models.py` reproduces
the tree from the recorded job IDs, or seeds it from a local copy.

| Model | Compile job | Profile job |
|---|---|---|
| yolox | `jgo8m0l1p` | `jpv74olzp` |
| quicksrnetmedium | `jpey21785` | `jg9d81zm5` |
| pose_detector.bin | *(catalog export)* | `j56wvj8ng` |
| pose_landmark_detector.bin | *(catalog export)* | `jp3683zmp` |

> **INT8 calibration used a flagged placeholder set.** No real trainee-floor
> footage existed at export time. Every accuracy figure that would follow from
> these artifacts is therefore unverified, and one consequence is already
> visible in the quantization parameters — see §5. This is the single largest
> open risk in the product; [VALIDATION.md](VALIDATION.md) tracks it.

---

## 3. Measured NPU performance

Snapdragon X Elite CRD, HTP v73. `latency` is AI Hub's
`estimated_inference_time`; cycles and layer counts are from the same job's
per-layer trace.

| Model | Latency | Cycles | Layers | Layers on NPU | Peak memory |
|---|---:|---:|---:|---:|---:|
| yolox | **2,748 µs** | 13,509,779 | 298 | **298 (100%)** | 16.5 MB |
| pose_detector.bin | **421 µs** | 1,383,780 | 136 | **136 (100%)** | 14.5 MB |
| pose_landmark_detector.bin | **425 µs** | 1,208,659 | 291 | **291 (100%)** | 14.5 MB |
| quicksrnetmedium | **632 µs** | 2,418,643 | 19 | **19 (100%)** | 14.8 MB |

**Every layer of every graph runs on the NPU.** There is not one CPU-assigned
layer across all 744 profiled layers.

### Per-op cost — YOLO-X (13,509,779 cycles)

| Op | Layers | Cycles | Share |
|---|---:|---:|---:|
| `Mul` (SiLU activations) | 2 | 1,658,691 | 12.3% |
| `backbone/stem/Concat` | 1 | 1,525,056 | 11.3% |
| `backbone/C3` blocks | 64 | 1,232,613 | 9.1% |
| `Slice` (stem focus) | 4 | 514,235 | 3.8% |
| `stem/conv/act/Mul` | 1 | 477,426 | 3.5% |
| `ReduceMax` (class score reduction) | 1 | 300,431 | 2.2% |

### Per-op cost — pose detector (1,383,780 cycles)

| Op | Layers | Cycles | Share |
|---|---:|---:|---:|
| `Conv` | 29 | 544,877 | 39.4% |
| `backbone1/1/Relu` | 1 | 145,829 | 10.5% |
| `backbone1/2/act/Relu` | 1 | 110,641 | 8.0% |
| `backbone1/3/act/Relu` | 1 | 105,960 | 7.7% |
| `backbone1/4/act/Relu` | 1 | 102,915 | 7.4% |

### Per-op cost — pose landmarks (1,208,659 cycles)

| Op | Layers | Cycles | Share |
|---|---:|---:|---:|
| `Conv` | 65 | 311,313 | 25.8% |
| `backbone1/1/Relu` | 1 | 174,006 | 14.4% |
| `detector/Resize` | 3 | 89,965 | 7.4% |
| `backbone1/3/act/Relu` | 1 | 41,218 | 3.4% |
| `detector/Add` | 6 | 39,875 | 3.3% |

### Per-op cost — QuickSRNet-Medium (2,418,643 cycles)

| Op | Layers | Cycles | Share |
|---|---:|---:|---:|
| `DepthToSpace` (pixel shuffle) | 1 | **1,796,483** | **74.3%** |
| output marshalling | 1 | 383,634 | 15.9% |
| `Clip` (output) | 1 | 51,760 | 2.1% |
| input marshalling | 1 | 36,204 | 1.5% |

Three quarters of super-resolution is the single final pixel-shuffle. That is
inherent to the architecture, not a placement mistake, and it is the reason
super-resolution is **gated on crop size** rather than always-on
(`super_res.min_bbox_area_frac`).

---

## 4. CPU/NPU split, re-derived

The prototype's document claimed that `Resize_bilinear`, `NonMaxSuppression`,
and a heatmap `ArgMax` were "killed from the NPU graph" as bottlenecks, and
attributed specific cycle counts to each. The real profile contradicts every
part of that:

| Prototype claim | What the profile shows |
|---|---|
| `NonMaxSuppression` cost 812,410 cycles on the HTP and was removed | **No `NonMaxSuppression` node exists** in the YOLO-X graph, at any cost. The export ends at decoded `boxes`/`scores`/`class_idx`. |
| Heatmap `ArgMax` cost 610,332 cycles and was moved to CPU | **No `ArgMax` and no heatmaps exist.** The landmark stage regresses 25 points directly. |
| `Resize_bilinear` cost 1,203,880 cycles and was moved to CPU | The only `Resize` is 3 layers inside the *landmark* graph, costing 89,965 cycles (7.4%), and it runs **on the NPU**. |
| Pose was "96% supported, ArgMax falls back to CPU" | **100% of all 744 layers are on the NPU.** Nothing falls back. |

The actual split is therefore not a bottleneck-eviction story at all:

- **NPU (HTP):** all four graphs, end to end, 100% of layers.
- **CPU:** frame marshalling only — capture, letterbox, layout/channel
  conversion, box de-letterboxing, NMS, BlazePose SSD anchor decode, the
  25→COCO landmark remap, tracking, and the triage scorer. These are on the
  CPU because they are not tensor work, not because they were profiled and
  evicted.

NMS in particular is CPU work because **the graph does not contain it** — the
artifact hands back 8400 candidate boxes and suppression is the caller's job.
That is a different reason from the one the prototype gave, and it means there
is no NPU cycle budget to reclaim by moving it.

### Latency budget per trainee lane

From the measured figures, one trainee on one camera costs:

| Stage | NPU time | When |
|---|---:|---|
| YOLO-X detection | 2,748 µs | once per frame, all trainees |
| Pose detector | 421 µs | per trainee |
| Pose landmarks | 425 µs | per trainee |
| QuickSRNet | 632 µs | per trainee, **only** if the crop is < 2% of frame area |

So a frame with N trainees costs roughly `2748 + N × 846` µs of NPU time, plus
`632 µs` for each distant trainee. At 15 FPS (66.7 ms budget) that leaves
substantial headroom, which is what the VLM cadence gate is protecting: the VLM
is orders of magnitude more expensive than the entire vision head, so it is
sampled at ~1 Hz for at most `vlm.top_k` already-flagged trainees.

**These are single-graph, single-camera figures from AI Hub, not a measured
end-to-end frame time.** Wall-clock latency under sustained multi-camera load,
including CPU post-processing and thermal behaviour over a shift, has not been
measured. See [VALIDATION.md](VALIDATION.md).

### CPU and GPU comparison jobs

`models/argus_cu_jobs.json` records a CPU job (`jg9d8ell5`) and a GPU job
(`jgd20lxe5`). Their results are not summarised here because the summary file
does not contain them; re-fetch with `qai-hub` before quoting any
NPU-vs-CPU-vs-GPU ratio.

---

## 5. What the placeholder calibration actually did

The consequence is not abstract. `pose_detector.bin`'s `box_scores_*` outputs
were calibrated to a range that is **entirely non-positive**: scale 5.5528 with
zero_point 255. The largest representable logit is exactly 0.0, and the next
quantization step down is −5.55.

After the sigmoid, that means the detector has effectively **two** confidence
values:

| Quantized value | Logit | Score |
|---|---:|---:|
| 255 | 0.00 | 0.500 |
| 254 | −5.55 | 0.004 |

Any threshold in (0.004, 0.5] selects exactly the same anchors. Pose-detector
confidence carries almost no information in this build. `argus.default.toml`
sets `pose.detector_score_threshold = 0.4` and says why; the fix is to
re-quantize with real footage, not to tune the threshold.

This is pinned by `tests/test_artifacts.py::test_pose_detector_confidence_range_is_degenerate`
so it cannot be quietly forgotten, and it will start failing the moment a
re-export fixes it.

---

## 6. Runtime notes

**QAIRT version skew.** These artifacts were compiled with QAIRT 2.45.0. The
QAIRT install found on the development host is **2.32.6.250402**. A QNN context
binary is tied to the runtime that produced it, so the two `.bin` files are
expected to fail backend initialisation against 2.32.6. `argus doctor` reports
this explicitly and `QnnNpuBackend` refuses to load rather than surfacing an
opaque QNN error. Install matching QAIRT, or recompile the pose artifacts.

**Loading a context binary.** `onnxruntime-qnn` cannot open a `.bin` directly.
Argus wraps it in an ONNX `EPContext` node carrying the I/O contract from
`metadata.json` (`engine.context_binary_mode = "epcontext"`), or shells out to
the SDK's `qnn-net-run` (`"netrun"`, diagnostic-grade — one subprocess per
inference).

**ONNX Runtime QDQ fusion.** At `extended` optimization and above, ORT fails to
load the QuickSRNet artifact with `two nodes with same node name
(/model/cnn/0/Conv)`. The artifact is valid — `onnx.checker` passes with
`full_check=True`, all 56 node names are unique, and it loads cleanly at
`basic`. The collision is created by ORT's own QDQ fusion pass. Argus retries
one level down and prints which of the two it is, because "re-export the model"
would be the wrong remedy.

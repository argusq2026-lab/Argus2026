"""Generate the phone detector's contract sidecar and cross-platform parity fixture.

The phone runs YOLO-X on its own NPU, decoded in Kotlin. Two artifacts keep that
honest, both generated here from the *real* model file rather than written by
hand:

**The sidecar** (`<model>.json`, staged next to the model on the device) is the
phone's equivalent of the old artifact `metadata.json`: input/output names,
shapes, dtypes, and — critically — the output quantization parameters, read
from the graph's final QuantizeLinear initializers. The app refuses to open a
model whose session I/O disagrees with its sidecar, and never hardcodes a
scale: a re-quantized model ships with a regenerated sidecar or does not load.
(The alternative — baking scale 4.4157 into Kotlin — is exactly how a silent
garbage-decode would ship the day the model is re-quantized.)

**The parity fixture** (`tests/data/yolox_parity.json`) freezes what the model
computes on one deterministic input: a procedural pattern (`(x*7 + y*13 + c*31)
% 256`, exactly reproducible in Kotlin without shipping an image), run through
onnxruntime's CPU execution provider, with the raw quantized outputs and the
reference-decoded detections recorded. The phone's instrumented test builds the
same input, runs the same model on the Hexagon, and must land within the stated
tolerances — which is the "same function on both platforms" proof, achievable
with no camera and no real footage.

The reference decode below mirrors the deleted PC implementation
(`git show d3bd15e:src/argus/vision/detect.py` and `:src/argus/vision/nms.py`)
line for line; those files are the provenance for every constant here.

Usage:
    python scripts/gen_yolox_fixture.py <path/to/yolox.onnx>
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "data" / "yolox_parity.json"

#: Mirrors the old configs/argus.default.toml [detector] section.
SCORE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.45
PERSON_CLASS_INDEX = 0
MAX_DETECTIONS = 64
#: Low-threshold decode also recorded, so the fixture stays discriminating even
#: though the procedural pattern excites few high-confidence anchors.
LOOSE_THRESHOLD = 0.05


def pattern_input() -> np.ndarray:
    """The deterministic NCHW uint8 input, identical in Kotlin by construction."""
    c, y, x = np.meshgrid(
        np.arange(3), np.arange(640), np.arange(640), indexing="ij"
    )
    return ((x * 7 + y * 13 + c * 31) % 256).astype(np.uint8)[np.newaxis, ...]


def quant_params(model: onnx.ModelProto) -> dict:
    """Output scale/zero-point, read from the final QuantizeLinear nodes."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    outputs = {o.name for o in graph.output}
    params: dict[str, dict] = {}
    for node in graph.node:
        for out in node.output:
            if out in outputs and node.op_type == "QuantizeLinear":
                params[out] = {
                    "scale": float(numpy_helper.to_array(inits[node.input[1]])),
                    "zero_point": int(numpy_helper.to_array(inits[node.input[2]])),
                }
    return params


def nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Verbatim from d3bd15e:src/argus/vision/nms.py."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def reference_decode(
    boxes_q: np.ndarray,
    scores_q: np.ndarray,
    class_q: np.ndarray,
    qp: dict,
    score_threshold: float,
) -> list[dict]:
    """Mirrors d3bd15e:src/argus/vision/detect.py's post-processing, in canvas space."""
    boxes = (boxes_q[0].astype(np.float32) - qp["boxes"]["zero_point"]) * qp["boxes"]["scale"]
    scores = (scores_q[0].astype(np.float32) - qp["scores"]["zero_point"]) * qp["scores"]["scale"]
    classes = class_q[0].astype(np.int32)

    keep_mask = (scores >= score_threshold) & (classes == PERSON_CLASS_INDEX)
    if not keep_mask.any():
        return []
    boxes, scores, classes = boxes[keep_mask], scores[keep_mask], classes[keep_mask]
    keep = nms_cpu(boxes, scores, NMS_IOU_THRESHOLD)[:MAX_DETECTIONS]
    return [
        {
            "bbox_canvas_xyxy": [round(float(v), 3) for v in boxes[i]],
            "score": round(float(scores[i]), 5),
            "class_index": int(classes[i]),
        }
        for i in keep
    ]


def main() -> int:
    model_path = Path(sys.argv[1]).resolve()
    model = onnx.load(str(model_path))
    qp = quant_params(model)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    outs = {o.name: o for o in session.get_outputs()}
    assert inp.shape == [1, 3, 640, 640] and "uint8" in inp.type, (inp.name, inp.shape, inp.type)
    assert set(outs) == {"boxes", "scores", "class_idx"}, set(outs)

    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    ext_data = model_path.with_suffix(".data")

    # ---- sidecar -----------------------------------------------------------
    sidecar = {
        "sidecar_version": 1,
        "model_file": model_path.name,
        "model_sha256": sha256,
        "source": "AI Hub job jgo8m0l1p source model (yolox-onnx-w8a8-clean)",
        "input": {"name": inp.name, "shape": [1, 3, 640, 640], "dtype": "uint8",
                  "layout": "NCHW", "channel_order": "RGB", "letterbox_pad_value": 114},
        "outputs": {
            "boxes": {"shape": [1, 8400, 4], "dtype": "uint8", "space": "canvas_xyxy", **qp["boxes"]},
            "scores": {"shape": [1, 8400], "dtype": "uint8", **qp["scores"]},
            "class_idx": {"shape": [1, 8400], "dtype": "uint8", "person_class_index": PERSON_CLASS_INDEX},
        },
        "postprocess": {
            "score_threshold": SCORE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS,
        },
    }
    sidecar_path = model_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    # ---- parity fixture ----------------------------------------------------
    tensor = pattern_input()
    outputs = session.run(None, {inp.name: tensor})
    by_name = dict(zip([o.name for o in session.get_outputs()], outputs))
    boxes_q, scores_q, class_q = by_name["boxes"], by_name["scores"], by_name["class_idx"]

    scores_f = (scores_q[0].astype(np.float32) - qp["scores"]["zero_point"]) * qp["scores"]["scale"]
    top = np.argsort(scores_f)[::-1][:50]

    fixture = {
        "fixture_version": 1,
        "model_sha256": sha256,
        "input_pattern": "(x*7 + y*13 + c*31) % 256, NCHW uint8 (1,3,640,640)",
        "quant": qp,
        "raw_outputs_b64": {
            "boxes": base64.b64encode(boxes_q.tobytes()).decode(),
            "scores": base64.b64encode(scores_q.tobytes()).decode(),
            "class_idx": base64.b64encode(class_q.tobytes()).decode(),
        },
        "top50_anchors": [
            {"index": int(i), "score_q": int(scores_q[0][i]),
             "class": int(class_q[0][i]),
             "boxes_q": [int(v) for v in boxes_q[0][i]]}
            for i in top
        ],
        "decoded_loose": reference_decode(boxes_q, scores_q, class_q, qp, LOOSE_THRESHOLD),
        "decoded_default": reference_decode(boxes_q, scores_q, class_q, qp, SCORE_THRESHOLD),
        "tolerances": {
            # Measured on a Galaxy S25 Ultra (Hexagon v79, QAIRT 2.42 via
            # qnn-runtime): worst score delta 11 LSB, worst box delta 6 LSB
            # across the top-50 anchors. That is genuine accumulation — the HTP
            # does fixed-point arithmetic layer by layer while the CPU EP
            # simulates QDQ in fp32 — not a bug in either. Bounds sit ~50%
            # above the measurement; in dequantized units the score bound is
            # 16 * 0.00381 ≈ 0.061, which is material against a 0.35 threshold
            # and is therefore *stated*, not hidden: a borderline detection can
            # legitimately differ between the phone and a CPU replay.
            "note": "measured S25U worst: score 11 LSB, box 6 LSB; bounded ~1.5x above",
            "score_q_lsb": 16,
            "boxes_q_lsb": 9,
        },
    }
    # ---- crafted decode cases ---------------------------------------------
    # The procedural pattern excites no person-class anchors (its top scorer is
    # a non-person class), so on its own the fixture would never exercise the
    # decode chain past the class filter. These cases craft raw quantized
    # tensors with *known* person anchors — overlapping boxes for NMS, a
    # sub-threshold one, a non-person distractor — and freeze what the
    # reference decode says about them. The Kotlin decode must agree exactly:
    # no model involved, so any disagreement is arithmetic, not quantization.
    def crafted_case(name: str, anchors: list[dict]) -> dict:
        boxes = np.zeros((1, 8400, 4), dtype=np.uint8)
        scores = np.zeros((1, 8400), dtype=np.uint8)
        classes = np.full((1, 8400), 7, dtype=np.uint8)  # background: some non-person class
        bs, bz = qp["boxes"]["scale"], qp["boxes"]["zero_point"]
        ss, sz = qp["scores"]["scale"], qp["scores"]["zero_point"]
        for a in anchors:
            i = a["index"]
            boxes[0, i] = [int(round(v / bs + bz)) for v in a["bbox_canvas"]]
            scores[0, i] = int(round(a["score"] / ss + sz))
            classes[0, i] = a["class"]
        return {
            "name": name,
            "anchors_in": anchors,
            "raw_b64": {
                "boxes": base64.b64encode(boxes.tobytes()).decode(),
                "scores": base64.b64encode(scores.tobytes()).decode(),
                "class_idx": base64.b64encode(classes.tobytes()).decode(),
            },
            "expected": reference_decode(boxes, scores, class_q=classes, qp=qp,
                                         score_threshold=SCORE_THRESHOLD),
        }

    fixture["decode_cases"] = [
        crafted_case("single_person", [
            {"index": 100, "bbox_canvas": [100.0, 80.0, 300.0, 560.0], "score": 0.9, "class": 0},
        ]),
        crafted_case("nms_suppresses_the_overlap", [
            {"index": 200, "bbox_canvas": [100.0, 80.0, 300.0, 560.0], "score": 0.9, "class": 0},
            {"index": 201, "bbox_canvas": [110.0, 90.0, 310.0, 570.0], "score": 0.6, "class": 0},
            {"index": 202, "bbox_canvas": [400.0, 100.0, 560.0, 500.0], "score": 0.7, "class": 0},
        ]),
        crafted_case("threshold_and_class_filter", [
            {"index": 300, "bbox_canvas": [100.0, 80.0, 300.0, 560.0], "score": 0.34, "class": 0},
            {"index": 301, "bbox_canvas": [350.0, 80.0, 550.0, 560.0], "score": 0.9, "class": 16},
            {"index": 302, "bbox_canvas": [50.0, 50.0, 250.0, 500.0], "score": 0.5, "class": 0},
        ]),
        crafted_case("nothing_above_threshold", [
            {"index": 400, "bbox_canvas": [100.0, 80.0, 300.0, 560.0], "score": 0.2, "class": 0},
        ]),
    ]

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

    print(f"sidecar : {sidecar_path}")
    print(f"fixture : {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    print(f"model   : sha256 {sha256[:16]}…  (+ external data: {ext_data.exists()})")
    print(f"loose decode: {len(fixture['decoded_loose'])} boxes, "
          f"default decode: {len(fixture['decoded_default'])} boxes, "
          f"top score {scores_f[top[0]]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

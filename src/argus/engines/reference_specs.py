"""Built-in copies of the four artifact contracts.

`models/` is gitignored, so a fresh clone has no `metadata.json` to read — but
mock mode, the unit tests, and `argus doctor` all still need to know the tensor
contracts. These constants are transcribed from the real AI Hub exports.

They are not a second source of truth: `tests/test_artifacts.py` asserts each
one is byte-for-byte equal to the corresponding `metadata.json` whenever
`models/` is provisioned, so a re-export that changes a contract fails CI here
rather than at runtime on the floor.

Provenance — AI Hub jobs recorded in `models/argus_jobs.json`, QAIRT 2.45.0:
  yolox                        compile jgo8m0l1p / profile jpv74olzp
  quicksrnetmedium             compile jpey21785 / profile jg9d81zm5
  pose_detector.bin            profile j56wvj8ng
  pose_landmark_detector.bin   profile jp3683zmp
"""

from __future__ import annotations

from argus.engines.metadata import ModelSpec, TensorSpec

QAIRT_VERSION = "2.45.0.260326154327"

#: YOLO-X, w8a8 ONNX. NCHW uint8 in; THREE separate outputs, already decoded to
#: xyxy in the 640x640 letterboxed space. NMS is not in the graph.
YOLOX = ModelSpec(
    model_id="yolox",
    model_name="Yolo-X",
    runtime="onnx",
    precision="w8a8",
    file_name="yolox.onnx",
    inputs=(
        TensorSpec("image", (1, 3, 640, 640), "uint8", 0.003921568859368563, 0),
    ),
    outputs=(
        TensorSpec("boxes", (1, 8400, 4), "uint8", 4.415665626525879, 51),
        TensorSpec("scores", (1, 8400), "uint8", 0.0038128476589918137, 0),
        TensorSpec("class_idx", (1, 8400), "uint8", None, None),
    ),
    qairt_version=QAIRT_VERSION,
)

#: BlazePose stage 1 — SSD person detector. NHWC uint8, two anchor heads
#: (16x16x2 = 512 at stride 8, 8x8x6 = 384 at stride 16).
POSE_DETECTOR = ModelSpec(
    model_id="mediapipe_pose",
    model_name="MediaPipe-Pose-Estimation",
    runtime="qnn_context_binary",
    precision="w8a8",
    file_name="pose_detector.bin",
    inputs=(
        TensorSpec("image", (1, 128, 128, 3), "uint8", 0.003921568859368563, 0),
    ),
    outputs=(
        TensorSpec("box_scores_1", (1, 512, 1), "uint8", 5.552783966064453, 255),
        TensorSpec("box_coords_1", (1, 512, 12), "uint8", 0.7927474975585938, 89),
        TensorSpec("box_scores_2", (1, 384, 1), "uint8", 4.833410263061523, 254),
        TensorSpec("box_coords_2", (1, 384, 12), "uint8", 1.2209054231643677, 99),
    ),
    qairt_version=QAIRT_VERSION,
)

#: BlazePose stage 2 — landmark regressor. NOTE the 256x256 input: the two pose
#: binaries do NOT share an input size. 25 landmarks (upper-body BlazePose),
#: each (x, y, z, visibility), normalised to the ROI. There are no heatmaps.
POSE_LANDMARK = ModelSpec(
    model_id="mediapipe_pose",
    model_name="MediaPipe-Pose-Estimation",
    runtime="qnn_context_binary",
    precision="w8a8",
    file_name="pose_landmark_detector.bin",
    inputs=(
        TensorSpec("image", (1, 256, 256, 3), "uint8", 0.003921568859368563, 0),
    ),
    outputs=(
        TensorSpec("scores", (1,), "uint8", 0.00390625, 0),
        TensorSpec("landmarks", (1, 25, 4), "uint8", 0.006140740588307381, 112),
    ),
    qairt_version=QAIRT_VERSION,
)

#: QuickSRNet-Medium, w8a8 ONNX. NCHW, fixed 4x: 128 -> 512. A crop must be
#: resized to exactly 128x128 first; there is no dynamic axis.
QUICKSRNET = ModelSpec(
    model_id="quicksrnetmedium",
    model_name="QuickSRNetMedium",
    runtime="onnx",
    precision="w8a8",
    file_name="quicksrnetmedium.onnx",
    inputs=(
        TensorSpec("image", (1, 3, 128, 128), "uint8", 0.003921568859368563, 0),
    ),
    outputs=(
        TensorSpec(
            "upscaled_image", (1, 3, 512, 512), "uint8", 0.003921568859368563, 0
        ),
    ),
    qairt_version=QAIRT_VERSION,
)

#: Role name -> reference contract. Roles are what the rest of Argus asks for.
BY_ROLE: dict[str, ModelSpec] = {
    "detector": YOLOX,
    "pose_detector": POSE_DETECTOR,
    "pose_landmark": POSE_LANDMARK,
    "super_res": QUICKSRNET,
}

#: Role name -> the metadata.json config key that describes it.
METADATA_KEY_BY_ROLE: dict[str, str] = {
    "detector": "detector_metadata",
    "pose_detector": "pose_metadata",
    "pose_landmark": "pose_metadata",
    "super_res": "super_res_metadata",
}

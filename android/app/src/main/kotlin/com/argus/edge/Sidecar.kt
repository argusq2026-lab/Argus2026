package com.argus.edge

import org.json.JSONObject
import java.io.File

/**
 * The model's contract, read from its sidecar — never hardcoded.
 *
 * `scripts/gen_yolox_fixture.py` generates `<model>.json` from the real ONNX
 * graph: I/O names, shapes, dtypes, and the output quantization parameters
 * extracted from the final QuantizeLinear initializers. The app refuses to
 * open a model without its sidecar, and `QnnDetector` refuses one whose live
 * session I/O disagrees with it.
 *
 * This is the old repo's `metadata.json` discipline carried to the phone. The
 * alternative — baking `scale 4.4157 / zero_point 51` into Kotlin — works
 * until the day the model is re-quantized, at which point every box decodes to
 * plausible-looking garbage with no error anywhere. A contract mismatch must
 * fail at load, naming the tensor, exactly as `argus.engines.metadata` did.
 */
data class QuantParams(val scale: Double, val zeroPoint: Int) {
    fun dequantize(q: Int): Float = ((q - zeroPoint) * scale).toFloat()
}

data class ModelSidecar(
    val modelFile: String,
    val modelSha256: String,
    val inputName: String,
    val inputShape: List<Int>,
    val boxes: QuantParams,
    val scores: QuantParams,
    val personClassIndex: Int,
    val scoreThreshold: Double,
    val nmsIouThreshold: Double,
    val maxDetections: Int,
    val letterboxPadValue: Int,
) {
    val inputSize: Int get() = inputShape[2]

    companion object {
        fun load(file: File): ModelSidecar {
            if (!file.isFile) {
                throw NpuUnavailableException(
                    "model sidecar not found at $file — generate it with " +
                        "scripts/gen_yolox_fixture.py and stage it next to the model; " +
                        "a model without its contract does not load"
                )
            }
            val root = JSONObject(file.readText())
            if (root.getInt("sidecar_version") != 1) {
                throw NpuUnavailableException("unsupported sidecar_version in $file")
            }
            val input = root.getJSONObject("input")
            val outputs = root.getJSONObject("outputs")
            val post = root.getJSONObject("postprocess")

            fun quant(name: String): QuantParams {
                val o = outputs.getJSONObject(name)
                return QuantParams(o.getDouble("scale"), o.getInt("zero_point"))
            }

            val shape = input.getJSONArray("shape").let { a -> List(a.length()) { a.getInt(it) } }
            if (shape.size != 4 || shape[1] != 3 || shape[2] != shape[3]) {
                throw NpuUnavailableException("sidecar declares non-NCHW-square input $shape")
            }
            return ModelSidecar(
                modelFile = root.getString("model_file"),
                modelSha256 = root.getString("model_sha256"),
                inputName = input.getString("name"),
                inputShape = shape,
                boxes = quant("boxes"),
                scores = quant("scores"),
                personClassIndex = outputs.getJSONObject("class_idx").getInt("person_class_index"),
                scoreThreshold = post.getDouble("score_threshold"),
                nmsIouThreshold = post.getDouble("nms_iou_threshold"),
                maxDetections = post.getInt("max_detections"),
                letterboxPadValue = input.optInt("letterbox_pad_value", 114),
            )
        }
    }
}

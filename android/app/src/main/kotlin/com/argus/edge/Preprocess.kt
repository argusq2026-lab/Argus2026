package com.argus.edge

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint

/** Everything needed to map a box predicted in canvas space back to source pixels. */
data class LetterboxInfo(val scale: Float, val left: Int, val top: Int)

/**
 * Aspect-preserving resize into a centred square canvas, mirroring the deleted
 * PC implementation (`git show d3bd15e:src/argus/vision/preprocess.py`).
 *
 * ## Known contract gap, recorded rather than papered over
 *
 * The PC reference resized with `cv2.resize(INTER_LINEAR)` — a fixed-point
 * kernel. This implementation resamples through Android's `Canvas.drawBitmap`
 * with bilinear filtering, which is **not guaranteed to agree bit for bit**.
 * The detector is w8a8, so a one-LSB input difference can move an anchor
 * across a quantization step. The parity fixture therefore feeds the tensor
 * *directly* (a procedural pattern, no resampling), which pins the model and
 * decode exactly while leaving the resampler an explicitly open comparison —
 * to be measured against real footage, not assumed away.
 *
 * The geometry either side of the resample — scale, centring offsets,
 * [undoLetterbox] — is exact arithmetic and is what the unit tests pin.
 */
const val YOLOX_PAD_VALUE: Int = 114

fun letterboxGeometry(srcWidth: Int, srcHeight: Int, size: Int): LetterboxInfo {
    require(srcWidth > 0 && srcHeight > 0) { "source must be non-empty" }
    val scale = size.toFloat() / maxOf(srcWidth, srcHeight)
    val nw = maxOf(Math.round(srcWidth * scale), 1)
    val nh = maxOf(Math.round(srcHeight * scale), 1)
    return LetterboxInfo(scale = scale, left = (size - nw) / 2, top = (size - nh) / 2)
}

/** Map a box from letterboxed canvas space back to source-frame pixels. */
fun undoLetterbox(
    x0: Float, y0: Float, x1: Float, y1: Float, info: LetterboxInfo,
): FloatArray {
    val s = maxOf(info.scale, 1e-9f)
    return floatArrayOf(
        (x0 - info.left) / s,
        (y0 - info.top) / s,
        (x1 - info.left) / s,
        (y1 - info.top) / s,
    )
}

/** Rotate an upright-pending camera bitmap by the ImageProxy's rotation. */
fun rotateUpright(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return bitmap
    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

/** Letterbox a source bitmap onto a `size`×`size` canvas padded with grey 114. */
fun letterbox(src: Bitmap, size: Int, padValue: Int = YOLOX_PAD_VALUE): Pair<Bitmap, LetterboxInfo> {
    val info = letterboxGeometry(src.width, src.height, size)
    val nw = maxOf(Math.round(src.width * info.scale), 1)
    val nh = maxOf(Math.round(src.height * info.scale), 1)

    val canvasBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(canvasBitmap)
    canvas.drawColor(Color.rgb(padValue, padValue, padValue))
    val scaled = Bitmap.createScaledBitmap(src, nw, nh, true)
    canvas.drawBitmap(scaled, info.left.toFloat(), info.top.toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))
    if (scaled !== src) scaled.recycle()
    return canvasBitmap to info
}

/**
 * ARGB bitmap → NCHW RGB uint8 tensor bytes, the layout the artifact declares.
 *
 * The graph dequantizes with scale 1/255 internally, so raw 0–255 bytes go in —
 * no normalization here, matching `to_nchw_uint8` on the PC.
 */
fun toNchwRgbBytes(bitmap: Bitmap): ByteArray {
    val size = bitmap.width
    require(bitmap.height == size) { "expected a square letterboxed bitmap" }
    val pixels = IntArray(size * size)
    bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

    val plane = size * size
    val out = ByteArray(3 * plane)
    for (i in 0 until plane) {
        val p = pixels[i]
        out[i] = ((p shr 16) and 0xFF).toByte()             // R
        out[plane + i] = ((p shr 8) and 0xFF).toByte()      // G
        out[2 * plane + i] = (p and 0xFF).toByte()          // B
    }
    return out
}

/** The parity fixture's procedural input: `(x*7 + y*13 + c*31) % 256`, NCHW. */
fun patternInput(size: Int): ByteArray {
    val plane = size * size
    val out = ByteArray(3 * plane)
    for (c in 0 until 3) {
        var i = c * plane
        for (y in 0 until size) {
            for (x in 0 until size) {
                out[i++] = ((x * 7 + y * 13 + c * 31) % 256).toByte()
            }
        }
    }
    return out
}

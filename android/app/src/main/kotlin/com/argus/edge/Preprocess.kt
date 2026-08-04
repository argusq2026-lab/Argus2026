package com.argus.edge

/** Everything needed to map a box predicted in canvas space back to source pixels. */
data class LetterboxInfo(val scale: Float, val left: Int, val top: Int)

/**
 * Aspect-preserving resize into a centred square canvas, mirroring
 * `argus.vision.preprocess.letterbox`.
 *
 * ## Known contract gap, deliberately recorded rather than papered over
 *
 * The PC implementation resizes with `cv2.resize(..., INTER_LINEAR)`, which
 * uses OpenCV's fixed-point kernel. The bilinear filter below is the textbook
 * float version. **These are not guaranteed to agree bit for bit**, and the
 * detector is w8a8, so a one-LSB input difference can move an anchor across a
 * quantisation step and change what the two platforms detect -- for reasons
 * that have nothing to do with the model.
 *
 * That matters because phone-vs-PC comparison is how the port gets validated at
 * all. It is an open decision, not a solved problem:
 *
 *  - mandate OpenCV-Android here so the two are exact, at the cost of ~10 MB of
 *    native library for one function, or
 *  - accept a tolerance and measure the detection-level impact deliberately, on
 *    real footage, before trusting any phone-vs-PC accuracy comparison.
 *
 * Until that is decided, treat any phone/PC detection difference as unexplained
 * rather than attributing it to the model or the quantisation.
 *
 * The geometry either side of the resample -- the scale factor, the centring
 * offsets, and [undoLetterbox] -- is exact integer and float arithmetic with no
 * resampling, so that part *is* portable and is what the unit tests pin.
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

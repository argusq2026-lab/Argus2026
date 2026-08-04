package com.argus.edge

/**
 * Which detected person is *this station's trainee*.
 *
 * A station watches one trainee (`docs/PROTOCOL.md`: one connection, one
 * `trainee_id`, and that id is the triage key an alert is dispatched against).
 * The camera does not respect that: a partner spotting, someone walking past,
 * or a neighbouring station's trainee all land in frame as person detections.
 *
 * Picking the highest-scoring box each frame — the obvious thing, and what this
 * app did first — lets the reported subject flip between two people as scores
 * jitter. Nothing downstream can detect that: the laptop keeps one rolling
 * history per `trainee_id`, so two people's boxes and keypoints get blended
 * into one trainee's stillness, fall and off-task features, and every resulting
 * number stays plausible. `ARCHITECTURE.md` made the same point about the
 * multi-camera tracker it replaced — "a swapped ID sends an instructor to the
 * wrong person and resets the real one's triage history mid-incident". The
 * network moved; the correctness concern did not.
 *
 * The rule here is deliberately simple, and simple is the point — this is not a
 * re-identification tracker, and it should not grow into one:
 *
 *  - **Area, not score.** The trainee at this station is the nearest person to
 *    a phone placed in front of them, so the largest box wins. Confidence
 *    ranks how sure the detector is that something is a person, which is a
 *    different question entirely.
 *  - **Hysteresis.** Once chosen, a subject is kept as long as a detection
 *    still overlaps its last box, even if another person is momentarily
 *    larger. Switching requires the incumbent to be gone for
 *    [missesBeforeSwitch] consecutive frames.
 *
 * Switches are counted, not hidden: a station whose subject keeps changing is
 * misplaced, and [switches] is what says so on screen.
 */
class SubjectTracker(
    private val iouKeepThreshold: Float = 0.3f,
    private val missesBeforeSwitch: Int = 8,
) {
    private var lastBox: FloatArray? = null
    private var misses = 0

    var switches: Long = 0
        private set

    /** Choose this frame's subject, or null when nobody is detected. */
    fun select(detections: List<Detection>): Detection? {
        if (detections.isEmpty()) {
            misses += 1
            if (misses >= missesBeforeSwitch) lastBox = null
            return null
        }

        val incumbent = lastBox?.let { prev ->
            detections.maxByOrNull { iouOf(prev, it) }
                ?.takeIf { iouOf(prev, it) >= iouKeepThreshold }
        }
        if (incumbent != null) {
            misses = 0
            lastBox = boxOf(incumbent)
            return incumbent
        }

        // Incumbent not found this frame: tolerate a short gap (occlusion, a
        // missed detection) before conceding the subject has actually left.
        if (lastBox != null) {
            misses += 1
            if (misses < missesBeforeSwitch) return null
        }

        val chosen = detections.maxByOrNull { area(it) } ?: return null
        if (lastBox != null) switches += 1
        lastBox = boxOf(chosen)
        misses = 0
        return chosen
    }

    fun reset() {
        lastBox = null
        misses = 0
    }

    private fun boxOf(d: Detection) = floatArrayOf(d.x0, d.y0, d.x1, d.y1)
    private fun area(d: Detection) = maxOf(d.x1 - d.x0, 0f) * maxOf(d.y1 - d.y0, 0f)

    private fun iouOf(a: FloatArray, d: Detection): Float {
        val xx0 = maxOf(a[0], d.x0); val yy0 = maxOf(a[1], d.y0)
        val xx1 = minOf(a[2], d.x1); val yy1 = minOf(a[3], d.y1)
        val inter = maxOf(0f, xx1 - xx0) * maxOf(0f, yy1 - yy0)
        val areaA = maxOf(a[2] - a[0], 0f) * maxOf(a[3] - a[1], 0f)
        return inter / maxOf(areaA + area(d) - inter, 1e-9f)
    }
}

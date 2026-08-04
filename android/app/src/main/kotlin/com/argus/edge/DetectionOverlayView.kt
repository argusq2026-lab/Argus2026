package com.argus.edge

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Draws detection boxes over the camera preview.
 *
 * Boxes arrive in upright source-frame pixels; the preview renders that frame
 * with PreviewView's FILL_CENTER (center-crop). This view applies the same
 * transform — uniform max-scale plus centring — so a box lands on the person
 * it was detected on. If the two ever disagree visibly, suspect rotation
 * handling before suspecting the model.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var pose: PoseResult? = null
    private var frameWidth = 1
    private var frameHeight = 1

    /** COCO skeleton edges among the joints the 25-point export can supply. */
    private val skeletonEdges = arrayOf(
        5 to 6,                     // shoulders
        5 to 7, 7 to 9,             // left arm
        6 to 8, 8 to 10,            // right arm
        5 to 11, 6 to 12, 11 to 12, // torso
    )
    private val bonePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.rgb(255, 200, 60)
    }
    private val jointPaint = Paint().apply { color = Color.rgb(255, 200, 60) }
    private val jointDimPaint = Paint().apply { color = Color.argb(90, 255, 200, 60) }

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.rgb(60, 220, 100)
    }
    private val textPaint = Paint().apply {
        color = Color.rgb(60, 220, 100)
        textSize = 42f
        typeface = android.graphics.Typeface.MONOSPACE
    }
    private val textBackground = Paint().apply { color = Color.argb(160, 0, 0, 0) }

    fun update(
        detections: List<Detection>, pose: PoseResult?, frameWidth: Int, frameHeight: Int,
    ) {
        this.detections = detections
        this.pose = pose
        this.frameWidth = maxOf(frameWidth, 1)
        this.frameHeight = maxOf(frameHeight, 1)
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty() && pose == null) return

        // FILL_CENTER: uniform scale to cover, then center the overflow.
        val scale = maxOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val dx = (width - frameWidth * scale) / 2f
        val dy = (height - frameHeight * scale) / 2f

        pose?.let { p ->
            for ((a, b) in skeletonEdges) {
                if (p.keypointsConf[a] < 0.3f || p.keypointsConf[b] < 0.3f) continue
                canvas.drawLine(
                    p.keypointsXy[a * 2] * scale + dx, p.keypointsXy[a * 2 + 1] * scale + dy,
                    p.keypointsXy[b * 2] * scale + dx, p.keypointsXy[b * 2 + 1] * scale + dy,
                    bonePaint,
                )
            }
            for (k in 0 until 17) {
                val conf = p.keypointsConf[k]
                if (conf <= 0f) continue // unmapped joints never draw
                canvas.drawCircle(
                    p.keypointsXy[k * 2] * scale + dx, p.keypointsXy[k * 2 + 1] * scale + dy,
                    10f, if (conf >= 0.3f) jointPaint else jointDimPaint,
                )
            }
        }

        for (det in detections) {
            val x0 = det.x0 * scale + dx
            val y0 = det.y0 * scale + dy
            val x1 = det.x1 * scale + dx
            val y1 = det.y1 * scale + dy
            canvas.drawRect(x0, y0, x1, y1, boxPaint)

            val label = "person %.2f".format(det.score)
            val tw = textPaint.measureText(label)
            canvas.drawRect(x0, y0 - 52f, x0 + tw + 16f, y0, textBackground)
            canvas.drawText(label, x0 + 8f, y0 - 12f, textPaint)
        }
    }
}

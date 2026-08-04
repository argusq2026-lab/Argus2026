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
    private var frameWidth = 1
    private var frameHeight = 1

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

    fun update(detections: List<Detection>, frameWidth: Int, frameHeight: Int) {
        this.detections = detections
        this.frameWidth = maxOf(frameWidth, 1)
        this.frameHeight = maxOf(frameHeight, 1)
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        // FILL_CENTER: uniform scale to cover, then center the overflow.
        val scale = maxOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val dx = (width - frameWidth * scale) / 2f
        val dy = (height - frameHeight * scale) / 2f

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

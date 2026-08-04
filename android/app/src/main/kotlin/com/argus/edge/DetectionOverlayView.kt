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
    private var subject: Detection? = null
    private var subjectPose: PoseResult? = null
    private var otherPoses: List<PoseResult> = emptyList()
    private var frameWidth = 1
    private var frameHeight = 1

    /**
     * COCO skeleton edges among the joints the 25-point export can supply.
     * Knees and ankles (13-16) have no source landmark, so no leg edges exist
     * to draw — their absence on screen is accurate, not a rendering bug.
     */
    private val skeletonEdges = arrayOf(
        0 to 1, 0 to 2, 1 to 3, 2 to 4, // face: nose-eyes-ears
        5 to 6,                        // shoulders
        5 to 7, 7 to 9,                // left arm
        6 to 8, 8 to 10,               // right arm
        5 to 11, 6 to 12, 11 to 12,    // torso
    )
    private val bonePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 9f
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
        color = Color.rgb(255, 190, 40)
    }
    private val jointPaint = Paint().apply {
        color = Color.rgb(255, 240, 120); isAntiAlias = true
    }
    private val jointOutline = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 3f
        color = Color.argb(200, 40, 30, 0); isAntiAlias = true
    }
    private val jointDimPaint = Paint().apply {
        color = Color.argb(110, 255, 190, 40); isAntiAlias = true
    }

    // Bystanders: landmarked so the operator can see who else is in frame while
    // placing the phone, drawn muted because only the subject is ever reported.
    private val bystanderBoxPaint = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 3f
        color = Color.argb(140, 120, 170, 255)
    }
    private val bystanderBonePaint = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND; isAntiAlias = true
        color = Color.argb(150, 120, 170, 255)
    }
    private val bystanderJointPaint = Paint().apply {
        color = Color.argb(170, 150, 190, 255); isAntiAlias = true
    }
    private val bystanderTextPaint = Paint().apply {
        color = Color.argb(180, 120, 170, 255)
        textSize = 36f
        typeface = android.graphics.Typeface.MONOSPACE
    }

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
        detections: List<Detection>,
        subject: Detection?,
        subjectPose: PoseResult?,
        otherPoses: List<PoseResult>,
        frameWidth: Int,
        frameHeight: Int,
    ) {
        this.detections = detections
        this.subject = subject
        this.subjectPose = subjectPose
        this.otherPoses = otherPoses
        this.frameWidth = maxOf(frameWidth, 1)
        this.frameHeight = maxOf(frameHeight, 1)
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty() && subjectPose == null && otherPoses.isEmpty()) return

        // FILL_CENTER: uniform scale to cover, then center the overflow.
        val scale = maxOf(width.toFloat() / frameWidth, height.toFloat() / frameHeight)
        val dx = (width - frameWidth * scale) / 2f
        val dy = (height - frameHeight * scale) / 2f

        // Bystanders first and dimmed, so the subject reads on top of them.
        for (p in otherPoses) drawSkeleton(canvas, p, scale, dx, dy, primary = false)
        subjectPose?.let { drawSkeleton(canvas, it, scale, dx, dy, primary = true) }

        for (det in detections) {
            val isSubject = det === subject
            val x0 = det.x0 * scale + dx
            val y0 = det.y0 * scale + dy
            val x1 = det.x1 * scale + dx
            val y1 = det.y1 * scale + dy
            canvas.drawRect(x0, y0, x1, y1, if (isSubject) boxPaint else bystanderBoxPaint)

            val label = if (isSubject) "subject %.2f".format(det.score)
                        else "person %.2f".format(det.score)
            val paint = if (isSubject) textPaint else bystanderTextPaint
            val tw = paint.measureText(label)
            canvas.drawRect(x0, y0 - 52f, x0 + tw + 16f, y0, textBackground)
            canvas.drawText(label, x0 + 8f, y0 - 12f, paint)
        }
    }

    private fun drawSkeleton(
        canvas: Canvas, p: PoseResult, scale: Float, dx: Float, dy: Float, primary: Boolean,
    ) {
        run {
            val bonePaint = if (primary) this.bonePaint else bystanderBonePaint
            val jointPaint = if (primary) this.jointPaint else bystanderJointPaint
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
                val x = p.keypointsXy[k * 2] * scale + dx
                val y = p.keypointsXy[k * 2 + 1] * scale + dy
                if (conf >= 0.3f) {
                    canvas.drawCircle(x, y, if (primary) 13f else 8f, jointPaint)
                    if (primary) canvas.drawCircle(x, y, 13f, jointOutline)
                } else if (primary) {
                    canvas.drawCircle(x, y, 9f, jointDimPaint)
                }
            }
        }
    }
}

package com.argus.edge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

/**
 * Subject stability, which is a correctness property rather than a nicety.
 *
 * The laptop keeps one rolling history per `trainee_id`. If the phone silently
 * swaps which person it reports, two people's boxes and keypoints blend into
 * one trainee's features and every resulting score stays plausible — the exact
 * failure mode `ARCHITECTURE.md` described for the multi-camera tracker this
 * design replaced.
 */
class SubjectTrackerTest {

    private fun box(x0: Float, y0: Float, x1: Float, y1: Float, score: Float = 0.9f) =
        Detection(x0, y0, x1, y1, score)

    @Test
    fun `picks the largest person, not the most confident`() {
        val tracker = SubjectTracker()
        val near = box(100f, 100f, 400f, 900f, score = 0.55f)   // big, less certain
        val far = box(600f, 300f, 700f, 600f, score = 0.98f)    // small, very certain
        // Confidence ranks "is this a person"; size ranks "is this the trainee
        // standing in front of this station's phone".
        assertSame(near, tracker.select(listOf(far, near)))
    }

    @Test
    fun `keeps the subject when a larger person walks into frame`() {
        val tracker = SubjectTracker()
        val trainee = box(100f, 100f, 400f, 900f)
        assertSame(trainee, tracker.select(listOf(trainee)))

        // A spotter steps in closer for several frames. The incumbent still
        // overlaps its last box, so it holds.
        val spotter = box(420f, 50f, 900f, 1000f)
        repeat(20) {
            assertSame("subject flipped to the bystander", trainee, tracker.select(listOf(trainee, spotter)))
        }
        assertEquals(0L, tracker.switches)
    }

    @Test
    fun `tolerates a short detection gap without switching`() {
        val tracker = SubjectTracker()
        val trainee = box(100f, 100f, 400f, 900f)
        val other = box(500f, 100f, 800f, 900f)
        tracker.select(listOf(trainee))

        // A few frames where only the other person is detected: an occlusion or
        // a missed detection must not hand the station to someone else.
        repeat(4) { assertNull(tracker.select(listOf(other))) }
        assertEquals(0L, tracker.switches)

        // The trainee comes back and is reacquired without a switch.
        assertSame(trainee, tracker.select(listOf(trainee, other)))
        assertEquals(0L, tracker.switches)
    }

    @Test
    fun `switches, and counts it, once the subject is really gone`() {
        val tracker = SubjectTracker(missesBeforeSwitch = 3)
        val trainee = box(100f, 100f, 400f, 900f)
        val other = box(500f, 100f, 800f, 900f)
        tracker.select(listOf(trainee))

        repeat(2) { assertNull(tracker.select(listOf(other))) }
        val now = tracker.select(listOf(other))
        assertSame(other, now)
        assertEquals("a real handover must be counted, not hidden", 1L, tracker.switches)
    }

    @Test
    fun `an empty frame is not a subject`() {
        val tracker = SubjectTracker()
        assertNull(tracker.select(emptyList()))
        assertEquals(0L, tracker.switches)
    }

    @Test
    fun `reset forgets the incumbent so a restart is clean`() {
        val tracker = SubjectTracker()
        val a = box(100f, 100f, 400f, 900f)
        val b = box(500f, 100f, 900f, 1000f)
        tracker.select(listOf(a))
        tracker.reset()
        // With no incumbent the largest wins immediately, and it is not counted
        // as a switch because there was nothing to switch away from.
        assertSame(b, tracker.select(listOf(a, b)))
        assertEquals(0L, tracker.switches)
    }

    @Test
    fun `a moving subject is followed frame to frame`() {
        val tracker = SubjectTracker()
        var x = 100f
        var first: Detection? = null
        repeat(30) {
            val d = box(x, 100f, x + 300f, 900f)
            val chosen = tracker.select(listOf(d, box(1200f, 100f, 1600f, 950f)))
            if (first == null) first = chosen
            assertNotNull(chosen)
            x += 12f   // walks across frame; each step still overlaps the last
        }
        assertEquals("walking must not look like a handover", 0L, tracker.switches)
    }
}

package com.argus.edge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RepCounterTest {

    private fun counter() = RepCounter(contractedDeg = 90.0, extendedDeg = 160.0)

    @Test
    fun `a full cycle counts one rep`() {
        val c = counter()
        c.update(170.0) // extended (start)
        c.update(60.0)  // contracted
        c.update(170.0) // extended again -- one full cycle
        assertEquals(1, c.count)
    }

    @Test
    fun `never leaving extended counts nothing`() {
        val c = counter()
        repeat(5) { c.update(175.0) }
        assertEquals(0, c.count)
    }

    @Test
    fun `dipping without reaching the contracted threshold counts nothing`() {
        val c = counter()
        c.update(170.0)
        c.update(120.0) // dips, but not past contractedDeg=90
        c.update(170.0)
        assertEquals(0, c.count)
    }

    @Test
    fun `jitter around one threshold does not double count`() {
        val c = counter()
        c.update(170.0)
        c.update(89.0)
        c.update(91.0) // wobble back above contractedDeg, still below extendedDeg
        c.update(88.0)
        c.update(170.0) // one clean return to extended
        assertEquals(1, c.count)
    }

    @Test
    fun `multiple cycles count multiple reps`() {
        val c = counter()
        repeat(3) {
            c.update(170.0)
            c.update(60.0)
            c.update(170.0)
        }
        assertEquals(3, c.count)
    }

    @Test
    fun `a NaN reading is ignored, not treated as a signal`() {
        val c = counter()
        assertFalse(c.everUpdated)
        c.update(Double.NaN)
        assertFalse("a NaN reading should not mark the counter as having seen a signal", c.everUpdated)
        assertEquals(0, c.count)
    }

    @Test
    fun `everUpdated flips on the first real reading`() {
        val c = counter()
        c.update(170.0)
        assertTrue(c.everUpdated)
    }

    @Test
    fun `reset clears count, phase and everUpdated`() {
        val c = counter()
        c.update(170.0)
        c.update(60.0)
        c.update(170.0)
        assertEquals(1, c.count)
        c.reset()
        assertEquals(0, c.count)
        assertFalse(c.everUpdated)
        // Confirms phase was cleared too: a fresh contraction-then-extension
        // after reset must count as a normal first cycle, not skip because
        // the counter thought it was already "contracted".
        c.update(170.0)
        c.update(60.0)
        c.update(170.0)
        assertEquals(1, c.count)
    }

    @Test
    fun `constructor rejects a contracted threshold at or above extended`() {
        try {
            RepCounter(contractedDeg = 160.0, extendedDeg = 160.0)
            throw AssertionError("expected an IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }
}

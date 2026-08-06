package com.argus.edge

/**
 * Counts reps from one cyclic joint angle -- a debounced peak/valley
 * crossing, not a phase classifier or a form judgement. A rep counts once
 * per full cycle regardless of quality: `rep_count` is display-only on the
 * wire (docs/PROTOCOL.md), the scorer never reads it, so this counts effort,
 * not correctness -- a `knee_angle_out_of_range` rep still counts.
 *
 * Hysteresis between [contractedDeg] and [extendedDeg] is deliberate: a
 * single threshold double-counts on the jitter around it. A rep is only
 * counted on the crossing back past `extendedDeg` after a confirmed dip past
 * `contractedDeg`, so a partial dip that never reaches `contractedDeg`
 * counts nothing, and an idle trainee who never leaves `extendedDeg` counts
 * nothing either.
 */
class RepCounter(private val contractedDeg: Double, private val extendedDeg: Double) {

    init {
        require(contractedDeg < extendedDeg) {
            "contractedDeg ($contractedDeg) must be less than extendedDeg ($extendedDeg)"
        }
    }

    private var contracted = false

    var count = 0
        private set

    /**
     * Whether [update] has ever received a readable angle. `count == 0`
     * means two different things -- "no signal yet" and "signal seen, zero
     * full cycles so far" -- and only this field tells them apart; callers
     * use it to send no `rep_count` at all for the former (docs/PROTOCOL.md
     * treats an absent field as unknown, not zero).
     */
    var everUpdated = false
        private set

    /** Feed one frame's angle. A `NaN` or otherwise unreadable angle should not be passed at all. */
    fun update(angleDeg: Double) {
        if (angleDeg.isNaN()) return
        everUpdated = true
        if (!contracted && angleDeg <= contractedDeg) {
            contracted = true
        } else if (contracted && angleDeg >= extendedDeg) {
            contracted = false
            count += 1
        }
    }

    fun reset() {
        contracted = false
        count = 0
        everUpdated = false
    }
}

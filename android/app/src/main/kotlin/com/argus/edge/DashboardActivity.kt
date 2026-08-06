package com.argus.edge

import android.content.Intent
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * The app's launcher screen: one tile per domain this triage engine could
 * watch. Fitness is the one that is actually built -- it opens [MainActivity]
 * unchanged. The others exist to say where this is headed, honestly, rather
 * than to pretend they already work: tapping one explains exactly what it
 * would take (see `coming_soon_body`), the same gap `docs/ADDING_AN_EXERCISE.md`
 * already documents for fitness's own exercises.
 *
 * Deliberately a second plain Activity, not a Fragment or a Navigation-Component
 * destination: the app had zero navigation infrastructure before this, and a
 * second Activity is the smallest addition consistent with the single-Activity,
 * single-layout style [MainActivity] already uses.
 */
class DashboardActivity : AppCompatActivity() {

    private data class Tile(
        val badgeId: Int,
        val containerId: Int,
        val color: Int,
        val titleResId: Int,
        val onClick: () -> Unit,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dashboard)

        val tiles = listOf(
            Tile(
                R.id.tileFitnessBadge, R.id.tileFitness, ContextCompat.getColor(this, R.color.live),
                R.string.tile_fitness_title,
            ) { startActivity(Intent(this, MainActivity::class.java)) },
            Tile(
                R.id.tileNursingBadge, R.id.tileNursing, ContextCompat.getColor(this, R.color.tile_nursing),
                R.string.tile_nursing_title,
            ) { showComingSoon(R.string.tile_nursing_title) },
            Tile(
                R.id.tileLabBadge, R.id.tileLab, ContextCompat.getColor(this, R.color.tile_lab),
                R.string.tile_lab_title,
            ) { showComingSoon(R.string.tile_lab_title) },
            Tile(
                R.id.tileWeldingBadge, R.id.tileWelding, ContextCompat.getColor(this, R.color.tile_welding),
                R.string.tile_welding_title,
            ) { showComingSoon(R.string.tile_welding_title) },
        )

        for (tile in tiles) {
            // One shared oval drawable (bg_badge.xml), re-colored per tile
            // rather than shipping four near-identical XML shapes that differ
            // only by a hardcoded color.
            val badge = findViewById<View>(tile.badgeId)
            (badge.background.mutate() as GradientDrawable).setColor(tile.color)
            findViewById<View>(tile.containerId).setOnClickListener { tile.onClick() }
        }
    }

    private fun showComingSoon(titleResId: Int) {
        val name = getString(titleResId)
        AlertDialog.Builder(this)
            .setTitle(R.string.coming_soon_title)
            .setMessage(getString(R.string.coming_soon_body, name))
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }
}

"""
demo.py — Canned demo loop for video capture (no camera, voice, or inference).

Plays the full visual UX from boot through a failed scan and a successful
identification, then loops. The presenter says "scan" out loud, timed to the
display transitions.

Sequence (loops until Ctrl-C):
  1. Boot splash with vine animation
  2. READY screen
  3. Pause WAIT_BEFORE_FIRST_SCAN  — presenter says "scan"
  4. SCANNING ...
  5. MOVE CLOSER (abstention — first scan "fails")
  6. Pause WAIT_BETWEEN_SCANS      — presenter resets pose, says "scan" again
  7. SCANNING ...
  8. Wild Blueberry committed identification
  9. Pause WAIT_ON_SUCCESS         — viewers read the result

Run on the Pi from the inference/ directory:
    python demo.py
"""

import signal
import time

from pipeline.convergence import ForagerResult
from pipeline.display     import EinkDisplay


# ── Timing (seconds) — tweak for video pacing ────────────────────────────────
WAIT_BEFORE_FIRST_SCAN = 5
WAIT_BETWEEN_SCANS     = 3
WAIT_ON_SUCCESS        = 8
WAIT_BETWEEN_LOOPS     = 2

# Splash vine animation steps. Each show_splash() triggers a 4-gray refresh
# (~2.5s on the 3.7" Waveshare) so the refresh itself paces the animation.
VINE_STEPS = (0.0, 0.35, 0.7, 1.0)


# ── Canned results ────────────────────────────────────────────────────────────
FAIL_RESULT = ForagerResult(
    domain="berry",
    species="unknown",
    scientific_name="Unknown",
    confidence=0.0,
    safety="UNKNOWN",
    lookalike="N/A",
    key_diff="No confident identification",
    low_confidence=True,
    expert_model="none",
)

SUCCESS_RESULT = ForagerResult(
    domain="berry",
    species="blueberry_wild",
    scientific_name="Vaccinium angustifolium",
    confidence=0.91,
    safety="SAFE",
    lookalike="Canada moonseed",
    key_diff="Moonseed has one crescent seed, no true drupelets",
    low_confidence=False,
    expert_model="berry_expert",
)


def run_iteration(display: EinkDisplay) -> None:
    print("\n[demo] Boot splash (vine animation) ...")
    for progress in VINE_STEPS:
        display.show_splash(vine_progress=progress)

    print("[demo] READY ...")
    display.show_ready()

    print(f"[demo] Hold {WAIT_BEFORE_FIRST_SCAN}s — presenter says 'scan'")
    time.sleep(WAIT_BEFORE_FIRST_SCAN)

    print("[demo] SCANNING (will fail) ...")
    display.show_scanning()
    display.show(FAIL_RESULT)
    print("[demo] MOVE CLOSER displayed")

    print(f"[demo] Hold {WAIT_BETWEEN_SCANS}s — presenter resets pose, says 'scan'")
    time.sleep(WAIT_BETWEEN_SCANS)

    print("[demo] SCANNING (will succeed) ...")
    display.show_scanning()
    display.show(SUCCESS_RESULT)
    print(f"[demo] Wild Blueberry identified — hold {WAIT_ON_SUCCESS}s")
    time.sleep(WAIT_ON_SUCCESS)


def main() -> None:
    running = True
    def _shutdown(*_):
        nonlocal running
        print("\nShutting down ...")
        running = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("=" * 50)
    print("Walking Man — DEMO loop")
    print("  No camera, no voice, no inference.")
    print("  Ctrl-C to exit.")
    print("=" * 50)

    with EinkDisplay() as display:
        while running:
            run_iteration(display)

            # Clear the cached last-image so the next iteration's boot/scan
            # starts from a clean baseline (otherwise show_scanning() would
            # overlay on the previous blueberry result instead of doing the
            # full scanning template).
            display._last_img = None
            display._partial_count = 0

            if running:
                time.sleep(WAIT_BETWEEN_LOOPS)

    print("Done.")


if __name__ == "__main__":
    main()

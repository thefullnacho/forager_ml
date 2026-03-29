"""
main.py — Forager inference entry point (two-stage router pipeline).

Full loop:
    1. Load domain router + expert .hef models onto Hailo 8L
    2. Wait for voice trigger ("scan", "go", "identify", etc.)
    3. Capture frame from Camera Module 3
    4. Stage 1: Run domain router to classify berry/mushroom/plant
    5. Stage 2: Route to relevant expert(s), get single prediction
    6. Build ForagerResult with species metadata + safety info
    7. Render single-result layout to Waveshare 3.7" eInk display
    8. Speak result via TTS (DEADLY warnings first)
    9. Go back to step 2

Run on the Pi:
    python main.py

Optional flags:
    --models-dir   path to directory containing .hef + _classes.json files
                   (default: ./models)
    --no-voice     skip voice trigger, capture immediately on Enter key
    --no-display   skip eInk push (useful for debugging over SSH)
    --no-tts       skip spoken output
"""

import argparse
import os
import sys
import signal

from pipeline.loader      import HailoModelLoader
from pipeline.runner      import AsyncRunner
from pipeline.convergence import build_result
from pipeline.camera      import Camera
from pipeline.display     import EinkDisplay
from pipeline.voice       import VoiceTrigger, build_speech_message


DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def parse_args():
    p = argparse.ArgumentParser(description="Forager — two-stage Hailo inference")
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--no-voice",   action="store_true", help="Use Enter key instead of voice")
    p.add_argument("--no-display", action="store_true", help="Skip eInk push")
    p.add_argument("--no-tts",     action="store_true", help="Skip spoken output")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        print("\nShutting down ...")
        running = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Init subsystems ───────────────────────────────────────────────────────
    print("=" * 50)
    print("Forager — starting up (two-stage router pipeline)")
    print("=" * 50)

    print(f"\nModels dir : {args.models_dir}")

    voice   = None if args.no_voice   else VoiceTrigger()
    display = EinkDisplay()              # always construct; conditionally push

    with HailoModelLoader(args.models_dir) as loader:
        print(f"\nRouter : {loader.router.name}  ({len(loader.router.classes)} classes)")
        print(f"Experts: {list(loader.experts)}")

        runner = AsyncRunner(loader.router, loader.experts)

        with Camera() as cam:
            with display:

                print("\nReady. ", end="")
                if args.no_voice:
                    print("Press Enter to capture.")
                else:
                    print(f"Say a trigger word to capture.")

                # ── Main loop ─────────────────────────────────────────────────
                while running:
                    # Step 1: wait for trigger
                    if args.no_voice:
                        try:
                            input()
                        except EOFError:
                            break
                    else:
                        voice.wait_for_trigger()

                    if not running:
                        break

                    if not args.no_display:
                        display.show_scanning()

                    print("Capturing ...")
                    image = cam.capture()

                    print("Running inference ...")
                    domain, prediction = runner.run(image)

                    print("Building result ...")
                    result = build_result(domain, prediction)

                    # ── Log to terminal ───────────────────────────────────────
                    print("\n" + "-" * 40)
                    print(f"  Domain : {result.domain}")
                    if result.is_unknown:
                        if result.domain == "other":
                            print("  Result : NOT A FORAGING TARGET")
                        else:
                            print(f"  Result : LOW ROUTER CONFIDENCE ({result.domain} @ below threshold) — reposition")
                    elif result.low_confidence:
                        print(f"  Result : LOW CONFIDENCE ({result.expert_model})")
                    else:
                        print(f"  Species: {result.species}")
                        print(f"  Sci    : {result.scientific_name}")
                        print(f"  Conf   : {result.confidence:.1%}")
                        print(f"  Safety : {result.safety}")
                        print(f"  Expert : {result.expert_model}")
                    if result.is_deadly:
                        print("  *** DEADLY SPECIES DETECTED ***")
                    print("-" * 40 + "\n")

                    # ── Push to eInk ──────────────────────────────────────────
                    if not args.no_display:
                        print("Updating display ...")
                        display.show(result)

                    # ── Speak result ──────────────────────────────────────────
                    if not args.no_tts and voice:
                        voice.speak(build_speech_message(result))

                    print("\nReady. ", end="")
                    if args.no_voice:
                        print("Press Enter to capture.")
                    else:
                        print("Say a trigger word to capture.")

        runner.shutdown()

    print("Done.")


if __name__ == "__main__":
    main()

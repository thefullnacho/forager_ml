"""
runner.py — Two-stage inference: domain router → expert routing.

Stage 1: Run the domain router to classify the image into berry/mushroom/plant.
Stage 2: Route to the relevant expert(s) based on the router's prediction.

For mushroom domain, both highvalue_expert and psychedelics_expert run in
parallel and the one with higher top_confidence wins.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

from .loader import ModelHandle


# ── Domain-to-expert routing table ───────────────────────────────────────────

DOMAIN_EXPERTS: dict[str, list[str]] = {
    "berry":    ["berry_expert"],
    "mushroom": ["highvalue_expert", "psychedelics_expert"],
    "plant":    ["highvalue_expert"],
    # "other" intentionally absent — triggers abstention (no experts invoked)
}

ROUTER_CONFIDENCE_THRESHOLD = 0.74


@dataclass
class RawPrediction:
    model: str
    classes: list[str]
    probabilities: np.ndarray   # shape: (num_classes,), sum ~= 1.0
    top_class: str
    top_confidence: float


def _infer_single(handle: ModelHandle, image: np.ndarray) -> RawPrediction | None:
    """
    Run one forward pass through a single model.
    Blocking — called from a thread.

    image: uint8 numpy array (H, W, C) = (224, 224, 3), RGB
    """
    batch = np.expand_dims(image.astype(np.float32), axis=0)  # (1, H, W, C) [0, 255]

    input_params  = InputVStreamParams.make(handle.network_group,
                        quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(handle.network_group,
                        quantized=False, format_type=FormatType.FLOAT32)

    with InferVStreams(handle.network_group, input_params, output_params) as pipeline:
        input_name  = list(input_params.keys())[0]
        raw_output  = pipeline.infer({input_name: batch})
        output_name = list(raw_output.keys())[0]
        logits = raw_output[output_name][0]   # remove batch dim

    # HEF outputs raw logits (exported with --no-activation).
    # Apply energy-based OOD check before softmax.
    if handle.energy_threshold is not None:
        energy = -float(np.log(np.sum(np.exp(logits))))
        if energy > handle.energy_threshold:
            print(f"  [{handle.name}] OOD rejected (energy={energy:.4f} > threshold={handle.energy_threshold:.4f})")
            return None

    # Softmax to get probabilities
    logits_shifted = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits)

    top_idx   = int(np.argmax(probs))
    top_conf  = float(probs[top_idx])
    top_class = handle.classes[top_idx]

    return RawPrediction(
        model=handle.name,
        classes=handle.classes,
        probabilities=probs,
        top_class=top_class,
        top_confidence=top_conf,
    )


class AsyncRunner:
    """
    Two-stage inference: router determines domain, then only relevant
    expert(s) are invoked.

    Usage:
        runner = AsyncRunner(router_handle, expert_handles)
        domain, prediction = runner.run(image)
    """

    def __init__(
        self,
        router: ModelHandle,
        experts: dict[str, ModelHandle],
        max_workers: int = 3,
    ):
        self._router   = router
        self._experts  = experts
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(self, image: np.ndarray) -> tuple[str, RawPrediction | None]:
        """
        Two-stage inference on a single image.

        Returns:
            (domain, prediction) where prediction is None if the router
            confidence is below threshold (unknown domain).
        """
        # ── Stage 1: Router ──────────────────────────────────────────────────
        router_pred = _infer_single(self._router, image)
        domain     = router_pred.top_class
        confidence = router_pred.top_confidence

        print(f"  [router] {domain} @ {confidence:.1%}")

        if confidence < ROUTER_CONFIDENCE_THRESHOLD:
            return domain, None

        # ── Stage 2: Route to expert(s) ──────────────────────────────────────
        expert_names = DOMAIN_EXPERTS.get(domain, [])
        if not expert_names:
            return domain, None

        # Filter to experts that are actually loaded
        targets = {
            name: self._experts[name]
            for name in expert_names
            if name in self._experts
        }

        if not targets:
            print(f"  [runner] No loaded experts for domain '{domain}'")
            return domain, None

        if len(targets) == 1:
            # Single expert — run directly
            name, handle = next(iter(targets.items()))
            try:
                pred = _infer_single(handle, image)
                if pred is None:
                    return domain, None   # OOD rejected
                return domain, pred
            except Exception as e:
                print(f"  [runner] {name} inference error: {e}")
                return domain, None

        # Multiple experts — run in parallel, pick highest confidence
        futures = {
            self._executor.submit(_infer_single, handle, image): name
            for name, handle in targets.items()
        }

        results: list[RawPrediction] = []
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                pred = future.result()
                if pred is not None:
                    results.append(pred)
            except Exception as e:
                print(f"  [runner] {model_name} inference error: {e}")

        if not results:
            return domain, None

        # Pick the expert with the highest top confidence
        best = max(results, key=lambda r: r.top_confidence)
        return domain, best

    def shutdown(self):
        self._executor.shutdown(wait=True)

"""
runner.py — Two-stage inference: domain router → expert routing.

Stage 1: Run the domain router to classify the image into berry/mushroom/plant.
Stage 2: Route to the relevant expert(s) based on the router's prediction.

For domains served by more than one expert (mushroom, plant), every expert
runs in parallel and ALL surviving predictions are returned. This layer does
not pick a winner: a less-confident DEADLY verdict must be able to veto a more
confident SAFE one, and that safety-aware resolution lives in
convergence.build_result (deadly-vetoes-safe).
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
    "plant":    ["highvalue_expert", "medicinals_expert"],
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


def _energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
    """
    Free-energy OOD score: E(x) = -T * logsumexp(logits / T).

    Lower energy = in-distribution; higher = OOD candidate. Computed with the
    numerically stable log-sum-exp (subtract the max, then add it back) so large
    logits can't overflow np.exp to inf/nan and silently disable the OOD gate.
    Must use the same temperature the threshold was calibrated with.
    """
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    m = float(np.max(scaled))
    lse = m + float(np.log(np.sum(np.exp(scaled - m))))   # stable log-sum-exp
    return float(-temperature * lse)


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
        energy = _energy_score(logits, handle.energy_temperature)
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
        domain, predictions = runner.run(image)
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

    def run(self, image: np.ndarray) -> tuple[str, list[RawPrediction]]:
        """
        Two-stage inference on a single image.

        Returns:
            (domain, predictions) where predictions is the list of expert
            predictions that survived the OOD gate. The list is empty when the
            router abstains (low confidence / OOD) or no expert produced a
            usable result. This layer does NOT select a winner — safety-aware
            resolution (deadly-vetoes-safe) happens in convergence.build_result.
        """
        # ── Stage 1: Router ──────────────────────────────────────────────────
        router_pred = _infer_single(self._router, image)
        if router_pred is None:
            # Router OOD-rejected the frame — not a foraging target.
            return "other", []
        domain     = router_pred.top_class
        confidence = router_pred.top_confidence

        print(f"  [router] {domain} @ {confidence:.1%}")

        if confidence < ROUTER_CONFIDENCE_THRESHOLD:
            return domain, []

        # ── Stage 2: Route to expert(s) ──────────────────────────────────────
        expert_names = DOMAIN_EXPERTS.get(domain, [])
        if not expert_names:
            return domain, []

        # Filter to experts that are actually loaded
        targets = {
            name: self._experts[name]
            for name in expert_names
            if name in self._experts
        }

        if not targets:
            print(f"  [runner] No loaded experts for domain '{domain}'")
            return domain, []

        if len(targets) == 1:
            # Single expert — run directly
            name, handle = next(iter(targets.items()))
            try:
                pred = _infer_single(handle, image)
                return domain, ([pred] if pred is not None else [])  # [] = OOD rejected
            except Exception as e:
                print(f"  [runner] {name} inference error: {e}")
                return domain, []

        # Multiple experts — run all in parallel and return EVERY surviving
        # prediction. We deliberately do not pick the most confident one here:
        # a less-confident DEADLY verdict must be able to veto a confident SAFE
        # one, and that resolution needs the safety metadata in convergence.py.
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

        return domain, results

    def shutdown(self):
        self._executor.shutdown(wait=True)

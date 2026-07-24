"""Hailo-free two-stage inference over the exported ONNX (logits) models.

Mirrors inference/pipeline/{runner,convergence}.py exactly — router →
expert(s) → energy-based OOD gate → deadly-vetoes-safe resolution — but runs
on CPU via onnxruntime so it works off-device and in CI. Same decisions, same
thresholds, same safety metadata (imported, not copied).

The logits ONNX exports (``*_logits.onnx``) output raw logits, matching the HEF
``--no-activation`` export, so energy scores are computed on the same quantity
the on-device path calibrated against.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .outcome import ExpertOutcome, InferenceOutcome
from .safety_meta import SPECIES_METADATA, UNKNOWN_META, safety_of

# ── Constants mirrored from the live pipeline ────────────────────────────────
ROUTER_CONFIDENCE_THRESHOLD = 0.74      # runner.ROUTER_CONFIDENCE_THRESHOLD
LOW_CONFIDENCE_THRESHOLD = 0.50         # convergence.LOW_CONFIDENCE_THRESHOLD
DOMAIN_EXPERTS: dict[str, list[str]] = {
    "berry":    ["berry_expert"],
    "mushroom": ["highvalue_expert", "psychedelics_expert"],
    "plant":    ["highvalue_expert", "medicinals_expert"],
    # "other" intentionally absent — triggers abstention
}

# ImageNet preprocessing, matching training/scripts/benchmark_ood.py
_IMG = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(path: str) -> np.ndarray:
    """Resize(224*1.14) → CenterCrop(224) → ToTensor → ImageNet-normalize → NCHW."""
    img = Image.open(path).convert("RGB")
    resize = int(_IMG * 1.14)
    img = img.resize((resize, resize), Image.BILINEAR)
    left = (resize - _IMG) // 2
    img = img.crop((left, left, left + _IMG, left + _IMG))
    arr = np.asarray(img, dtype=np.float32) / 255.0        # HWC [0,1]
    arr = (arr - _MEAN) / _STD
    arr = np.transpose(arr, (2, 0, 1))                     # CHW
    return arr[None, :, :, :].astype(np.float32)           # NCHW


def _energy(logits: np.ndarray, temperature: float = 1.0) -> float:
    """Free energy: -T * logsumexp(logits / T). Stable log-sum-exp."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    m = float(np.max(scaled))
    lse = m + float(np.log(np.sum(np.exp(scaled - m))))
    return float(-temperature * lse)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class _Model:
    """One ONNX logits model plus its classes and energy threshold."""

    def __init__(self, name: str, onnx_dir: Path, meta_dir: Path,
                 onnx_stem: str | None = None):
        self.name = name
        # The logits ONNX file can be versioned (e.g. domain_router_v2) while the
        # metadata files keep the stable name (domain_router_*). Resolve by
        # explicit stem, else exact match, else the newest matching *_logits.onnx.
        stem = onnx_stem or name
        onnx_path = onnx_dir / f"{stem}_logits.onnx"
        if not onnx_path.exists():
            candidates = sorted(onnx_dir.glob(f"{name}*_logits.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No logits ONNX for '{name}' in {onnx_dir}")
            onnx_path = candidates[-1]
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        classes_path = meta_dir / f"{name}_classes.json"
        self.classes: list[str] = json.loads(classes_path.read_text())["classes"]

        energy_path = meta_dir / f"{name}_energy.json"
        if energy_path.exists():
            e = json.loads(energy_path.read_text())
            self.energy_threshold: float | None = e.get("threshold_p95")
            self.temperature: float = e.get("temperature", 1.0)
        else:
            self.energy_threshold = None       # router has no OOD gate
            self.temperature = 1.0

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0][0]  # logits, drop batch dim


class OnnxPipeline:
    """Two-stage router→expert inference producing InferenceOutcome objects."""

    def __init__(self, onnx_dir: str | Path, model_version: str,
                 meta_dir: str | Path = "inference/models"):
        self.onnx_dir = Path(onnx_dir)
        self.meta_dir = Path(meta_dir)
        self.model_version = model_version
        self.router = _Model("domain_router", self.onnx_dir, self.meta_dir)
        expert_names = {n for names in DOMAIN_EXPERTS.values() for n in names}
        self.experts = {n: _Model(n, self.onnx_dir, self.meta_dir)
                        for n in sorted(expert_names)}

    def _run_expert(self, model: _Model, batch: np.ndarray) -> ExpertOutcome:
        logits = model.infer(batch)
        energy = _energy(logits, model.temperature)
        ood = model.energy_threshold is not None and energy > model.energy_threshold
        probs = _softmax(logits)
        top = int(np.argmax(probs))
        return ExpertOutcome(
            expert_name=model.name,
            top_class=model.classes[top],
            top_confidence=float(probs[top]),
            energy_score=energy,
            energy_threshold=model.energy_threshold if model.energy_threshold is not None else float("nan"),
            ood_rejected=ood,
        )

    def run(self, image_path: str, ground_truth: str | None = None) -> InferenceOutcome:
        t0 = time.perf_counter()
        batch = _preprocess(image_path)

        # ── Stage 1: router ──────────────────────────────────────────────────
        r_logits = self.router.infer(batch)
        r_probs = _softmax(r_logits)
        r_top = int(np.argmax(r_probs))
        domain = self.router.classes[r_top]
        r_conf = float(r_probs[r_top])

        base = dict(
            image_ref=image_path, ground_truth=ground_truth,
            model_version=self.model_version, source="onnx_cpu",
            router_domain=domain, router_confidence=r_conf,
        )

        if r_conf < ROUTER_CONFIDENCE_THRESHOLD or domain not in DOMAIN_EXPERTS:
            return self._abstain(base, router_abstained=r_conf < ROUTER_CONFIDENCE_THRESHOLD,
                                 t0=t0, experts=[])

        # ── Stage 2: experts ─────────────────────────────────────────────────
        expert_outcomes = [self._run_expert(self.experts[n], batch)
                           for n in DOMAIN_EXPERTS[domain] if n in self.experts]
        survivors = [e for e in expert_outcomes if not e.ood_rejected]

        if not survivors:
            return self._abstain(base, router_abstained=False, t0=t0, experts=expert_outcomes)

        winner, deadly_veto = self._resolve(survivors)
        meta = SPECIES_METADATA.get(winner.top_class, UNKNOWN_META)
        total_ms = (time.perf_counter() - t0) * 1000
        return InferenceOutcome(
            **base,
            router_abstained=False,
            species=winner.top_class,
            safety_tier=meta["safety"],
            confidence=winner.top_confidence,
            low_confidence=winner.top_confidence < LOW_CONFIDENCE_THRESHOLD,
            winning_expert=winner.expert_name,
            abstained=False,
            deadly_veto=deadly_veto,
            total_ms=total_ms,
            experts=expert_outcomes,
        )

    @staticmethod
    def _resolve(survivors: list[ExpertOutcome]) -> tuple[ExpertOutcome, bool]:
        """Deadly-vetoes-safe: any DEADLY verdict beats a non-deadly one even if
        less confident. Within a tier, most confident wins. Returns (winner,
        deadly_veto) where deadly_veto marks that the safety rule changed the
        answer from what max-confidence alone would have picked."""
        deadly = [e for e in survivors if safety_of(e.top_class) == "DEADLY"]
        pool = deadly if deadly else survivors
        winner = max(pool, key=lambda e: e.top_confidence)
        max_conf = max(survivors, key=lambda e: e.top_confidence)
        deadly_veto = bool(deadly) and winner is not max_conf
        return winner, deadly_veto

    @staticmethod
    def _abstain(base: dict, router_abstained: bool, t0: float,
                 experts: list[ExpertOutcome]) -> InferenceOutcome:
        return InferenceOutcome(
            **base,
            router_abstained=router_abstained,
            species="unknown",
            safety_tier=UNKNOWN_META["safety"],
            confidence=0.0,
            low_confidence=True,
            winning_expert="none",
            abstained=True,
            deadly_veto=False,
            total_ms=(time.perf_counter() - t0) * 1000,
            experts=experts,
        )

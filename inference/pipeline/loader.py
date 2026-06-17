"""
loader.py — Load domain router + expert .hef models into a single Hailo VDevice.

Uses the Hailo scheduler (ROUND_ROBIN) so all network groups share
the Hailo 8L chip without manual time-slicing.

The router model is loaded separately from the experts so the inference
pipeline can run the two-stage flow: router first, then only relevant experts.
"""

import json
import os
from dataclasses import dataclass, field

from hailo_platform import (
    VDevice,
    HEF,
    ConfigureParams,
    HailoStreamInterface,
    HailoSchedulingAlgorithm,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


ROUTER_MODEL_NAME = "domain_router"


@dataclass
class ModelHandle:
    name: str
    network_group: object
    classes: list[str]
    energy_threshold: float | None = None  # p95 OOD rejection threshold (None = no OOD check)
    energy_temperature: float = 1.0        # must match the T used during calibration


class HailoModelLoader:
    """
    Loads the domain router and expert .hef models onto one VDevice.

    Usage:
        loader = HailoModelLoader(models_dir="/path/to/models")
        with loader:
            router  = loader.router    # ModelHandle for domain_router
            experts = loader.experts   # dict[str, ModelHandle] for experts
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._device: VDevice | None = None
        self.router: ModelHandle | None = None
        self.experts: dict[str, ModelHandle] = {}

    def __enter__(self):
        self._device = self._init_device()
        self._load_all()
        return self

    def __exit__(self, *_):
        self._device = None
        self.router = None
        self.experts = {}

    # ── Private ───────────────────────────────────────────────────────────────

    def _init_device(self) -> VDevice:
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        return VDevice(params=params)

    def _load_single(self, name: str) -> ModelHandle:
        hef_path = os.path.join(self.models_dir, f"{name}.hef")
        manifest_path = os.path.join(self.models_dir, f"{name}_classes.json")

        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF file missing: {hef_path}")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Class manifest missing: {manifest_path}")

        classes = json.load(open(manifest_path))["classes"]

        energy_threshold = None
        energy_temperature = 1.0
        energy_path = os.path.join(self.models_dir, f"{name}_energy.json")
        if os.path.exists(energy_path):
            energy_cfg = json.load(open(energy_path))
            energy_threshold = energy_cfg.get("threshold_p95")
            energy_temperature = energy_cfg.get("temperature", 1.0)

        hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self._device.configure(hef, configure_params)
        ng = network_groups[0]

        ood_str = f"  OOD threshold: {energy_threshold:.4f}" if energy_threshold is not None else "  OOD threshold: none"
        print(f"  Loaded: {name}  ({len(classes)} classes)  {ood_str}")
        return ModelHandle(name=name, network_group=ng, classes=classes,
                           energy_threshold=energy_threshold,
                           energy_temperature=energy_temperature)

    def _load_all(self):
        hef_files = sorted(
            f for f in os.listdir(self.models_dir) if f.endswith(".hef")
        )
        if not hef_files:
            raise FileNotFoundError(f"No .hef files found in {self.models_dir}")

        for hef_file in hef_files:
            name = hef_file.replace(".hef", "")
            handle = self._load_single(name)

            if name == ROUTER_MODEL_NAME:
                self.router = handle
            else:
                self.experts[name] = handle

        if self.router is None:
            raise FileNotFoundError(
                f"Router model '{ROUTER_MODEL_NAME}.hef' not found in {self.models_dir}"
            )

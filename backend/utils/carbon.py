"""
Estimation de l'empreinte carbone des étapes de la pipeline.

Formule reprise de **Green Algorithms** (Lannelongue et al. 2021,
http://calculator.green-algorithms.org) :

    energy_kWh = runtime_h × power_kW × PUE
    CO2_kg     = energy_kWh × carbon_intensity_kgCO2_per_kWh

Pour la France (mix très bas-carbone grâce au nucléaire) :
    carbon_intensity ≈ 0.052 kgCO2/kWh (RTE 2024)
    PUE typique data center ≈ 1.5

Pour les appels LLM externes (OpenAI), on utilise une estimation
basée sur la consommation observée des modèles GPT-3/4 :
    ~0.04 g CO2eq / 1000 tokens output (gpt-4o-mini estimation publique)
    ~0.01 g CO2eq / 1000 tokens input

C'est volontairement transparent et conservateur ; le but est d'avoir un ordre
de grandeur traçable dans les rapports, pas une comptabilité au gramme près.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from loguru import logger


# ─────────────────────── Constantes par défaut ─────────────────────────────

# Carbon intensity (kg CO2eq / kWh) — France 2024
DEFAULT_CARBON_INTENSITY_FR = 0.052
# UE moyenne ~0.30, Pologne ~0.75, Suède ~0.04 (pour comparaison)
DEFAULT_CARBON_INTENSITY_EU = 0.30

DEFAULT_PUE = 1.5  # data center moyen

# Puissances typiques (W)
DEFAULT_CPU_POWER_W = 65.0     # CPU desktop/laptop sous charge moyenne
DEFAULT_GPU_POWER_W_RTX3060 = 170.0
DEFAULT_GPU_POWER_W_RTX4090 = 320.0
DEFAULT_GPU_POWER_W_A10 = 150.0
DEFAULT_GPU_POWER_W_A100 = 400.0

# Empreinte par 1000 tokens LLM (g CO2eq) — estimations publiques pour gpt-4o-mini
LLM_CO2_PER_KTOKEN_IN = 0.01
LLM_CO2_PER_KTOKEN_OUT = 0.04


# ──────────────────────── Métriques agrégées ───────────────────────────────

@dataclass
class CarbonMetrics:
    """
    Métriques d'empreinte cumulées sur une période.
    Tout est exposé pour pouvoir tracer / refaire le calcul à la main si besoin.
    """

    label: str = "run"
    runtime_seconds: float = 0.0
    device: str = "cpu"  # "cpu" | "cuda"
    cpu_power_w: float = DEFAULT_CPU_POWER_W
    gpu_power_w: Optional[float] = None  # None si CPU
    pue: float = DEFAULT_PUE
    carbon_intensity_kgCO2_per_kWh: float = DEFAULT_CARBON_INTENSITY_FR
    # Tokens LLM (additionnés au fur et à mesure des appels)
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def energy_kWh(self) -> float:
        power_w = self.gpu_power_w if self.device == "cuda" and self.gpu_power_w else self.cpu_power_w
        runtime_h = self.runtime_seconds / 3600.0
        return runtime_h * (power_w / 1000.0) * self.pue

    @property
    def co2_kg_compute(self) -> float:
        return self.energy_kWh * self.carbon_intensity_kgCO2_per_kWh

    @property
    def co2_kg_llm(self) -> float:
        kt_in = self.llm_tokens_in / 1000.0
        kt_out = self.llm_tokens_out / 1000.0
        # g → kg
        return (kt_in * LLM_CO2_PER_KTOKEN_IN + kt_out * LLM_CO2_PER_KTOKEN_OUT) / 1000.0

    @property
    def co2_kg_total(self) -> float:
        return self.co2_kg_compute + self.co2_kg_llm

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "device": self.device,
            "cpu_power_w": self.cpu_power_w,
            "gpu_power_w": self.gpu_power_w,
            "pue": self.pue,
            "carbon_intensity_kgCO2_per_kWh": self.carbon_intensity_kgCO2_per_kWh,
            "llm_tokens_in": self.llm_tokens_in,
            "llm_tokens_out": self.llm_tokens_out,
            "energy_kWh": round(self.energy_kWh, 6),
            "co2_kg_compute": round(self.co2_kg_compute, 6),
            "co2_kg_llm": round(self.co2_kg_llm, 6),
            "co2_kg_total": round(self.co2_kg_total, 6),
            "co2_g_total": round(self.co2_kg_total * 1000, 3),
            "extra": self.extra,
        }


# ────────────────────────── Tracker ────────────────────────────────────────

class CarbonTracker:
    """
    Tracker simple basé sur le temps wall-clock et des puissances supposées.
    Utilisation :

        tracker = CarbonTracker(label="rag_inference", device="cpu")
        with tracker.measure():
            response = engine.answer(...)
        tracker.add_llm_tokens(input_tokens=500, output_tokens=300)
        print(tracker.metrics.to_dict())
    """

    def __init__(
        self,
        label: str = "run",
        device: str = "cpu",
        cpu_power_w: float = DEFAULT_CPU_POWER_W,
        gpu_power_w: Optional[float] = None,
        pue: float = DEFAULT_PUE,
        carbon_intensity: float = DEFAULT_CARBON_INTENSITY_FR,
    ) -> None:
        self.metrics = CarbonMetrics(
            label=label,
            device=device,
            cpu_power_w=cpu_power_w,
            gpu_power_w=gpu_power_w,
            pue=pue,
            carbon_intensity_kgCO2_per_kWh=carbon_intensity,
        )

    @contextmanager
    def measure(self) -> Iterator["CarbonTracker"]:
        t0 = time.monotonic()
        try:
            yield self
        finally:
            self.metrics.runtime_seconds += time.monotonic() - t0

    def add_llm_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.metrics.llm_tokens_in += input_tokens
        self.metrics.llm_tokens_out += output_tokens

    def log_summary(self) -> None:
        m = self.metrics
        logger.info(
            f"[Carbon] {m.label} | {m.runtime_seconds:.1f}s {m.device} | "
            f"{m.llm_tokens_in + m.llm_tokens_out} tokens | "
            f"E={m.energy_kWh*1000:.2f}Wh | "
            f"CO2={m.co2_kg_total*1000:.2f}g "
            f"(compute={m.co2_kg_compute*1000:.2f}g + llm={m.co2_kg_llm*1000:.2f}g)"
        )

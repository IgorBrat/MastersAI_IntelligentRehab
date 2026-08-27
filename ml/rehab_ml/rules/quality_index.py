from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exercises.base import ExerciseDefinition
from ..taxonomy import ClinicianGrade, ErrorType, eqi_to_grade


EPS = 1e-9

@dataclass(frozen=True)
class EqiWeights:
    """Weights of the additive performance term. Normalised on construction"""
    rom: float = 0.45
    smoothness: float = 0.20
    tempo: float = 0.15
    hold: float = 0.10
    return_to_start: float = 0.10

    def normalised(self) -> "EqiWeights":
        total = self.rom + self.smoothness + self.tempo + self.hold + self.return_to_start
        if total <= EPS:
            raise ValueError("EQI weights must sum to a positive value")
        return EqiWeights(
            rom=self.rom / total,
            smoothness=self.smoothness / total,
            tempo=self.tempo / total,
            hold=self.hold / total,
            return_to_start=self.return_to_start / total,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "rom": self.rom,
            "smoothness": self.smoothness,
            "tempo": self.tempo,
            "hold": self.hold,
            "return_to_start": self.return_to_start,
        }


@dataclass(frozen=True)
class EqiConfig:
    """Tunables of the index"""
    weights: EqiWeights = field(default_factory=EqiWeights)
    # Lowest value the compensation gate may take.
    gate_floor: float = 0.35
    # Per-error multipliers on compensation penalties
    compensation_weights: dict[ErrorType, float] = field(
        default_factory=lambda: {
            ErrorType.TRUNK_FORWARD_FLEXION: 1.3,
            ErrorType.TRUNK_LATERAL_LEAN: 1.2,
            ErrorType.SHOULDER_HIKING: 1.0,
            ErrorType.ELBOW_FLEXION_SYNERGY: 1.0,
            ErrorType.TRUNK_ROTATION: 0.8,
            ErrorType.OUT_OF_PLANE_DRIFT: 0.7,
        }
    )

    def weight_for(self, error: ErrorType) -> float:
        return self.compensation_weights.get(error, 1.0)


@dataclass(frozen=True)
class QualityIndex:
    """A computed EQI with its full decomposition"""
    value: float
    performance: float
    gate: float
    components: dict[str, float]
    penalties: dict[str, float]

    @property
    def grade(self) -> ClinicianGrade:
        return eqi_to_grade(self.value)

    @property
    def limiting_component(self) -> str:
        """Name of the weakest performance component - what to coach next"""
        if not self.components:
            return ""
        return min(self.components.items(), key=lambda kv: kv[1])[0]

    @property
    def dominant_compensation(self) -> str | None:
        """Slug of the heaviest compensation, or ``None`` if compensation is clean"""
        if not self.penalties:
            return None
        slug, penalty = max(self.penalties.items(), key=lambda kv: kv[1])
        return slug if penalty > EPS else None

    def as_dict(self) -> dict:
        return {
            "eqi": round(self.value, 2),
            "grade": int(self.grade),
            "performance": round(self.performance, 4),
            "compensation_gate": round(self.gate, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
        }


def rom_score(peak: float, baseline: float, target: float) -> float:
    """Fraction of the prescribed excursion achieved, clipped to ``[0, 1]``"""
    target_excursion = abs(target - baseline)
    if target_excursion <= EPS:
        return 1.0
    achieved = abs(peak - baseline)
    return float(np.clip(achieved / target_excursion, 0.0, 1.0))


def smoothness_score(sparc: float, floor: float, severe_deficit: float) -> float:
    """Mapping SPARC onto ``[0, 1]``"""
    if not np.isfinite(sparc):
        return 0.5
    if severe_deficit <= EPS:
        return 1.0 if sparc >= floor else 0.0
    deficit = max(0.0, floor - sparc)
    return float(np.clip(1.0 - deficit / severe_deficit, 0.0, 1.0))


def tempo_score(duration_s: float, prescribed_s: float, tolerance: float) -> float:
    """Penalise repetitions performed faster than prescribed"""
    if prescribed_s <= EPS:
        return 1.0
    minimum = prescribed_s * (1.0 - tolerance)
    if duration_s >= minimum:
        return 1.0
    return float(np.clip(duration_s / max(minimum, EPS), 0.0, 1.0))


def hold_score(hold_s: float, prescribed_s: float) -> float:
    """Fraction of the prescribed end-range hold achieved. ``1`` if none prescribed"""
    if prescribed_s <= EPS:
        return 1.0
    return float(np.clip(hold_s / prescribed_s, 0.0, 1.0))


def return_score(return_error: float, severe: float) -> float:
    """How completely the limb came back to its starting position"""
    if severe <= EPS:
        return 1.0
    return float(np.clip(1.0 - return_error / severe, 0.0, 1.0))



def compute_eqi(
    *,
    exercise: ExerciseDefinition,
    peak: float,
    baseline: float,
    sparc: float,
    duration_s: float,
    hold_s: float,
    return_error: float,
    compensation_penalties: dict[ErrorType, float],
    config: EqiConfig | None = None,
) -> QualityIndex:
    """Computing the Exercise Quality Index for one repetition"""
    cfg = config or EqiConfig()
    weights = cfg.weights.normalised()

    components = {
        "rom": rom_score(peak, baseline, exercise.rom.target),
        "smoothness": smoothness_score(
            sparc, exercise.sparc_floor, exercise.sparc_tolerance.severe
        ),
        "tempo": tempo_score(
            duration_s, exercise.tempo.total_s, exercise.tempo.speed_tolerance
        ),
        "hold": hold_score(hold_s, exercise.tempo.hold_s),
        "return_to_start": return_score(return_error, exercise.return_tolerance.severe),
    }

    performance = float(
        sum(components[name] * getattr(weights, name) for name in components)
    )

    # compensation gate: a weighted mean of the penalties
    penalties: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for error, penalty in compensation_penalties.items():
        w = cfg.weight_for(error)
        penalties[error.name.lower()] = float(np.clip(penalty, 0.0, 1.0))
        weighted_sum += w * float(np.clip(penalty, 0.0, 1.0))
        weight_total += w

    mean_penalty = weighted_sum / weight_total if weight_total > EPS else 0.0
    gate = 1.0 - (1.0 - cfg.gate_floor) * mean_penalty

    value = 100.0 * performance * gate
    return QualityIndex(
        value=float(np.clip(value, 0.0, 100.0)),
        performance=performance,
        gate=float(gate),
        components=components,
        penalties=penalties,
    )

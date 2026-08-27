from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..exercises.base import ExerciseDefinition, Tolerance
from ..taxonomy import ERROR_SPECS, ErrorType
from ..exercises.registry import as_self_paced, get_exercise

# Percentiles of the correct-execution distribution used for the three bands.
DEFAULT_PERCENTILES: tuple[float, float, float] = (85.0, 95.0, 99.0)

# criteria whose thresholds are fitted
CALIBRATABLE: frozenset[str] = frozenset(
    ERROR_SPECS[e].name_short
    for e in (
        ErrorType.TRUNK_LATERAL_LEAN,
        ErrorType.TRUNK_FORWARD_FLEXION,
        ErrorType.TRUNK_ROTATION,
        ErrorType.SHOULDER_HIKING,
        ErrorType.ELBOW_FLEXION_SYNERGY,
        ErrorType.OUT_OF_PLANE_DRIFT,
        ErrorType.INCOMPLETE_RETURN,
        ErrorType.JERKY_MOVEMENT,
    )
)

# Below this many correct examples a percentile estimate is too unstable to use, and the protocol threshold is kept instead.
MIN_SAMPLES = 20


@dataclass(frozen=True)
class Calibration:
    """Fitted thresholds, keyed by error slug"""

    tolerances: dict[str, Tolerance]
    n_reference: int
    percentiles: tuple[float, float, float]
    # Slugs (short names) that kept their protocol threshold and why
    skipped: dict[str, str]

    def as_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "percentiles": list(self.percentiles),
            "tolerances": {
                slug: [round(t.mild, 4), round(t.moderate, 4), round(t.severe, 4)]
                for slug, t in self.tolerances.items()
            },
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        return cls(
            tolerances={
                slug: Tolerance(*bands) for slug, bands in payload["tolerances"].items()
            },
            n_reference=int(payload.get("n_reference", 0)),
            percentiles=tuple(payload.get("percentiles", DEFAULT_PERCENTILES)),
            skipped=dict(payload.get("skipped", {})),
        )


def fit(
    samples,
    exercise: ExerciseDefinition,
    *,
    percentiles: tuple[float, float, float] = DEFAULT_PERCENTILES,
    min_samples: int = MIN_SAMPLES,
) -> Calibration:
    """Fitting thresholds from correctly-labelled repetitions"""
    reference = [s for s in samples if s.correct > 0.5 and s.meta.get("rule_measured")]
    protocol = _protocol_tolerances(exercise)
    tolerances: dict[str, Tolerance] = dict(protocol)
    skipped: dict[str, str] = {}

    if len(reference) < min_samples:
        return Calibration(
            tolerances=tolerances,
            n_reference=len(reference),
            percentiles=percentiles,
            skipped={s: f"only {len(reference)} correct examples" for s in protocol},
        )

    for slug in protocol:
        if slug not in CALIBRATABLE:
            skipped[slug] = "defined by the prescription, not by observation"
            continue
        values = np.array(
            [s.meta["rule_measured"][slug] for s in reference if slug in s.meta["rule_measured"]],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size < min_samples:
            skipped[slug] = f"only {values.size} measurements"
            continue

        bands = [float(np.percentile(values, p)) for p in percentiles]

        for i in range(1, 3):
            bands[i] = max(bands[i], bands[i - 1] * 1.2 + 1e-6)
        tolerances[slug] = Tolerance(*bands)

    return Calibration(
        tolerances=tolerances,
        n_reference=len(reference),
        percentiles=percentiles,
        skipped=skipped,
    )


def fit_corpus(
    dataset,
    *,
    percentiles: tuple[float, float, float] = DEFAULT_PERCENTILES,
    min_samples: int = MIN_SAMPLES,
    self_paced: bool = False,
) -> dict[str, Calibration]:
    """Fitting one `Calibration` per exercise across an entire corpus"""
    out: dict[str, Calibration] = {}
    for exercise_id in sorted(dataset.exercise_ids):
        exercise = get_exercise(exercise_id)
        if self_paced:
            exercise = as_self_paced(exercise)
        subset = dataset.filter(exercise_id=exercise_id)
        out[exercise_id] = fit(
            subset.samples,
            exercise,
            percentiles=percentiles,
            min_samples=min_samples,
        )
    return out


def save_calibrations(
    path: str | Path, calibrations: dict[str, Calibration], provenance: dict
) -> Path:
    """Writing fitted thresholds plus the provenance needed to interpret them"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rehab-ml/thresholds@1",
        "provenance": provenance,
        "exercises": {ex_id: cal.as_dict() for ex_id, cal in calibrations.items()},
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_calibrations(path: str | Path) -> tuple[dict[str, Calibration], dict]:
    """Reading a threshold file, returning ``(calibrations_by_exercise, provenance)``"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "rehab-ml/thresholds@1":
        raise ValueError(
            f"{path}: not a rehab-ml threshold file (format={payload.get('format')!r}). Produce one with `rehab-ml calibrate`."
        )
    calibrations = {
        ex_id: Calibration.from_dict(body) for ex_id, body in payload["exercises"].items()
    }
    return calibrations, dict(payload.get("provenance", {}))


def _protocol_tolerances(exercise: ExerciseDefinition) -> dict[str, Tolerance]:
    """The exercise's own thresholds, keyed by slug"""
    out = {ERROR_SPECS[limit.error].name_short: limit.tolerance for limit in exercise.compensations}
    out[ERROR_SPECS[ErrorType.INCOMPLETE_RETURN].name_short] = exercise.return_tolerance
    out[ERROR_SPECS[ErrorType.JERKY_MOVEMENT].name_short] = exercise.sparc_tolerance
    return out


def verdict(sample, calibration: Calibration) -> bool:
    """Re-derive "was this repetition correct" under calibrated thresholds"""
    measured = sample.meta.get("rule_measured") or {}
    for slug, value in measured.items():
        tolerance = calibration.tolerances.get(slug)
        if tolerance is None:
            continue
        if value > tolerance.mild:
            return False
    return True


def apply_to_exercise(
    exercise: ExerciseDefinition, calibration: Calibration
) -> ExerciseDefinition:
    """Returning a copy of ``exercise`` carrying the fitted thresholds"""
    comps = tuple(
        replace(limit, tolerance=calibration.tolerances[ERROR_SPECS[limit.error].name_short])
        if ERROR_SPECS[limit.error].name_short in calibration.tolerances
        else limit
        for limit in exercise.compensations
    )
    updated = replace(exercise, compensations=comps)
    for error, field_name in (
        (ErrorType.INCOMPLETE_RETURN, "return_tolerance"),
        (ErrorType.JERKY_MOVEMENT, "sparc_tolerance"),
    ):
        slug = ERROR_SPECS[error].name_short
        if slug in calibration.tolerances:
            updated = replace(updated, **{field_name: calibration.tolerances[slug]})
    return updated

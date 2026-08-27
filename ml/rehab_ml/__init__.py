from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .schema import Joint, N_JOINTS, PoseSequence, Side
from .taxonomy import ClinicianGrade, ErrorType, Severity

__version__ = "0.1.0"

_LAZY: dict[str, str] = {
    "RuleEngine": "rehab_ml.rules.engine",
    "RepetitionAssessment": "rehab_ml.rules.engine",
    "SessionAssessment": "rehab_ml.rules.engine",
    "compute_angles": "rehab_ml.features.angles",
    "compute_eqi": "rehab_ml.rules.quality_index",
    "QualityIndex": "rehab_ml.rules.quality_index",
    "get_exercise": "rehab_ml.exercises.registry",
    "all_exercises": "rehab_ml.exercises.registry",
    "Prescription": "rehab_ml.exercises.registry",
    "ExerciseProgram": "rehab_ml.exercises.registry",
    "preprocess": "rehab_ml.pose.normalize",
    "segment_repetitions": "rehab_ml.segmentation.repetition",
}

if TYPE_CHECKING:
    from .exercises.registry import ExerciseProgram, Prescription, all_exercises, get_exercise
    from .features.angles import compute_angles
    from .pose.normalize import preprocess
    from .rules.engine import RepetitionAssessment, RuleEngine, SessionAssessment
    from .rules.quality_index import QualityIndex, compute_eqi
    from .segmentation.repetition import segment_repetitions


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "ClinicianGrade",
    "ErrorType",
    "Joint",
    "N_JOINTS",
    "PoseSequence",
    "Severity",
    "Side",
    *_LAZY,
]

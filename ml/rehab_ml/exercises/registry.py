from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..schema import Side
from .base import ExerciseDefinition, RomTarget, Tolerance
from .stroke_upper_limb import PROGRESSION_ORDER, PROTOCOL

_REGISTRY: dict[str, ExerciseDefinition] = {ex.id: ex for ex in PROTOCOL}


def get_exercise(exercise_id: str) -> ExerciseDefinition:
    try:
        return _REGISTRY[exercise_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown exercise {exercise_id!r}; known: {known}") from None


def all_exercises() -> tuple[ExerciseDefinition, ...]:
    return PROTOCOL


def exercise_ids() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def progression_order() -> tuple[str, ...]:
    return PROGRESSION_ORDER


def as_self_paced(exercise: ExerciseDefinition) -> ExerciseDefinition:
    """Dropping the prescribed tempo and end-range hold from an exercise"""
    tempo = replace(exercise.tempo, hold_s=0.0, speed_tolerance=1.0)
    return replace(exercise, tempo=tempo)


@dataclass
class Prescription:
    """One patient's current parameters for one exercise"""
    exercise_id: str
    side: Side
    rom_target: float
    sets: int
    reps: int
    tolerance_scale: float = 1.0
    note_uk: str = ""

    @classmethod
    def default_for(
        cls, exercise_id: str, side: Side, *, tolerance_scale: float = 1.0
    ) -> "Prescription":
        ex = get_exercise(exercise_id)
        return cls(
            exercise_id=exercise_id,
            side=side,
            rom_target=ex.rom.target,
            sets=ex.default_sets,
            reps=ex.default_reps,
            tolerance_scale=tolerance_scale,
        )

    def resolve(self) -> ExerciseDefinition:
        """Materialising the effective definition for this patient"""
        ex = get_exercise(self.exercise_id)
        rom: RomTarget = ex.rom.with_target(self.rom_target)

        if self.tolerance_scale == 1.0:
            return replace(ex, rom=rom)

        scale = float(self.tolerance_scale)
        comps = tuple(
            replace(limit, tolerance=limit.tolerance.scaled(scale))
            for limit in ex.compensations
        )
        return replace(
            ex,
            rom=rom,
            compensations=comps,
            return_tolerance=ex.return_tolerance.scaled(scale),
            hold_tolerance=ex.hold_tolerance.scaled(scale),
            sparc_tolerance=ex.sparc_tolerance.scaled(scale),
        )


@dataclass
class ExerciseProgram:
    """An ordered set of prescriptions - the patient's home program"""

    patient_id: str
    affected_side: Side
    prescriptions: list[Prescription] = field(default_factory=list)

    @classmethod
    def starter(
        cls, patient_id: str, affected_side: Side, *, n_exercises: int = 3
    ) -> "ExerciseProgram":
        """Building an initial program from the easiest ``n_exercises``"""
        ids = PROGRESSION_ORDER[: max(1, n_exercises)]
        return cls(
            patient_id=patient_id,
            affected_side=affected_side,
            prescriptions=[Prescription.default_for(i, affected_side) for i in ids],
        )

    def find(self, exercise_id: str) -> Prescription | None:
        for p in self.prescriptions:
            if p.exercise_id == exercise_id:
                return p
        return None

    def total_reps(self) -> int:
        return sum(p.sets * p.reps for p in self.prescriptions)

    def unlocked_ids(self) -> tuple[str, ...]:
        return tuple(p.exercise_id for p in self.prescriptions)

    def next_unlockable(self) -> str | None:
        """The next exercise in progression order not yet in the program"""
        current = set(self.unlocked_ids())
        for exercise_id in PROGRESSION_ORDER:
            if exercise_id not in current:
                return exercise_id
        return None


__all__ = [
    "ExerciseDefinition",
    "ExerciseProgram",
    "Prescription",
    "Tolerance",
    "all_exercises",
    "exercise_ids",
    "get_exercise",
    "progression_order",
]

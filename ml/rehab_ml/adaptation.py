from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .exercises.base import ExerciseDefinition
from .exercises.registry import Prescription, get_exercise, progression_order
from .taxonomy import EQI_GRADE_BOUNDARIES, ERROR_SPECS, safety_errors


ADVANCE_EQI = EQI_GRADE_BOUNDARIES[3]   # grade NORMAL
REGRESS_EQI = EQI_GRADE_BOUNDARIES[1]   # grade SEVERE_COMPENSATION
CONSECUTIVE_SESSIONS = 3
STEP_FRACTION = 0.05                    # of the exercise's own floor-ceiling span
FATIGUE_EQI_PER_REP = -1.8              # one grade band over a 10-rep set
VOLUME_STEP = 2
MIN_REPS = 5
SAFETY_FIRING_RATE = 0.5

SAFETY_SLUGS = frozenset(ERROR_SPECS[e].name_short for e in safety_errors())


class Action(Enum):
    """Naming the one dose change a decision asks for"""
    HOLD = "hold"
    NEED_MORE_SESSIONS = "need_more_sessions"
    SAFETY_HOLD = "safety_hold"
    REDUCE_VOLUME = "reduce_volume"
    RESTORE_VOLUME = "restore_volume"
    REGRESS_RANGE = "regress_range"
    ADVANCE_RANGE = "advance_range"
    UNLOCK_NEXT = "unlock_next"


@dataclass(frozen=True)
class SessionRecord:
    """Holding one completed session's aggregate scores for the coach to read"""
    exercise_id: str
    mean_eqi: float
    n_reps: int
    correct_ratio: float
    fatigue_trend: float
    error_frequency: dict[str, float]
    timestamp: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "SessionRecord":
        """Building a record from the client's JSON, coercing missing scores to zero"""
        return cls(
            exercise_id=str(payload["exercise_id"]),
            mean_eqi=float(payload.get("mean_eqi") or 0.0),
            n_reps=int(payload.get("n_reps") or 0),
            correct_ratio=float(payload.get("correct_ratio") or 0.0),
            fatigue_trend=float(payload.get("fatigue_trend") or 0.0),
            error_frequency={
                str(k): float(v)
                for k, v in (payload.get("error_frequency") or {}).items()
            },
            timestamp=str(payload.get("timestamp") or ""),
        )


@dataclass(frozen=True)
class Decision:
    """Holding a dose change together with the numbers that justified it"""
    action: Action
    prescription: Prescription
    reason: str
    evidence: dict
    unlocked_exercise_id: str | None = None

    @property
    def changed(self) -> bool:
        """Reporting whether this decision alters the dose at all"""
        return self.action not in (
            Action.HOLD,
            Action.NEED_MORE_SESSIONS,
            Action.SAFETY_HOLD,
        )

    def as_dict(self) -> dict:
        """Converting the decision to the JSON the service returns"""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "changed": self.changed,
            "unlocked_exercise_id": self.unlocked_exercise_id,
            "prescription": {
                "exercise_id": self.prescription.exercise_id,
                "side": self.prescription.side.name.lower(),
                "rom_target": round(self.prescription.rom_target, 4),
                "sets": self.prescription.sets,
                "reps": self.prescription.reps,
                "tolerance_scale": self.prescription.tolerance_scale,
            },
            "evidence": self.evidence,
        }


def _step(exercise: ExerciseDefinition) -> float:
    """Calculating one progression step in the exercise's own units"""
    return STEP_FRACTION * (exercise.rom.ceiling - exercise.rom.floor)


def _recent(history: list[SessionRecord], exercise_id: str) -> list[SessionRecord]:
    """Selecting the last few sessions of one exercise, oldest first"""
    same = [s for s in history if s.exercise_id == exercise_id]
    return same[-CONSECUTIVE_SESSIONS:]


def _worst_safety_rate(session: SessionRecord) -> tuple[str | None, float]:
    """Finding the most frequent safety-relevant error in a session and its rate"""
    rates = {
        slug: rate
        for slug, rate in session.error_frequency.items()
        if slug in SAFETY_SLUGS
    }
    if not rates:
        return None, 0.0
    slug = max(rates, key=lambda k: rates[k])
    return slug, rates[slug]


def next_exercise(exercise_id: str, unlocked: frozenset[str]) -> str | None:
    """Finding the next exercise in the progression that is not yet unlocked"""
    order = progression_order()
    if exercise_id not in order:
        return None
    for candidate in order[order.index(exercise_id) + 1 :]:
        if candidate not in unlocked:
            return candidate
    return None


def decide(
    history: list[SessionRecord],
    current: Prescription,
    *,
    unlocked: frozenset[str] | None = None,
) -> Decision:
    """Choosing the next dose for one exercise from the patient's session history"""
    exercise = get_exercise(current.exercise_id)
    recent = _recent(history, current.exercise_id)
    evidence: dict = {
        "n_sessions_considered": len(recent),
        "sessions_required": CONSECUTIVE_SESSIONS,
        "rom_target": round(current.rom_target, 4),
        "reps": current.reps,
    }

    if len(recent) < CONSECUTIVE_SESSIONS:
        return Decision(
            Action.NEED_MORE_SESSIONS,
            current,
            f"{len(recent)} of {CONSECUTIVE_SESSIONS} sessions recorded",
            evidence,
        )

    eqis = [s.mean_eqi for s in recent]
    mean_eqi = sum(eqis) / len(eqis)
    evidence["mean_eqi_recent"] = round(mean_eqi, 2)
    evidence["mean_eqi_per_session"] = [round(v, 2) for v in eqis]

    # safety is judged on the last session only
    slug, rate = _worst_safety_rate(recent[-1])
    evidence["worst_safety_error"] = slug
    evidence["worst_safety_rate"] = round(rate, 3)
    if slug is not None and rate > SAFETY_FIRING_RATE:
        return Decision(
            Action.SAFETY_HOLD,
            current,
            f"{slug} fired in {rate:.0%} of the last session's repetitions",
            evidence,
        )

    # volume is cut before range, and restored before range is raised
    fatigue = sum(s.fatigue_trend for s in recent) / len(recent)
    evidence["mean_fatigue_trend"] = round(fatigue, 3)
    if fatigue <= FATIGUE_EQI_PER_REP and current.reps > MIN_REPS:
        return Decision(
            Action.REDUCE_VOLUME,
            replace(current, reps=max(MIN_REPS, current.reps - VOLUME_STEP)),
            f"quality fell {abs(fatigue):.1f} EQI per repetition across the set",
            evidence,
        )

    if all(v < REGRESS_EQI for v in eqis):
        lowered = exercise.rom.with_target(current.rom_target - _step(exercise))
        if lowered.target < current.rom_target:
            return Decision(
                Action.REGRESS_RANGE,
                replace(current, rom_target=lowered.target),
                f"mean EQI {mean_eqi:.0f} below {REGRESS_EQI:.0f} for "
                f"{CONSECUTIVE_SESSIONS} sessions",
                evidence,
            )
        return Decision(
            Action.HOLD,
            current,
            "already at the therapist's floor for this exercise",
            evidence,
        )

    if all(v >= ADVANCE_EQI for v in eqis):
        if current.reps < exercise.default_reps:
            return Decision(
                Action.RESTORE_VOLUME,
                replace(
                    current,
                    reps=min(exercise.default_reps, current.reps + VOLUME_STEP),
                ),
                f"mean EQI {mean_eqi:.0f} with volume still reduced",
                evidence,
            )
        raised = exercise.rom.with_target(current.rom_target + _step(exercise))
        if raised.target > current.rom_target:
            return Decision(
                Action.ADVANCE_RANGE,
                replace(current, rom_target=raised.target),
                f"mean EQI {mean_eqi:.0f} at or above {ADVANCE_EQI:.0f} for "
                f"{CONSECUTIVE_SESSIONS} sessions",
                evidence,
            )
        following = next_exercise(current.exercise_id, unlocked or frozenset())
        if following is not None:
            return Decision(
                Action.UNLOCK_NEXT,
                current,
                f"at the ceiling ({exercise.rom.ceiling:g}) with mean EQI {mean_eqi:.0f}",
                evidence,
                unlocked_exercise_id=following,
            )
        return Decision(
            Action.HOLD,
            current,
            "at the ceiling with no further exercise in the progression",
            evidence,
        )

    return Decision(
        Action.HOLD,
        current,
        f"mean EQI {mean_eqi:.0f} is between {REGRESS_EQI:.0f} and {ADVANCE_EQI:.0f} - working at the right difficulty",
        evidence,
    )

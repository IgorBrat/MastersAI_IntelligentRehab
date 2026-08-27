from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from ..pose.view import CameraView
from ..taxonomy import ErrorType


class Direction(str, Enum):
    """Which sign of a deviation counts as an error"""

    # Magnitude in either direction is a deviation. Correct for trunk rotation and plane drift, where both ways off-target are equally wrong
    ABSOLUTE = "absolute"

    # Correct for shoulder hiking (dropping the shoulder below its resting height is not an error) 
    # and for elbow-flexion synergy (extending further than baseline is desirable, not a fault)
    POSITIVE = "positive"


@dataclass(frozen=True)
class Tolerance:
    """A graded threshold - how far a measurement may deviate before it counts"""

    mild: float
    moderate: float
    severe: float

    def __post_init__(self) -> None:
        if not (self.mild <= self.moderate <= self.severe):
            raise ValueError(f"tolerance bands must be non-decreasing, got {self.mild} / {self.moderate} / {self.severe}")

    def scaled(self, factor: float) -> "Tolerance":
        return Tolerance(self.mild * factor, self.moderate * factor, self.severe * factor)


@dataclass(frozen=True)
class RomTarget:
    """Prescribed range of motion for the primary angle in degrees"""
    target: float
    floor: float
    ceiling: float
    normative: float

    def __post_init__(self) -> None:
        if not (self.floor <= self.target <= self.ceiling):
            raise ValueError(f"target {self.target} must lie within [{self.floor}, {self.ceiling}]")

    def with_target(self, value: float) -> "RomTarget":
        return replace(self, target=float(min(max(value, self.floor), self.ceiling)))

    @property
    def percent_of_normative(self) -> float:
        return 100.0 * self.target / self.normative


@dataclass(frozen=True)
class TempoPrescription:
    """Prescribed timing of one repetition in seconds"""
    concentric_s: float
    hold_s: float
    eccentric_s: float
    # Fractional tolerance on total duration before EXCESSIVE_SPEED fires
    # 0.4 means "40% faster than prescribed is still acceptable"
    speed_tolerance: float = 0.4

    @property
    def total_s(self) -> float:
        return self.concentric_s + self.hold_s + self.eccentric_s

    @property
    def min_acceptable_s(self) -> float:
        return self.total_s * (1.0 - self.speed_tolerance)


@dataclass(frozen=True)
class CompensationLimit:
    """A compensation channel that this exercise watches"""
    error: ErrorType
    channel: str
    tolerance: Tolerance
    direction: Direction = Direction.ABSOLUTE
    baseline_relative: bool = False
    gate_primary_frac: float = 0.0


@dataclass(frozen=True)
class ExerciseDefinition:
    """Complete machine-readable prescription for one exercise"""

    id: str
    name_uk: str
    name_en: str
    # `AngleSeries` channel that defines the repetition excursion
    primary_channel: str
    rom: RomTarget
    tempo: TempoPrescription
    required_view: CameraView
    compensations: tuple[CompensationLimit, ...]

    # plane of elevation in degrees (0 = sagittal, 90 = frontal)
    target_plane_deg: float | None = None
    #: Elbow angle (e.g. 90 deg for external rotation, ~0 deg for straight-arm raises)
    required_elbow_deg: float | None = None

    # How far short of the start position still counts as a completed return
    return_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(10.0, 20.0, 35.0)
    )
    # Fraction of the prescribed hold that may be *missed* before the error fires
    hold_tolerance: Tolerance = field(default_factory=lambda: Tolerance(0.2, 0.4, 0.7))

    # SPARC value at or above which a movement counts as acceptably smooth
    sparc_floor: float = -1.50
    sparc_tolerance: Tolerance = field(default_factory=lambda: Tolerance(0.05, 0.20, 0.45))

    default_sets: int = 2
    default_reps: int = 10
    instructions_uk: str = ""
    clinical_note_en: str = ""
    bilateral_capable: bool = True

    def compensation_for(self, error: ErrorType) -> CompensationLimit | None:
        for limit in self.compensations:
            if limit.error is error:
                return limit
        return None

    @property
    def watched_errors(self) -> tuple[ErrorType, ...]:
        base = (
            ErrorType.INSUFFICIENT_ROM,
            ErrorType.INCOMPLETE_RETURN,
            ErrorType.EXCESSIVE_SPEED,
            ErrorType.JERKY_MOVEMENT,
        )
        hold = (ErrorType.INSUFFICIENT_HOLD,) if self.tempo.hold_s > 0 else ()
        return base + hold + tuple(limit.error for limit in self.compensations)

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class ErrorFamily(str, Enum):
    PERFORMANCE = "performance"
    COMPENSATION = "compensation"
    CONTROL = "control"


class Severity(IntEnum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class ErrorType(IntEnum):
    INSUFFICIENT_ROM = 0
    INCOMPLETE_RETURN = 1
    INSUFFICIENT_HOLD = 2

    TRUNK_LATERAL_LEAN = 3
    TRUNK_FORWARD_FLEXION = 4
    TRUNK_ROTATION = 5
    SHOULDER_HIKING = 6
    ELBOW_FLEXION_SYNERGY = 7
    OUT_OF_PLANE_DRIFT = 8

    EXCESSIVE_SPEED = 9
    JERKY_MOVEMENT = 10


N_ERROR_TYPES = len(ErrorType)


@dataclass(frozen=True)
class ErrorSpec:
    error: ErrorType
    family: ErrorFamily
    name_short: str
    clinical: str
    patient: str
    safety_relevant: bool = False


ERROR_SPECS: dict[ErrorType, ErrorSpec] = {
    ErrorType.INSUFFICIENT_ROM: ErrorSpec(
        error=ErrorType.INSUFFICIENT_ROM,
        family=ErrorFamily.PERFORMANCE,
        name_short="insufficient_rom",
        clinical="Peak active range of motion falls short of the prescribed target angle.",
        patient="Рух виконано не в повній амплітуді - ви не досягли цільового кута.",
    ),
    ErrorType.INCOMPLETE_RETURN: ErrorSpec(
        error=ErrorType.INCOMPLETE_RETURN,
        family=ErrorFamily.PERFORMANCE,
        name_short="incomplete_return",
        clinical="The limb does not return to the starting position between repetitions.",
        patient="Ви не повертаєте руку у вихідне положення між повтореннями.",
    ),
    ErrorType.INSUFFICIENT_HOLD: ErrorSpec(
        error=ErrorType.INSUFFICIENT_HOLD,
        family=ErrorFamily.PERFORMANCE,
        name_short="insufficient_hold",
        clinical="End-range isometric hold is shorter than prescribed.",
        patient="Затримка в кінцевій точці коротша за призначену.",
    ),
    ErrorType.TRUNK_LATERAL_LEAN: ErrorSpec(
        error=ErrorType.TRUNK_LATERAL_LEAN,
        family=ErrorFamily.COMPENSATION,
        name_short="trunk_lateral_lean",
        clinical=(
            "Lateral trunk inclination substituting for glenohumeral excursion "
            "(contralateral lean during abduction)."
        ),
        patient="Ви нахиляєте тулуб убік замість того, щоб піднімати руку.",
    ),
    ErrorType.TRUNK_FORWARD_FLEXION: ErrorSpec(
        error=ErrorType.TRUNK_FORWARD_FLEXION,
        family=ErrorFamily.COMPENSATION,
        name_short="trunk_forward_flexion",
        clinical=(
            "Anterior trunk displacement used to extend functional reach beyond "
            "available shoulder flexion and elbow extension."
        ),
        patient="Ви нахиляєтеся вперед, щоб дотягнутися, - тримайте спину прямо.",
    ),
    ErrorType.TRUNK_ROTATION: ErrorSpec(
        error=ErrorType.TRUNK_ROTATION,
        family=ErrorFamily.COMPENSATION,
        name_short="trunk_rotation",
        clinical="Axial trunk rotation recruited to advance the shoulder girdle.",
        patient="Ви розвертаєте корпус - плечі мають залишатися рівними.",
    ),
    ErrorType.SHOULDER_HIKING: ErrorSpec(
        error=ErrorType.SHOULDER_HIKING,
        family=ErrorFamily.COMPENSATION,
        name_short="shoulder_hiking",
        clinical=(
            "Excessive scapular elevation relative to the pelvis, substituting for "
            "glenohumeral motion; associated with impingement risk."
        ),
        patient="Плече піднімається до вуха - опустіть і розслабте плече.",
        safety_relevant=True,
    ),
    ErrorType.ELBOW_FLEXION_SYNERGY: ErrorSpec(
        error=ErrorType.ELBOW_FLEXION_SYNERGY,
        family=ErrorFamily.COMPENSATION,
        name_short="elbow_flexion_synergy",
        clinical=(
            "Involuntary elbow flexion coupled to shoulder elevation, i.e. the "
            "pathological flexor synergy pattern."
        ),
        patient="Лікоть згинається разом із підйомом руки - тримайте руку прямою.",
    ),
    ErrorType.OUT_OF_PLANE_DRIFT: ErrorSpec(
        error=ErrorType.OUT_OF_PLANE_DRIFT,
        family=ErrorFamily.COMPENSATION,
        name_short="out_of_plane_drift",
        clinical="Movement departs from the prescribed anatomical plane of motion.",
        patient="Рух відходить від заданої площини - ведіть руку чітко у площині.",
    ),
    ErrorType.EXCESSIVE_SPEED: ErrorSpec(
        error=ErrorType.EXCESSIVE_SPEED,
        family=ErrorFamily.CONTROL,
        name_short="excessive_speed",
        clinical=(
            "Repetition tempo faster than prescribed; momentum substitutes for "
            "controlled concentric/eccentric muscle work."
        ),
        patient="Ви виконуєте рух надто швидко - сповільніться і контролюйте рух.",
        safety_relevant=True,
    ),
    ErrorType.JERKY_MOVEMENT: ErrorSpec(
        error=ErrorType.JERKY_MOVEMENT,
        family=ErrorFamily.CONTROL,
        name_short="jerky_movement",
        clinical=(
            "Fragmented, multi-peaked velocity profile indicating poor motor "
            "control rather than a single smooth submovement."
        ),
        patient="Рух переривчастий - намагайтеся рухатись плавно, без ривків.",
    ),
}

# Stable slug -> ErrorType lookup, for parsing dataset annotations
ERROR_BY_SLUG: dict[str, ErrorType] = {spec.name_short: err for err, spec in ERROR_SPECS.items()}

# Vector order used by every multi-label model head and metric report
ERROR_ORDER: tuple[ErrorType, ...] = tuple(ErrorType)

# Column names for report tables / confusion matrices
ERROR_SLUGS: tuple[str, ...] = tuple(ERROR_SPECS[e].name_short for e in ERROR_ORDER)


def errors_in_family(family: ErrorFamily) -> tuple[ErrorType, ...]:
    return tuple(e for e in ERROR_ORDER if ERROR_SPECS[e].family is family)


def safety_errors() -> tuple[ErrorType, ...]:
    return tuple(e for e in ERROR_ORDER if ERROR_SPECS[e].safety_relevant)


class ClinicianGrade(IntEnum):
    UNABLE = 0
    SEVERE_COMPENSATION = 1
    MODERATE_COMPENSATION = 2
    MINOR_DEVIATION = 3
    NORMAL = 4


EQI_GRADE_BOUNDARIES: tuple[float, ...] = (25.0, 50.0, 70.0, 88.0)


def eqi_to_grade(eqi: float, boundaries: tuple[float, ...] = EQI_GRADE_BOUNDARIES) -> ClinicianGrade:
    """Discretising a continuous quality index into a clinician-comparable grade"""
    grade = 0
    for bound in boundaries:
        if eqi >= bound:
            grade += 1
    return ClinicianGrade(min(grade, int(ClinicianGrade.NORMAL)))

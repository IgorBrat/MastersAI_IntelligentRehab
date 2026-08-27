from __future__ import annotations

from ..pose.view import CameraView
from ..taxonomy import ErrorType
from .base import (
    CompensationLimit,
    Direction,
    ExerciseDefinition,
    RomTarget,
    TempoPrescription,
    Tolerance,
)

# Compensation limits recur across exercises; define them once

_TRUNK_LEAN = CompensationLimit(
    error=ErrorType.TRUNK_LATERAL_LEAN,
    channel="trunk_lean_contralateral",
    tolerance=Tolerance(mild=5.0, moderate=10.0, severe=18.0),
    direction=Direction.POSITIVE,
    baseline_relative=True,
    gate_primary_frac=0.25,
)

_TRUNK_FLEXION = CompensationLimit(
    error=ErrorType.TRUNK_FORWARD_FLEXION,
    channel="trunk_forward_flexion",
    tolerance=Tolerance(mild=8.0, moderate=15.0, severe=25.0),
    baseline_relative=True,
    gate_primary_frac=0.25,
)

_TRUNK_ROTATION = CompensationLimit(
    error=ErrorType.TRUNK_ROTATION,
    channel="trunk_rotation",
    tolerance=Tolerance(mild=8.0, moderate=15.0, severe=25.0),
    baseline_relative=True,
    gate_primary_frac=0.25,
)

# Shoulder hiking is measured in torso lengths, so its bands are small numbers: 0.04 torso lengths is roughly 2 cm of acromion rise on an average adult
_SHOULDER_HIKE = CompensationLimit(
    error=ErrorType.SHOULDER_HIKING,
    channel="shoulder_hike",
    tolerance=Tolerance(mild=0.04, moderate=0.08, severe=0.14),
    direction=Direction.POSITIVE,
    baseline_relative=True,
    gate_primary_frac=0.33,
)

_ELBOW_SYNERGY = CompensationLimit(
    error=ErrorType.ELBOW_FLEXION_SYNERGY,
    channel="elbow_flexion_coupling",
    tolerance=Tolerance(mild=15.0, moderate=30.0, severe=50.0),
    direction=Direction.POSITIVE,
    baseline_relative=True,
    gate_primary_frac=0.33,
)

_PLANE_DRIFT = CompensationLimit(
    error=ErrorType.OUT_OF_PLANE_DRIFT,
    channel="plane_deviation",
    tolerance=Tolerance(mild=15.0, moderate=25.0, severe=40.0),
    gate_primary_frac=0.28,
)


SHOULDER_ABDUCTION = ExerciseDefinition(
    id="shoulder_abduction_seated",
    name_uk="Відведення руки в сторону (сидячи)",
    name_en="Seated shoulder abduction",
    primary_channel="shoulder_abduction",
    rom=RomTarget(target=90.0, floor=45.0, ceiling=150.0, normative=180.0),
    tempo=TempoPrescription(concentric_s=3.0, hold_s=2.0, eccentric_s=3.0),
    required_view=CameraView.FRONTAL,
    target_plane_deg=90.0,
    required_elbow_deg=0.0,
    compensations=(_TRUNK_LEAN, _SHOULDER_HIKE, _ELBOW_SYNERGY, _PLANE_DRIFT, _TRUNK_ROTATION),
    instructions_uk=(
        "Сядьте прямо, спина притиснута до спинки стільця. Повільно підніміть "
        "пряму руку в сторону до рівня плеча, затримайтесь на 2 секунди і так само "
        "повільно опустіть. Плече не піднімайте до вуха, тулуб тримайте рівно."
    ),
    clinical_note_en=(
        "The canonical test of glenohumeral abduction against the two dominant "
        "post-stroke substitutions: contralateral trunk lean and scapular "
        "elevation. Elbow coupling is watched because abduction recruits the "
        "flexor synergy in patients at Brunnstrom stages 3-4."
    ),
)

SHOULDER_FLEXION = ExerciseDefinition(
    id="shoulder_flexion_seated",
    name_uk="Підйом руки вперед (сидячи)",
    name_en="Seated shoulder flexion",
    primary_channel="shoulder_flexion",
    rom=RomTarget(target=90.0, floor=45.0, ceiling=160.0, normative=180.0),
    tempo=TempoPrescription(concentric_s=3.0, hold_s=2.0, eccentric_s=3.0),
    required_view=CameraView.SAGITTAL,
    target_plane_deg=None,
    required_elbow_deg=0.0,
    compensations=(_TRUNK_FLEXION, _SHOULDER_HIKE, _ELBOW_SYNERGY),
    instructions_uk=(
        "Сядьте боком до камери, спина пряма. Повільно підніміть пряму руку "
        "вперед і вгору до рівня плеча, затримайтесь на 2 секунди, повільно "
        "опустіть. Не відхиляйтеся назад і не нахиляйтеся вперед."
    ),
    clinical_note_en=(
        "Filmed sagittally so flexion lies in the image plane. Trunk extension "
        "(leaning back) is the mirror-image compensation to reaching flexion and "
        "shows up as a negative forward-flexion excursion."
    ),
)

ELBOW_FLEXION_EXTENSION = ExerciseDefinition(
    id="elbow_flexion_extension",
    name_uk="Згинання та розгинання в лікті",
    name_en="Elbow flexion / extension",
    primary_channel="elbow_flexion",
    rom=RomTarget(target=110.0, floor=60.0, ceiling=145.0, normative=150.0),
    tempo=TempoPrescription(concentric_s=2.0, hold_s=1.0, eccentric_s=3.0),
    required_view=CameraView.SAGITTAL,
    compensations=(_SHOULDER_HIKE, _TRUNK_FLEXION),
    instructions_uk=(
        "Притисніть лікоть до тулуба. Повільно зігніть руку в лікті, підводячи "
        "кисть до плеча, потім повністю випряміть. Плече залишається нерухомим."
    ),
    clinical_note_en=(
        "The eccentric phase is prescribed longer than the concentric because "
        "controlled elbow *extension* against flexor spasticity is the harder and "
        "more clinically valuable half of this movement."
    ),
)

SHOULDER_EXTERNAL_ROTATION = ExerciseDefinition(
    id="shoulder_external_rotation",
    name_uk="Зовнішня ротація плеча (лікоть 90°)",
    name_en="Shoulder external rotation at 90 degrees elbow flexion",
    primary_channel="shoulder_rotation",
    rom=RomTarget(target=45.0, floor=15.0, ceiling=80.0, normative=90.0),
    tempo=TempoPrescription(concentric_s=3.0, hold_s=2.0, eccentric_s=3.0),
    required_view=CameraView.FRONTAL,
    required_elbow_deg=90.0,
    compensations=(_TRUNK_ROTATION, _SHOULDER_HIKE, _TRUNK_LEAN),
    instructions_uk=(
        "Притисніть лікоть до тулуба і зігніть його під прямим кутом. Повільно "
        "відведіть передпліччя назовні, не відриваючи лікоть від тулуба. "
        "Затримайтесь на 2 секунди і повернітесь."
    ),
    clinical_note_en=(
        "A Fugl-Meyer UE item and a strong predictor of functional recovery. "
        "Rotation estimation from monocular pose is only well-conditioned with "
        "the elbow flexed, so the 90-degree elbow is a measurement precondition "
        "as well as a clinical one -- the rule engine gates on it."
    ),
)

FORWARD_REACH = ExerciseDefinition(
    id="forward_reach_to_target",
    name_uk="Дотягування вперед до цілі",
    name_en="Seated forward reach to target",
    primary_channel="hand_forward",
    # Measured in torso lengths, not degrees: 0.55 torso lengths forward of the
    # pelvis is roughly an arm's-length reach for an adult.
    rom=RomTarget(target=0.55, floor=0.25, ceiling=0.85, normative=0.90),
    tempo=TempoPrescription(concentric_s=2.0, hold_s=1.0, eccentric_s=2.0),
    required_view=CameraView.SAGITTAL,
    compensations=(_TRUNK_FLEXION, _TRUNK_ROTATION, _SHOULDER_HIKE),
    return_tolerance=Tolerance(0.08, 0.15, 0.25),
    instructions_uk=(
        "Сядьте боком до камери. Тягніться рукою вперед, намагаючись дістати "
        "якнайдалі, але НЕ нахиляючи тулуб. Спина залишається притиснутою до "
        "спинки стільця."
    ),
    clinical_note_en=(
        "The task where trunk compensation is most strongly documented "
        "(Cirstea & Levin 2000): patients recover reach distance by displacing "
        "the trunk rather than by recovering shoulder/elbow excursion. Scoring "
        "reach distance alone would reward exactly the wrong adaptation, which is "
        "why trunk flexion carries the heaviest penalty weight here."
    ),
)

HAND_TO_MOUTH = ExerciseDefinition(
    id="hand_to_mouth",
    name_uk="Піднесення руки до рота",
    name_en="Hand to mouth",
    primary_channel="hand_up",
    rom=RomTarget(target=0.45, floor=0.20, ceiling=0.60, normative=0.65),
    tempo=TempoPrescription(concentric_s=2.0, hold_s=1.0, eccentric_s=2.0),
    required_view=CameraView.FRONTAL,
    compensations=(_TRUNK_FLEXION, _TRUNK_LEAN, _SHOULDER_HIKE),
    return_tolerance=Tolerance(0.08, 0.15, 0.25),
    instructions_uk=(
        "Повільно піднесіть кисть до рота, ніби збираєтесь випити води. "
        "Затримайтесь на секунду і повільно опустіть руку. Голову не нахиляйте до руки."
    ),
    clinical_note_en=(
        "A Fugl-Meyer UE item and a self-care task with immediate functional "
        "meaning for the patient, which helps with the motivation problem the "
        "project targets. Watch for the patient bringing the head down to the "
        "hand instead of the hand up to the head."
    ),
)


PROTOCOL: tuple[ExerciseDefinition, ...] = (
    SHOULDER_ABDUCTION,
    SHOULDER_FLEXION,
    ELBOW_FLEXION_EXTENSION,
    SHOULDER_EXTERNAL_ROTATION,
    FORWARD_REACH,
    HAND_TO_MOUTH,
)

# Suggested progression order, easiest to hardest. Used by `adaptation.progress`
PROGRESSION_ORDER: tuple[str, ...] = (
    ELBOW_FLEXION_EXTENSION.id,
    SHOULDER_ABDUCTION.id,
    SHOULDER_FLEXION.id,
    SHOULDER_EXTERNAL_ROTATION.id,
    HAND_TO_MOUTH.id,
    FORWARD_REACH.id,
)

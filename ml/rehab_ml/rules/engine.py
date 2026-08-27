from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exercises.base import ExerciseDefinition, Tolerance
from ..features.angles import AngleSeries, compute_angles
from ..features.kinematics import (
    RomMetrics,
    TempoMetrics,
    ldlj,
    n_submovements,
    rom_metrics,
    smooth_signal,
    sparc,
    speed_profile,
    tempo_metrics,
)
from ..schema import Joint, PoseSequence, Side, joint_of
from ..segmentation.repetition import SegmentationConfig, segment_repetitions
from ..taxonomy import (
    ERROR_SPECS,
    ClinicianGrade,
    ErrorType,
    N_ERROR_TYPES,
    Severity,
    eqi_to_grade,
)
from ..pose.view import check_view as _check
from .criteria import Finding, channel_unit, compensation_penalty, grade
from .quality_index import EqiConfig, QualityIndex, compute_eqi


# Low-pass cutoff for the primary clinical channel
CLINICAL_CUTOFF_HZ = 2.0


@dataclass(frozen=True)
class SmoothnessMetrics:
    """Movement-quality summary of one repetition"""
    sparc: float
    ldlj: float
    n_submovements: int


@dataclass
class RepetitionAssessment:
    """Complete rule-based verdict on one repetition"""
    rep_index: int
    exercise_id: str
    side: Side
    frames: tuple[int, int]
    findings: tuple[Finding, ...]
    rom: RomMetrics
    tempo: TempoMetrics
    smoothness: SmoothnessMetrics
    quality: QualityIndex

    @property
    def triggered(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.triggered)

    @property
    def is_correct(self) -> bool:
        """Binary correct/incorrect verdict"""
        return not self.triggered

    @property
    def worst_severity(self) -> Severity:
        return max((f.severity for f in self.findings), default=Severity.NONE)

    @property
    def safety_flags(self) -> tuple[Finding, ...]:
        """Triggered findings that carry injury risk, at MODERATE or worse"""
        return tuple(
            f
            for f in self.triggered
            if ERROR_SPECS[f.error].safety_relevant and f.severity >= Severity.MODERATE
        )

    def finding_for(self, error: ErrorType) -> Finding | None:
        for f in self.findings:
            if f.error is error:
                return f
        return None

    def severity_vector(self) -> np.ndarray:
        """``(N_ERROR_TYPES,)`` severities as floats, for model targets"""
        out = np.full(N_ERROR_TYPES, np.nan, dtype=np.float32)
        for f in self.findings:
            out[int(f.error)] = float(f.severity)
        return out

    def label_vector(self) -> np.ndarray:
        """``(N_ERROR_TYPES,)`` binary multi-label targets, NaN where unwatched"""
        sev = self.severity_vector()
        with np.errstate(invalid="ignore"):
            return np.where(np.isnan(sev), np.nan, (sev > 0).astype(np.float32))

    def as_dict(self) -> dict:
        return {
            "rep_index": self.rep_index,
            "exercise_id": self.exercise_id,
            "side": self.side.name,
            "frames": list(self.frames),
            "is_correct": self.is_correct,
            "worst_severity": self.worst_severity.name,
            "eqi": round(self.quality.value, 2),
            "grade": int(self.quality.grade),
            "rom": {
                "peak": round(self.rom.peak, 2),
                "excursion": round(self.rom.excursion, 2),
                "return_error": round(self.rom.return_error, 2),
            },
            "tempo": {
                "duration_s": round(self.tempo.duration_s, 2),
                "hold_s": round(self.tempo.hold_s, 2),
            },
            "smoothness": {
                "sparc": None if not np.isfinite(self.smoothness.sparc)
                else round(self.smoothness.sparc, 3),
                "n_submovements": self.smoothness.n_submovements,
            },
            "errors": [
                {
                    "slug": ERROR_SPECS[f.error].name_short,
                    "severity": f.severity.name,
                    "measured": round(f.measured, 4),
                    "threshold": round(f.threshold, 4),
                    "unit": f.unit.strip(),
                }
                for f in self.triggered
            ],
        }


@dataclass
class SessionAssessment:
    """Aggregating verdict on one exercise set"""
    exercise_id: str
    side: Side
    repetitions: list[RepetitionAssessment] = field(default_factory=list)
    # Populated when the recording was rejected before scoring
    rejection_reason_uk: str | None = None

    @property
    def n_reps(self) -> int:
        return len(self.repetitions)

    @property
    def usable(self) -> bool:
        return self.rejection_reason_uk is None and self.n_reps > 0

    @property
    def mean_eqi(self) -> float:
        if not self.repetitions:
            return float("nan")
        return float(np.mean([r.quality.value for r in self.repetitions]))

    @property
    def correct_ratio(self) -> float:
        if not self.repetitions:
            return float("nan")
        return float(np.mean([r.is_correct for r in self.repetitions]))

    @property
    def best_eqi(self) -> float:
        if not self.repetitions:
            return float("nan")
        return float(max(r.quality.value for r in self.repetitions))

    @property
    def grade(self) -> ClinicianGrade:
        return eqi_to_grade(self.mean_eqi if self.repetitions else 0.0)

    def error_frequency(self) -> dict[ErrorType, float]:
        """Fraction of repetitions in which each error triggered"""
        if not self.repetitions:
            return {}
        counts: dict[ErrorType, int] = {}
        watched: dict[ErrorType, int] = {}
        for rep in self.repetitions:
            for f in rep.findings:
                watched[f.error] = watched.get(f.error, 0) + 1
                if f.triggered:
                    counts[f.error] = counts.get(f.error, 0) + 1
        return {
            err: counts.get(err, 0) / watched[err]
            for err in watched
            if watched[err] > 0
        }

    def dominant_error(self) -> ErrorType | None:
        """The most frequent error, tie-broken by mean severity"""
        freq = self.error_frequency()
        if not freq:
            return None
        best, best_key = None, (-1.0, -1.0)
        for err, f in freq.items():
            if f <= 0:
                continue
            sev = float(
                np.mean(
                    [
                        int(x.severity)
                        for r in self.repetitions
                        if (x := r.finding_for(err)) is not None
                    ]
                )
            )
            key = (f, sev)
            if key > best_key:
                best, best_key = err, key
        return best

    def fatigue_trend(self) -> float:
        """Slope of EQI across the set, in points per repetition"""
        if self.n_reps < 3:
            return 0.0
        y = np.array([r.quality.value for r in self.repetitions], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        return float(np.polyfit(x, y, 1)[0])

    def as_dict(self) -> dict:
        return {
            "exercise_id": self.exercise_id,
            "side": self.side.name,
            "usable": self.usable,
            "rejection_reason_uk": self.rejection_reason_uk,
            "n_reps": self.n_reps,
            "mean_eqi": None if not self.repetitions else round(self.mean_eqi, 2),
            "best_eqi": None if not self.repetitions else round(self.best_eqi, 2),
            "correct_ratio": None if not self.repetitions else round(self.correct_ratio, 3),
            "grade": int(self.grade) if self.repetitions else None,
            "fatigue_trend": round(self.fatigue_trend(), 3),
            "error_frequency": {
                ERROR_SPECS[e].name_short: round(v, 3) for e, v in self.error_frequency().items()
            },
            "repetitions": [r.as_dict() for r in self.repetitions],
        }


class RuleEngine:
    """Evaluates repetitions against an exercise's clinical criteria"""

    def __init__(self, eqi_config: EqiConfig | None = None) -> None:
        self.eqi_config = eqi_config or EqiConfig()

    # per repetition
    def assess_repetition(
        self,
        angles: AngleSeries,
        exercise: ExerciseDefinition,
        *,
        rep_index: int = 0,
        frames: tuple[int, int] = (0, 0),
        hand_speed: np.ndarray | None = None,
    ) -> RepetitionAssessment:
        """Scoring one already-segmented repetition"""
        raw_primary = np.asarray(angles[exercise.primary_channel], dtype=np.float64)
        primary = (
            smooth_signal(raw_primary, angles.fps, cutoff_hz=CLINICAL_CUTOFF_HZ)
            if raw_primary.size >= 5
            else raw_primary
        )
        rom = rom_metrics(primary)
        tempo = tempo_metrics(primary, angles.fps)

        speed = (
            np.asarray(hand_speed, dtype=np.float64)
            if hand_speed is not None
            else np.abs(np.gradient(smooth_signal(primary, angles.fps), 1.0 / angles.fps))
        )
        # Smoothness is measured over the CONCENTRIC phase only -- from the start of the repetition to peak excursion
        concentric_end = max(int(tempo.peak_index) + 1, 2)
        concentric_speed = speed[:concentric_end]
        smooth = SmoothnessMetrics(
            sparc=sparc(concentric_speed, angles.fps),
            ldlj=ldlj(concentric_speed, angles.fps),
            n_submovements=n_submovements(concentric_speed, angles.fps),
        )

        findings: list[Finding] = []
        penalties: dict[ErrorType, float] = {}

        # Performance criteria
        target_excursion = abs(exercise.rom.target - rom.start)
        rom_deficit = max(0.0, target_excursion - abs(rom.peak - rom.start))
        rom_bands = (
            0.10 * target_excursion,
            0.25 * target_excursion,
            0.45 * target_excursion,
        )
        findings.append(
            Finding(
                error=ErrorType.INSUFFICIENT_ROM,
                severity=_grade_bands(rom_deficit, rom_bands),
                measured=rom_deficit,
                threshold=rom_bands[0],
                unit=channel_unit(exercise.primary_channel),
            )
        )

        findings.append(
            Finding(
                error=ErrorType.INCOMPLETE_RETURN,
                severity=grade(rom.return_error, exercise.return_tolerance),
                measured=rom.return_error,
                threshold=exercise.return_tolerance.mild,
                unit=channel_unit(exercise.primary_channel),
            )
        )

        if exercise.tempo.hold_s > 0:
            shortfall = max(
                0.0, (exercise.tempo.hold_s - tempo.hold_s) / exercise.tempo.hold_s
            )
            findings.append(
                Finding(
                    error=ErrorType.INSUFFICIENT_HOLD,
                    severity=grade(shortfall, exercise.hold_tolerance),
                    measured=100.0 * shortfall,
                    threshold=100.0 * exercise.hold_tolerance.mild,
                    unit="%",
                )
            )

        # Control criteria
        speed_deficit = max(
            0.0, (exercise.tempo.min_acceptable_s - tempo.duration_s)
        )
        speed_bands = (
            0.10 * exercise.tempo.total_s,
            0.25 * exercise.tempo.total_s,
            0.45 * exercise.tempo.total_s,
        )
        findings.append(
            Finding(
                error=ErrorType.EXCESSIVE_SPEED,
                severity=_grade_bands(speed_deficit, speed_bands),
                measured=speed_deficit,
                threshold=speed_bands[0],
                unit="s",
            )
        )

        sparc_deficit = (
            0.0
            if not np.isfinite(smooth.sparc)
            else max(0.0, exercise.sparc_floor - smooth.sparc)
        )
        findings.append(
            Finding(
                error=ErrorType.JERKY_MOVEMENT,
                severity=grade(sparc_deficit, exercise.sparc_tolerance),
                measured=sparc_deficit,
                threshold=exercise.sparc_tolerance.mild,
                unit=" SPARC",
            )
        )

        # Compensation criteria
        for limit in exercise.compensations:
            finding, penalty = compensation_penalty(angles, exercise, limit)
            findings.append(finding)
            penalties[limit.error] = penalty

        quality = compute_eqi(
            exercise=exercise,
            peak=rom.peak,
            baseline=rom.start,
            sparc=smooth.sparc,
            duration_s=tempo.duration_s,
            hold_s=tempo.hold_s,
            return_error=rom.return_error,
            compensation_penalties=penalties,
            config=self.eqi_config,
        )

        return RepetitionAssessment(
            rep_index=rep_index,
            exercise_id=exercise.id,
            side=angles.side,
            frames=frames,
            findings=tuple(findings),
            rom=rom,
            tempo=tempo,
            smoothness=smooth,
            quality=quality,
        )

    # per session
    def assess_sequence(
        self,
        seq: PoseSequence,
        exercise: ExerciseDefinition,
        side: Side,
        *,
        segmentation: SegmentationConfig | None = None,
        check_view: bool = True,
    ) -> SessionAssessment:
        """Segmenting and score a whole recorded set"""
        session = SessionAssessment(exercise_id=exercise.id, side=side)

        if check_view:
            verdict = _check(seq, exercise.required_view)
            if not verdict.ok:
                session.rejection_reason_uk = verdict.message_uk
                return session

        low_conf = seq.low_confidence_joints(threshold=0.4)
        critical = {
            joint_of(side, "SHOULDER"),
            joint_of(side, "ELBOW"),
            joint_of(side, "WRIST"),
            Joint.L_HIP,
            Joint.R_HIP,
        }
        if critical & set(low_conf):
            session.rejection_reason_uk = (
                "Не вдалося надійно розпізнати ключові точки тіла. Переконайтесь, що вся рука і тулуб у кадрі, а освітлення достатнє."
            )
            return session

        angles = compute_angles(seq, side)
        reps = segment_repetitions(angles, exercise, segmentation)
        if not reps:
            session.rejection_reason_uk = (
                "Не виявлено жодного повторення. Виконайте вправу з більшою амплітудою або перевірте, що камера бачить вас повністю."
            )
            return session

        hand_joint = joint_of(side, "HAND")
        for rep in reps:
            session.repetitions.append(
                self.assess_repetition(
                    angles.slice(rep.start, rep.end),
                    exercise,
                    rep_index=rep.index,
                    frames=(rep.start, rep.end),
                    hand_speed=speed_profile(
                        seq.trajectory(hand_joint)[rep.start : rep.end], seq.fps
                    ),
                )
            )
        return session


def _grade_bands(deviation: float, bands: tuple[float, float, float]) -> Severity:
    """`criteria.grade` for ad-hoc bands that are computed rather than prescribed"""
    mild, moderate, severe = bands
    # Degenerate targets (zero prescribed excursion) must not report SEVERE for a deviation of exactly zero
    if severe <= 1e-9:
        return Severity.NONE
    return grade(deviation, Tolerance(mild, moderate, severe))


__all__ = [
    "RepetitionAssessment",
    "RuleEngine",
    "SessionAssessment",
    "SmoothnessMetrics",
]

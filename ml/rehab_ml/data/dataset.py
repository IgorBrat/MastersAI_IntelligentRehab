from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

from ..exercises.base import ExerciseDefinition
from ..exercises.registry import as_self_paced, get_exercise
from ..features.angles import AngleSeries, compute_angles
from ..features.extractor import (
    FeatureConfig,
    N_SUMMARY_FEATURES,
    sequence_features,
    summary_features,
)
from ..pose.normalize import preprocess
from ..pose.view import infer_working_side
from ..rules.engine import RuleEngine
from ..schema import Side
from ..segmentation.repetition import Repetition, SegmentationConfig, segment_repetitions
from ..taxonomy import ERROR_SPECS, N_ERROR_TYPES
from .adapters.base import RawTrial
from .synthetic import SyntheticCorpus


class LabelSource(str, Enum):
    """Where a sample's supervision came from"""

    # Annotated by a clinician in the source corpus. The only source valid for reporting model-versus-rule comparisons
    HUMAN = "human"

    # Derived by `rules.RuleEngine`. Valid for training, circular for evaluation against the rule baseline
    RULE = "rule"

    # Known by construction from `data.synthetic`. Never valid as evidence
    SYNTHETIC = "synthetic"


@dataclass
class RepetitionSample:
    """One repetition, featurised and labelled"""

    sequence_features: np.ndarray  # (T, C)
    summary_features: np.ndarray   # (F,)
    # Binary multi-label targets, NaN where the exercise does not watch the error
    error_labels: np.ndarray    # (N_ERROR_TYPES,)
    # Graded severities, NaN where unwatched
    severities: np.ndarray  # (N_ERROR_TYPES,)
    # Binary overall correct/incorrect target
    correct: float
    # Rule-engine EQI, used as the regression target and as the rule baseline
    eqi: float
    # Clinician score in [0, 100] when the corpus provides one, else NaN
    clinician_score: float

    subject_id: str
    exercise_id: str
    side: Side
    label_source: LabelSource
    meta: dict = field(default_factory=dict)

    @property
    def is_evaluable(self) -> bool:
        return self.label_source is LabelSource.HUMAN


@dataclass
class RepetitionDataset:
    """An in-memory collection of `RepetitionSample`"""

    samples: list[RepetitionSample] = field(default_factory=list)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> RepetitionSample:
        return self.samples[index]

    # Region - Views

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({s.subject_id for s in self.samples}))

    @property
    def exercise_ids(self) -> tuple[str, ...]:
        return tuple(sorted({s.exercise_id for s in self.samples}))

    def filter(
        self,
        *,
        exercise_id: str | None = None,
        subjects: Iterable[str] | None = None,
        label_source: LabelSource | None = None,
        evaluable_only: bool = False,
    ) -> "RepetitionDataset":
        subject_set = set(subjects) if subjects is not None else None
        selected = [
            s
            for s in self.samples
            if (exercise_id is None or s.exercise_id == exercise_id)
            and (subject_set is None or s.subject_id in subject_set)
            and (label_source is None or s.label_source is label_source)
            and (not evaluable_only or s.is_evaluable)
        ]
        return RepetitionDataset(samples=selected, feature_config=self.feature_config)

    def summary(self) -> dict:
        """Counts by exercise, subject, and label source"""
        by_exercise: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for s in self.samples:
            by_exercise[s.exercise_id] = by_exercise.get(s.exercise_id, 0) + 1
            by_source[s.label_source.value] = by_source.get(s.label_source.value, 0) + 1
        correct = [s.correct for s in self.samples if np.isfinite(s.correct)]
        scored = [s for s in self.samples if np.isfinite(s.clinician_score)]
        return {
            "n_samples": len(self.samples),
            "n_subjects": len(self.subjects),
            "by_exercise": by_exercise,
            "by_label_source": by_source,
            "correct_ratio": float(np.mean(correct)) if correct else float("nan"),
            "n_with_clinician_score": len(scored),
        }

    # Region - Tensor bridges

    def stack(self) -> dict[str, np.ndarray]:
        """Stacking all samples into arrays keyed by field name"""
        if not self.samples:
            raise ValueError("cannot stack an empty dataset")
        return {
            "sequence": np.stack([s.sequence_features for s in self.samples]),
            "summary": np.stack([s.summary_features for s in self.samples]),
            "errors": np.stack([s.error_labels for s in self.samples]),
            "severities": np.stack([s.severities for s in self.samples]),
            "correct": np.array([s.correct for s in self.samples], dtype=np.float32),
            "eqi": np.array([s.eqi for s in self.samples], dtype=np.float32),
            "clinician": np.array(
                [s.clinician_score for s in self.samples], dtype=np.float32
            ),
        }

    def to_tensors(self):
        """Returning the stacked arrays as torch tensors"""

        return {k: torch.from_numpy(v) for k, v in self.stack().items()}


# Region - Splitting


def leave_one_subject_out(
    dataset: RepetitionDataset,
) -> list[tuple[RepetitionDataset, RepetitionDataset]]:
    """Splitting into one fold per subject, each holding that subject out for testing"""
    folds = []
    for held_out in dataset.subjects:
        rest = [s for s in dataset.subjects if s != held_out]
        folds.append(
            (dataset.filter(subjects=rest), dataset.filter(subjects=[held_out]))
        )
    return folds


def build_dataset(
    trials: Iterable[RawTrial],
    *,
    feature_config: FeatureConfig | None = None,
    segmentation: SegmentationConfig | None = None,
    engine: RuleEngine | None = None,
    mirror_augment: bool = False,
    min_reps_per_trial: int = 1,
    self_paced: bool = False,
) -> RepetitionDataset:
    """Segmenting and featurise every trial into repetition samples"""
    cfg = feature_config or FeatureConfig()
    rule_engine = engine or RuleEngine()
    samples: list[RepetitionSample] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for trial in trials:
        if trial.exercise_id is None:
            skip("no_exercise_mapping")
            continue
        try:
            exercise = get_exercise(trial.exercise_id)
        except KeyError:
            skip("unknown_exercise")
            continue
        if self_paced:
            exercise = as_self_paced(exercise)

        variants = [trial.sequence]
        if mirror_augment:
            variants.append(trial.sequence.mirrored())

        for variant in variants:
            sequence = variant if variant.meta.get("preprocessed") else preprocess(variant)
            side = trial.side if trial.side is not None else infer_working_side(sequence)
            if side is None:
                # Bilateral or motionless. Score both arms rather than discarding: a bilateral arm raise is two valid unilateral observations
                sides: Sequence[Side] = (Side.LEFT, Side.RIGHT)
            else:
                sides = (side,)
            if variant.meta.get("mirrored") and trial.side is not None:
                sides = (trial.side.other,)

            for working in sides:
                new = _samples_from_sequence(
                    sequence, exercise, working, trial, cfg, segmentation, rule_engine
                )
                if len(new) < min_reps_per_trial:
                    skip("too_few_repetitions")
                    continue
                samples.extend(new)

    dataset = RepetitionDataset(samples=samples, feature_config=cfg)
    if skipped:
        for sample in dataset.samples[:1]:
            sample.meta.setdefault("build_skips", skipped)
    return dataset


def _samples_from_sequence(
    sequence,
    exercise: ExerciseDefinition,
    side: Side,
    trial: RawTrial,
    cfg: FeatureConfig,
    segmentation: SegmentationConfig | None,
    engine: RuleEngine,
) -> list[RepetitionSample]:
    angles: AngleSeries = compute_angles(sequence, side)
    if trial.pre_segmented:
        # The file is already exactly one repetition; segmenting it would break the alignment between the corpus label and the scored unit
        reps = [Repetition(index=0, start=0, end=angles.n_frames,
                           peak=int(np.argmax(angles[exercise.primary_channel])),
                           peak_value=float(np.max(angles[exercise.primary_channel])),
                           baseline_value=float(angles[exercise.primary_channel][0]))]
    else:
        reps = segment_repetitions(angles, exercise, segmentation)
    if not reps:
        return []

    source = _label_source(trial, sequence)
    out: list[RepetitionSample] = []
    for rep in reps:
        window = angles.slice(rep.start, rep.end)
        assessment = engine.assess_repetition(
            window, exercise, rep_index=rep.index, frames=(rep.start, rep.end)
        )

        # Trial-level human labels apply to every repetition in the trial
        if trial.correct is not None:
            correct = float(trial.correct)
        else:
            correct = float(assessment.is_correct)

        out.append(
            RepetitionSample(
                sequence_features=sequence_features(window, cfg),
                summary_features=summary_features(window, exercise),
                error_labels=assessment.label_vector(),
                severities=assessment.severity_vector(),
                correct=correct,
                eqi=float(assessment.quality.value),
                clinician_score=(
                    float(trial.clinician_score)
                    if trial.clinician_score is not None
                    else float("nan")
                ),
                subject_id=trial.subject_id,
                exercise_id=exercise.id,
                side=side,
                label_source=source,
                meta={
                    # Corpus-level metadata (cohort, diagnosis, affected side, recording position)
                    **trial.meta,
                    "frames": (rep.start, rep.end),
                    "dataset": sequence.meta.get("dataset", "unknown"),
                    "mirrored": bool(sequence.meta.get("mirrored", False)),
                    "rule_correct": assessment.is_correct,
                    "eqi_components": dict(assessment.quality.components),
                    "eqi_penalties": {
                        k: float(v) for k, v in assessment.quality.penalties.items()
                    },
                    "recording_id": str(
                        trial.meta.get("recording_id")
                        or sequence.meta.get("source_path")
                        or trial.subject_id
                    ),
                    "rule_measured": {
                        ERROR_SPECS[f.error].name_short: float(f.measured)
                        for f in assessment.findings
                    },
                },
            )
        )
    return out


def _label_source(trial: RawTrial, sequence) -> LabelSource:
    if sequence.meta.get("synthetic"):
        return LabelSource.SYNTHETIC
    if trial.correct is not None:
        return LabelSource.HUMAN
    return LabelSource.RULE


def dataset_from_synthetic(
    n_subjects: int = 8, reps_per_condition: int = 3, *, seed: int = 0
) -> RepetitionDataset:
    """TEST - Building a dataset from `data.synthetic`"""
    corpus = SyntheticCorpus(
        n_subjects=n_subjects, reps_per_condition=reps_per_condition, seed=seed
    )
    trials = [
        RawTrial(
            sequence=sequence,
            subject_id=subject,
            exercise_id=params.exercise.id,
            side=params.side,
            correct=None,
        )
        for sequence, params, subject in corpus.build()
    ]
    return build_dataset(trials)


__all__ = [
    "LabelSource",
    "N_ERROR_TYPES",
    "N_SUMMARY_FEATURES",
    "RepetitionDataset",
    "RepetitionSample",
    "build_dataset",
    "dataset_from_synthetic",
    "leave_one_subject_out",
]

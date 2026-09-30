from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from dataclasses import replace as _replace

import numpy as np
from fastapi import FastAPI, HTTPException

from ..exercises.base import ExerciseDefinition
from ..exercises.registry import all_exercises, as_self_paced, get_exercise, Prescription
from ..features.angles import compute_angles
from ..pose.mediapipe_estimator import landmarks_to_canonical
from ..pose.normalize import preprocess
from ..rules.engine import RuleEngine, framing_verdict
from ..rules.calibration import load_calibrations, apply_to_exercise
from ..schema import PoseSequence, Side
from ..taxonomy import ERROR_BY_SLUG, ERROR_SPECS
from ..adaptation import SessionRecord, decide
from .schemas import (
    CoachRequest,
    CoachResponse,
    ExerciseInfo,
    PoseFrames,
    ReadinessInfo,
    ServiceInfo,
    VideoRequest,
)
from ..training.export import _git_sha
from .coach import phrase
from .inference import ModelRunner


def _to_sequence(body: PoseFrames) -> PoseSequence:
    """Converting posted MediaPipe landmarks into a preprocessed canonical sequence"""
    n_frames = len(body.landmarks)
    joints = np.empty((n_frames, 18, 3), dtype=np.float64)
    confidence = np.empty((n_frames, 18), dtype=np.float64)

    for index, frame in enumerate(body.landmarks):
        xyz = np.asarray(frame, dtype=np.float64)
        if body.visibility is not None:
            vis = np.asarray(body.visibility[index], dtype=np.float64)
        else:
            vis = np.ones(xyz.shape[0], dtype=np.float64)
        joints[index], confidence[index] = landmarks_to_canonical(xyz, vis)

    sequence = PoseSequence(
        joints=joints,
        fps=float(body.fps),
        confidence=confidence,
        meta={"source": "mediapipe-client"},
    )
    return preprocess(sequence)


VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _sha256(path: str | Path) -> str | None:
    """Hashing the threshold file, so a report records the exact bytes used"""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def create_app(
    *,
    self_paced: bool = False,
    thresholds: str | Path | None = None,
    video_dir: str | Path | None = None,
    model: str | Path | None = None,
    exclude_criteria: Sequence[str] = (),
) -> FastAPI:
    calibrations: dict = {}
    provenance: dict | None = None
    if thresholds is not None:
        calibrations, provenance = load_calibrations(thresholds)

    warnings: list[str] = []
    if thresholds is None:
        # Three protocol bands are far too tight for real movement; with them the
        # rule engine marks nearly every repetition incorrect.
        warnings.append(
            "No --thresholds file: the protocol bands are in force, and three of "
            "them are known to be several times too tight for real movement."
        )

    excluded = set()
    for slug in exclude_criteria:
        if slug not in ERROR_BY_SLUG:
            raise ValueError(
                f"unknown criterion {slug!r}; known: {', '.join(sorted(ERROR_BY_SLUG))}"
            )
        excluded.add(ERROR_BY_SLUG[slug])
    if excluded:
        warnings.append(
            "Criteria switched off for this sensor: "
            + ", ".join(sorted(ERROR_SPECS[e].name_short for e in excluded))
            + ". They cannot appear in any assessment, and the quality index is "
            "computed without them."
        )

    runner: ModelRunner | None = None
    if model is not None:
        runner = ModelRunner.from_path(model)
        warnings.extend(runner.startup_warnings(self_paced=self_paced))
        if excluded:
            warnings.append(
                "The model was trained against rule-engine targets computed WITH "
                "every criterion. Its output is not adjusted for the exclusions above."
            )

    engine = RuleEngine(excluded_errors=frozenset(excluded))
    app = FastAPI(
        title="rehab-ml",
        summary="Movement-quality assessment for post-stroke home rehabilitation",
    )
    # So `serve` can print them where the operator sees them, not only in /health
    app.state.warnings = warnings

    def resolve(exercise_id: str) -> ExerciseDefinition:
        """Looking up an exercise and applying the process-wide scoring configuration"""
        try:
            definition = get_exercise(exercise_id)
        except KeyError:
            raise HTTPException(404, f"unknown exercise {exercise_id!r}") from None
        if self_paced:
            definition = as_self_paced(definition)
        if exercise_id in calibrations:
            definition = apply_to_exercise(definition, calibrations[exercise_id])
        return definition

    def parse_side(raw: str) -> Side:
        try:
            return Side.parse(raw)
        except (KeyError, ValueError):
            raise HTTPException(422, f"side must be 'left' or 'right', got {raw!r}") from None

    @app.get("/health", response_model=ServiceInfo)
    def health() -> ServiceInfo:
        return ServiceInfo(
            status="ok",
            self_paced=self_paced,
            thresholds=None if thresholds is None else str(thresholds),
            thresholds_provenance=provenance,
            thresholds_sha256=None if thresholds is None else _sha256(thresholds),
            git_sha=_git_sha(),
            model=None if runner is None else runner.info(),
            excluded_criteria=sorted(ERROR_SPECS[e].name_short for e in excluded),
            warnings=warnings,
        )

    @app.get("/exercises", response_model=list[ExerciseInfo])
    def exercises() -> list[ExerciseInfo]:
        out = []
        for definition in all_exercises():
            resolved = resolve(definition.id)
            out.append(
                ExerciseInfo(
                    id=resolved.id,
                    name_uk=resolved.name_uk,
                    name_en=resolved.name_en,
                    primary_channel=resolved.primary_channel,
                    required_view=resolved.required_view.name,
                    target_deg=resolved.rom.target,
                    floor_deg=resolved.rom.floor,
                    ceiling_deg=resolved.rom.ceiling,
                    concentric_s=resolved.tempo.concentric_s,
                    hold_s=resolved.tempo.hold_s,
                    eccentric_s=resolved.tempo.eccentric_s,
                    default_reps=resolved.default_reps,
                    instructions_uk=resolved.instructions_uk,
                    watched_errors=[
                        {
                            "slug": ERROR_SPECS[limit.error].name_short,
                            "cue_uk": ERROR_SPECS[limit.error].patient,
                            "safety_relevant": ERROR_SPECS[limit.error].safety_relevant,
                        }
                        for limit in resolved.compensations
                    ],
                )
            )
        return out

    def attach_model(
        payload: dict,
        session,
        sequence: PoseSequence,
        definition: ExerciseDefinition,
        side: Side,
    ) -> dict:
        """Adding the model's per-repetition output beside the rule engine's"""
        if runner is None or not session.repetitions:
            return payload
        if not runner.applies_to(definition.id):
            payload["model"] = {
                "skipped": (
                    f"artifact was fitted for {runner.exercise_id!r}, "
                    f"this request is {definition.id!r}"
                )
            }
            return payload

        angles = compute_angles(sequence, side)
        predictions = []
        for rep, rep_payload in zip(session.repetitions, payload["repetitions"]):
            start, end = rep.frames
            prediction = runner.predict(
                angles.slice(start, end),
                definition,
                rule_eqi=float(rep.quality.value),
            )
            rep_payload["model"] = prediction
            predictions.append(prediction)

        payload["model"] = {
            "artifact": runner.path.name,
            "exercise_id": runner.exercise_id,
            "mean_eqi": round(
                float(np.mean([p["eqi"] for p in predictions])), 2
            ),
            "mean_correct_prob": round(
                float(np.mean([p["correct_prob"] for p in predictions])), 3
            ),
            "mean_eqi_correction": round(
                float(np.mean([p["eqi_correction"] for p in predictions])), 2
            ),
            "validated_on_phone": False,
        }
        return payload

    @app.post("/check/readiness", response_model=ReadinessInfo)
    def check_readiness(body: PoseFrames) -> ReadinessInfo:
        """Is the camera placed well enough to start? Runs on ~2 s of frames"""
        definition = resolve(body.exercise_id)
        side = parse_side(body.side)
        verdict = framing_verdict(_to_sequence(body), definition, side)
        return ReadinessInfo(
            ready=verdict.ok,
            view_ok=verdict.view_ok,
            joints_ok=verdict.joints_ok,
            required_view=definition.required_view.name,
            detected_view=verdict.detected_view,
            yaw_deg=round(verdict.yaw_deg, 1),
            message_uk=verdict.message_uk or "Готово - можна починати",
            missing_joints=list(verdict.missing_joints),
        )

    @app.post("/assess/session")
    def assess_session(body: PoseFrames) -> dict:
        """Scoring a complete set by segmenting it into repetitions first"""
        definition = resolve(body.exercise_id)
        side = parse_side(body.side)
        sequence = _to_sequence(body)
        session = engine.assess_sequence(sequence, definition, side)
        return attach_model(session.as_dict(), session, sequence, definition, side)


    @app.post("/coach", response_model=CoachResponse)
    def coach(body: CoachRequest) -> CoachResponse:
        """Deciding the next dose from the posted history and phrasing it in Ukrainian"""
        definition = resolve(body.exercise_id)
        side = parse_side(body.side)

        current = Prescription.default_for(body.exercise_id, side)
        overrides = {
            name: value
            for name, value in (
                ("rom_target", body.rom_target),
                ("reps", body.reps),
                ("sets", body.sets),
            )
            if value is not None
        }
        if overrides:
            current = _replace(current, **overrides)
            current = _replace(
                current,
                rom_target=definition.rom.with_target(current.rom_target).target,
            )

        history = [
            SessionRecord.from_dict(item.model_dump())
            for item in body.history
            if item.exercise_id
        ]
        decision = decide(history, current, unlocked=frozenset(body.unlocked))

        last = next(
            (r for r in reversed(history) if r.exercise_id == body.exercise_id), None
        )
        text, source, error = phrase(decision, definition, last)

        payload = decision.as_dict()
        return CoachResponse(**payload, text=text, text_source=source, text_error=error)

    @app.get("/videos")
    def videos() -> list[dict]:
        """TEST - Pre-recorded clips available on the server, for the app's video mode"""
        if video_dir is None:
            return []
        root = Path(video_dir)
        if not root.is_dir():
            return []
        out = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
                out.append({
                    "name": path.name,
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "size_mb": round(path.stat().st_size / 1e6, 1),
                })
        return out

    @app.post("/assess/video")
    def assess_video(body: VideoRequest) -> dict:
        """TEST - Score a server-side video end to end"""
        if video_dir is None:
            raise HTTPException(400, "the service was started without --video-dir")
        root = Path(video_dir).resolve()
        target = (root / body.relative_path).resolve()
        if not target.is_file() or root not in target.parents:
            raise HTTPException(404, f"no such video under the served directory: {body.relative_path}")

        definition = resolve(body.exercise_id)
        side = parse_side(body.side)
        try:
            from ..pose.mediapipe_estimator import sequence_from_video

            sequence = preprocess(sequence_from_video(target))
        except Exception as exc:
            raise HTTPException(422, f"could not read poses from {target.name}: {exc}") from exc

        session = engine.assess_sequence(sequence, definition, side)
        out = attach_model(session.as_dict(), session, sequence, definition, side)
        out["video"] = {
            "name": target.name,
            "frames": int(sequence.n_frames),
            "fps": float(sequence.fps),
            "missing_frame_ratio": sequence.meta.get("missing_frame_ratio"),
            "per_joint_confidence": sequence.meta.get("per_joint_confidence"),
        }
        return out

    @app.post("/assess/repetition")
    def assess_repetition(body: PoseFrames) -> dict:
        """Score one already-segmented repetition - the live-feedback"""
        definition = resolve(body.exercise_id)
        side = parse_side(body.side)
        sequence = _to_sequence(body)
        angles = compute_angles(sequence, side)
        assessment = engine.assess_repetition(
            angles, definition, frames=(0, sequence.n_frames)
        )
        payload = assessment.as_dict()

        if runner is not None:
            if runner.applies_to(definition.id):
                payload["model"] = runner.predict(
                    angles, definition, rule_eqi=float(assessment.quality.value)
                )
            else:
                payload["model"] = {
                    "skipped": (
                        f"artifact was fitted for {runner.exercise_id!r}, "
                        f"this request is {definition.id!r}"
                    )
                }
        return payload

    return app


app = create_app(self_paced=True)

from __future__ import annotations

from pathlib import Path
from dataclasses import replace as _replace

import numpy as np
from fastapi import FastAPI, HTTPException

from ..exercises.base import ExerciseDefinition
from ..exercises.registry import all_exercises, as_self_paced, get_exercise, Prescription
from ..features.angles import compute_angles
from ..pose.mediapipe_estimator import landmarks_to_canonical
from ..pose.normalize import preprocess
from ..rules.engine import RuleEngine
from ..rules.calibration import load_calibrations, apply_to_exercise
from ..schema import PoseSequence, Side
from ..taxonomy import ERROR_SPECS
from ..adaptation import SessionRecord, decide
from .schemas import (
    CoachRequest,
    CoachResponse,
    ExerciseInfo,
    PoseFrames,
    ServiceInfo,
    VideoRequest,
)
from .coach import phrase


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


def create_app(
    *,
    self_paced: bool = False,
    thresholds: str | Path | None = None,
    video_dir: str | Path | None = None,
) -> FastAPI:
    calibrations: dict = {}
    provenance: dict | None = None
    if thresholds is not None:
        calibrations, provenance = load_calibrations(thresholds)

    engine = RuleEngine()
    app = FastAPI(
        title="rehab-ml",
        summary="Movement-quality assessment for post-stroke home rehabilitation",
    )

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

    @app.post("/assess/session")
    def assess_session(body: PoseFrames) -> dict:
        """Scoring a complete set by segmenting it into repetitions first"""
        definition = resolve(body.exercise_id)
        side = parse_side(body.side)
        session = engine.assess_sequence(_to_sequence(body), definition, side)
        return session.as_dict()


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
        out = session.as_dict()
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
        return assessment.as_dict()

    return app


app = create_app(self_paced=True)

"""Can the network learn the scapulohumeral invariance from data, or must it be given?"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter  
from rehab_ml.data.dataset import build_dataset  
from rehab_ml.features.angles import AngleSeries  
from rehab_ml.features.extractor import SUMMARY_FEATURE_NAMES  
from rehab_ml.rules import criteria as criteria_mod  
from rehab_ml.training.train import TrainConfig, cross_validate  

OUT = Path("reports/prior_vs_learned.json")

EXERCISES = {
    "elbow": "elbow_flexion_extension",
    "flexion": "shoulder_flexion_seated",
    "abduction": "shoulder_abduction_seated",
}

# feature path corrected, target path corrected
ARMS = {
    "neither": (False, False),
    "feature": (True, False),
    "target": (False, True),
    "both": (True, True),
}


@contextmanager
def paths(feature: bool, target: bool):
    """Correcting the feature path and the target path independently

    `shoulder_hike`'s DEFAULT drives the feature path, because
    `extractor.summary_features` calls it with no keyword. The patched
    `resolve_channel` passes an EXPLICIT keyword, which the wrapper respects, so the rule
    engine - and therefore every auxiliary target - can be set the other way.

    Must wrap `build_dataset`: features and rule-derived targets are both computed at
    build time and cached on the sample.
    """
    original_hike = AngleSeries.shoulder_hike
    original_resolve = criteria_mod.resolve_channel

    def patched_hike(self, baseline_frames=5, *, scapular_correction=None):
        enabled = feature if scapular_correction is None else scapular_correction
        return original_hike(self, baseline_frames, scapular_correction=enabled)

    def patched_resolve(angles, channel, exercise):
        if channel == "shoulder_hike":
            return angles.shoulder_hike(
                baseline_frames=criteria_mod.BASELINE_FRAMES,
                scapular_correction=target,
            )
        return original_resolve(angles, channel, exercise)

    AngleSeries.shoulder_hike = patched_hike
    criteria_mod.resolve_channel = patched_resolve
    try:
        yield
    finally:
        AngleSeries.shoulder_hike = original_hike
        criteria_mod.resolve_channel = original_resolve


def _sequence_digest(dataset) -> str:
    """Fingerprint of the raw sequence tensor, to prove the model input never moved"""
    import hashlib

    stacked = np.stack([s.sequence_features for s in dataset.samples])
    return hashlib.sha256(np.ascontiguousarray(stacked)).hexdigest()[:16]


def run(exercise_key: str, arm: str, seed: int = 0) -> dict:
    """Cross-validating one arm of the 2x2 at one seed"""
    feature, target = ARMS[arm]
    exercise_id = EXERCISES[exercise_key]

    with paths(feature, target):
        dataset = build_dataset(
            IntelliRehabAdapter("../data/IntelliRehabDS"),
            mirror_augment=False,
            self_paced=True,
        ).filter(exercise_id=exercise_id)
        print(f"{len(dataset)} repetitions, {len(dataset.subjects)} subjects")
        digest = _sequence_digest(dataset)
        hike_index = SUMMARY_FEATURE_NAMES.index("shoulder_hike_peak")
        hike = float(
            np.mean([s.summary_features[hike_index] for s in dataset.samples])
        )
        # Seed identical across arms at a given seed, so folds and initialisation are shared
        # WITHIN a paired comparison; varying it across runs exposes run-to-run noise.
        report = cross_validate(dataset, TrainConfig(model="hybrid", epochs=30, seed=seed))

    patient = report.get("by_cohort", {}).get("patient", {})
    return {
        "exercise": exercise_id,
        "seed": seed,
        "feature_corrected": feature,
        "target_corrected": target,
        "n": len(dataset),
        "n_subjects": len(dataset.subjects),
        "sequence_digest": digest,
        "mean_shoulder_hike_peak": round(hike, 5),
        "balanced_all": report["binary_correct"]["balanced_accuracy"],
        "balanced_patients": patient.get("model", {}).get("balanced_accuracy"),
        "recall_patients": patient.get("model", {}).get("recall"),
        "rule_calibrated_patients": patient.get("rule_calibrated", {}).get(
            "balanced_accuracy"
        ),
    }


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {}


def key_for(arm: str, seed: int) -> str:
    """Seed 0 keeps the bare arm name, so the 2026-09-18 results stay addressable"""
    return arm if seed == 0 else f"{arm}#s{seed}"


def split_key(key: str) -> tuple[str, int]:
    arm, _, tail = key.partition("#s")
    return arm, int(tail) if tail else 0


def save(exercise_key: str, arm: str, seed: int, value: dict) -> None:
    store = load()
    store.setdefault(exercise_key, {})[key_for(arm, seed)] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")


def compare() -> None:
    """The seed-0 2x2 decomposition, then the seed-variance check on neither-vs-both"""
    store = load()
    for exercise_key, entries in store.items():
        print(f"\n=== {exercise_key} ===")
        print(f"{'arm':10}{'feature':>9}{'target':>8}{'bal all':>10}{'bal patients':>14}{'seq digest':>18}")
        for arm in ARMS:
            got = entries.get(arm)
            if not got:
                continue
            print(
                f"{arm:10}{str(got['feature_corrected']):>9}{str(got['target_corrected']):>8}"
                f"{got['balanced_all']:>10.4f}"
                f"{(got['balanced_patients'] or float('nan')):>14.4f}"
                f"{got['sequence_digest']:>18}"
            )

        digests = {a["sequence_digest"] for a in entries.values()}
        print(
            f"\nsequence tensor identical across arms: {len(digests) == 1}"
            f"  ({len(digests)} distinct digest(s))"
        )

        for label, a_key, b_key in (
            ("feature path alone", "neither", "feature"),
            ("target path alone", "neither", "target"),
            ("both paths", "neither", "both"),
            ("feature path, given target", "target", "both"),
            ("target path, given feature", "feature", "both"),
        ):
            a, b = entries.get(a_key), entries.get(b_key)
            if not a or not b:
                continue
            delta = b["balanced_patients"] - a["balanced_patients"]
            print(f"effect of {label:28} {delta:+.4f} on patients   (seed 0)")

        _seed_variance(entries)


def _seed_variance(entries: dict) -> None:
    """Is the neither-vs-both difference bigger than run-to-run noise?"""
    by_arm: dict[str, dict[int, float]] = {}
    for key, value in entries.items():
        arm, seed = split_key(key)
        bal = value.get("balanced_patients")
        if bal is not None:
            by_arm.setdefault(arm, {})[seed] = bal

    seeds = sorted(set(by_arm.get("neither", {})) & set(by_arm.get("both", {})))
    if len(seeds) < 2:
        print(f"\nseed variance: {len(seeds)} paired seed(s) - need at least 2")
        return

    print(f"\nseed variance over {len(seeds)} seeds {seeds}")
    print(f"{'seed':>6}{'neither':>10}{'both':>10}{'delta':>10}")
    deltas = []
    for seed in seeds:
        a, b = by_arm["neither"][seed], by_arm["both"][seed]
        deltas.append(b - a)
        print(f"{seed:>6}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")

    arr = np.array(deltas)
    for arm in ("neither", "both"):
        values = np.array([by_arm[arm][s] for s in seeds])
        print(
            f"{arm:>10} mean {values.mean():.4f}  sd {values.std(ddof=1):.4f}"
            f"  range {values.min():.4f}-{values.max():.4f}"
        )
    print(
        f"{'delta':>10} mean {arr.mean():+.4f}  sd {arr.std(ddof=1):.4f}"
        f"  range {arr.min():+.4f}..{arr.max():+.4f}"
        f"  sign consistent: {bool(np.all(arr > 0) or np.all(arr < 0))}"
    )
    # A paired effect smaller than the spread of its own replicates is not an effect.
    if arr.std(ddof=1) > 1e-12:
        print(f"{'':>10} |mean| / sd = {abs(arr.mean()) / arr.std(ddof=1):.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    if sys.argv[1] == "compare":
        compare()
        raise SystemExit(0)

    exercise_key = sys.argv[1]
    arm = sys.argv[2] if len(sys.argv) > 2 else "compare"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if exercise_key not in EXERCISES:
        print(f"unknown exercise {exercise_key!r}; expected one of {list(EXERCISES)}")
        raise SystemExit(2)
    if arm == "compare":
        compare()
    elif arm in ARMS:
        save(exercise_key, arm, seed, run(exercise_key, arm, seed))
    else:
        print(__doc__)
        raise SystemExit(2)

"""Measuring what the scapulohumeral correction does to the elbow-flexion model
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter  
from rehab_ml.data.dataset import build_dataset  
from rehab_ml.features.angles import AngleSeries  
from rehab_ml.training.train import TrainConfig, cross_validate  

EXERCISE = "elbow_flexion_extension"
OUT = Path("reports/elbow_correction_ab.json")


@contextmanager
def correction(enabled: bool):
    """Forcing `shoulder_hike` on or off for the duration of the block"""
    original = AngleSeries.shoulder_hike

    def patched(self, baseline_frames=5, *, scapular_correction=True):
        return original(self, baseline_frames, scapular_correction=enabled)

    AngleSeries.shoulder_hike = patched
    try:
        yield
    finally:
        AngleSeries.shoulder_hike = original


def run(enabled: bool, mirror: bool) -> dict:
    """Cross-validating one arm and returning its patient-cohort scores"""
    with correction(enabled):
        dataset = build_dataset(
            IntelliRehabAdapter("../data/IntelliRehabDS"),
            mirror_augment=mirror,
            self_paced=True,
        ).filter(exercise_id=EXERCISE)
        print(f"{len(dataset)} repetitions, {len(dataset.subjects)} subjects")
        # Seed identical across arms, so folds and initialisation are shared.
        report = cross_validate(dataset, TrainConfig(model="hybrid", epochs=30, seed=0))

    patient = report.get("by_cohort", {}).get("patient", {})
    return {
        "correction": enabled,
        "mirror": mirror,
        "n": len(dataset),
        "n_subjects": len(dataset.subjects),
        "balanced_all": report["binary_correct"]["balanced_accuracy"],
        "balanced_patients": patient.get("model", {}).get("balanced_accuracy"),
        "recall_patients": patient.get("model", {}).get("recall"),
        "rule_calibrated_patients": patient.get("rule_calibrated", {}).get(
            "balanced_accuracy"
        ),
    }


def load() -> dict:
    """Reading the accumulated arms from disk"""
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {}


def save(key: str, value: dict) -> None:
    """Adding one arm's result to the store so arms need not share a process"""
    store = load()
    store[key] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")


def compare() -> None:
    """Printing the 2x2 grid and the isolated effect of each variable"""
    store = load()
    print(f"{'arm':22}{'mirror':>8}{'correction':>12}{'bal all':>10}{'bal patients':>14}")
    for key in ("off", "on", "off_mirror", "on_mirror"):
        arm = store.get(key)
        if not arm:
            continue
        print(
            f"{key:22}{str(arm['mirror']):>8}{str(arm['correction']):>12}"
            f"{arm['balanced_all']:>10.4f}"
            f"{(arm['balanced_patients'] or float('nan')):>14.4f}"
        )

    print()
    for label, a_key, b_key in (
        ("correction, no mirror", "off", "on"),
        ("correction, with mirror", "off_mirror", "on_mirror"),
        ("mirror, correction on", "on", "on_mirror"),
        ("mirror, correction off", "off", "off_mirror"),
    ):
        a, b = store.get(a_key), store.get(b_key)
        if not a or not b:
            continue
        d = (b["balanced_patients"] or 0) - (a["balanced_patients"] or 0)
        print(f"effect of {label:26} {d:+.4f} on patients")

    print()



if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    mirror = "mirror" in sys.argv[2:]
    if mode == "compare":
        compare()
    elif mode in ("on", "off"):
        key = f"{mode}{'_mirror' if mirror else ''}"
        save(key, run(mode == "on", mirror))
    else:
        print(__doc__)
        raise SystemExit(2)

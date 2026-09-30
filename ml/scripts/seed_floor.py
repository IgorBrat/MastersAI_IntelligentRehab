"""Does the interpretability floor generalise beyond IntelliRehabDS?"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.ui_prmd import UiPrmdAdapter  
from rehab_ml.data.dataset import build_dataset  
from rehab_ml.features.angles import AngleSeries  
from rehab_ml.training.train import TrainConfig, cross_validate  

OUT = Path("reports/seed_floor.json")
SENSORS = ("vicon", "kinect")
ARMS = ("off", "on")


@contextmanager
def correction(enabled: bool):
    original = AngleSeries.shoulder_hike

    def patched(self, baseline_frames=5, *, scapular_correction=True):
        return original(self, baseline_frames, scapular_correction=enabled)

    AngleSeries.shoulder_hike = patched
    try:
        yield
    finally:
        AngleSeries.shoulder_hike = original


def run(sensor: str, arm: str, seed: int) -> dict:
    with correction(arm == "on"):
        dataset = build_dataset(
            UiPrmdAdapter("../data/UIPRMD", sensor=sensor),
            mirror_augment=False,
            self_paced=True,
        )
        print(f"{len(dataset)} repetitions, {len(dataset.subjects)} subjects")
        report = cross_validate(dataset, TrainConfig(model="hybrid", epochs=30, seed=seed))
    return {
        "sensor": sensor,
        "arm": arm,
        "seed": seed,
        "n": len(dataset),
        "n_subjects": len(dataset.subjects),
        "balanced_all": report["binary_correct"]["balanced_accuracy"],
    }


def load() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save(sensor: str, arm: str, seed: int, value: dict) -> None:
    store = load()
    store.setdefault(sensor, {})[f"{arm}#s{seed}"] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2), encoding="utf-8")
    print(f"\nwritten to {OUT}")


def compare() -> None:
    store = load()
    for sensor, entries in store.items():
        by_arm: dict[str, dict[int, float]] = {}
        for key, value in entries.items():
            arm, _, tail = key.partition("#s")
            by_arm.setdefault(arm, {})[int(tail)] = value["balanced_all"]

        seeds = sorted(set(by_arm.get("off", {})) & set(by_arm.get("on", {})))
        print(f"\n=== UI-PRMD {sensor} ===")
        if len(seeds) < 2:
            print(f"{len(seeds)} paired seed(s) - need at least 2")
            continue

        print(f"{'seed':>6}{'off':>10}{'on':>10}{'delta':>10}")
        deltas = []
        for seed in seeds:
            a, b = by_arm["off"][seed], by_arm["on"][seed]
            deltas.append(b - a)
            print(f"{seed:>6}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")

        arr = np.array(deltas)
        for arm in ARMS:
            values = np.array([by_arm[arm][s] for s in seeds])
            print(
                f"{arm:>6} mean {values.mean():.4f}  sd {values.std(ddof=1):.4f}"
                f"  range {values.min():.4f}-{values.max():.4f}"
            )
        print(
            f"{'delta':>6} mean {arr.mean():+.4f}  sd {arr.std(ddof=1):.4f}"
            f"  sign consistent: {bool(np.all(arr > 0) or np.all(arr < 0))}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "compare":
        compare()
        raise SystemExit(0)

    sensor = sys.argv[1]
    arm = sys.argv[2] if len(sys.argv) > 2 else ""
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if sensor not in SENSORS or arm not in ARMS:
        print(__doc__)
        raise SystemExit(2)
    save(sensor, arm, seed, run(sensor, arm, seed))

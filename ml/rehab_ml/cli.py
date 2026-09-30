from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

ADAPTERS = {
    "kimore": ("rehab_ml.data.adapters.kimore", "KimoreAdapter"),
    "ui_prmd": ("rehab_ml.data.adapters.ui_prmd", "UiPrmdAdapter"),
    "intellirehab": ("rehab_ml.data.adapters.intellirehab", "IntelliRehabAdapter"),
}


def _load_adapter(corpus: str, root: Path):
    import importlib

    if corpus not in ADAPTERS:
        raise typer.BadParameter(
            f"unknown corpus {corpus!r}; choose from {', '.join(ADAPTERS)}"
        )
    module_name, class_name = ADAPTERS[corpus]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(root)


# protocol


@app.command()
def protocol() -> None:
    """Printing the post-stroke upper-limb exercise protocol"""
    from .exercises.registry import all_exercises, progression_order

    table = Table(title="Post-stroke upper-limb protocol", show_lines=True)
    for column in ("id", "primary channel", "target", "view", "tempo (c/h/e)", "watches"):
        table.add_column(column, overflow="fold")
    for ex in all_exercises():
        unit = "" if "hand_" in ex.primary_channel else "°"
        table.add_row(
            ex.id,
            ex.primary_channel,
            f"{ex.rom.target:g}{unit} (of {ex.rom.normative:g}{unit})",
            ex.required_view.value,
            f"{ex.tempo.concentric_s:g}/{ex.tempo.hold_s:g}/{ex.tempo.eccentric_s:g}s",
            ", ".join(e.name.lower() for e in ex.watched_errors),
        )
    console.print(table)
    console.print("\n[bold]Progression order:[/bold] " + " -> ".join(progression_order()))


# selftest


@app.command()
def selftest(
    epochs: int = typer.Option(10, help="Training epochs for the pipeline check."),
    subjects: int = typer.Option(6, help="Synthetic subjects to generate."),
) -> None:
    """Exercise the whole pipeline on synthetic data"""
    from .data.dataset import dataset_from_synthetic, leave_one_subject_out
    from .data.synthetic import MovementParams, generate_set
    from .exercises.stroke_upper_limb import SHOULDER_ABDUCTION
    from .models.sequence import count_parameters
    from .pose.normalize import preprocess
    from .rules.engine import RuleEngine
    from .schema import Side
    from .training.train import TrainConfig, predict, train_model

    console.print("[bold]1. rule engine on a clean repetition[/bold]")
    engine = RuleEngine()
    clean = engine.assess_sequence(
        preprocess(generate_set(
            MovementParams(exercise=SHOULDER_ABDUCTION, side=Side.LEFT,
                           peak_deg=90.0, noise_m=0.004), 3)),
        SHOULDER_ABDUCTION, Side.LEFT,
    )
    rep = clean.repetitions[1]
    console.print(
        f"   reps={clean.n_reps}  EQI={rep.quality.value:.1f}  "
        f"grade={int(rep.quality.grade)}  findings={len(rep.triggered)}"
    )

    console.print("\n[bold]2. rule engine on a compensating repetition[/bold]")
    faulty = engine.assess_sequence(
        preprocess(generate_set(
            MovementParams(exercise=SHOULDER_ABDUCTION, side=Side.LEFT, peak_deg=90.0,
                           trunk_lean_deg=20.0, shoulder_hike_tl=0.10, noise_m=0.004), 3)),
        SHOULDER_ABDUCTION, Side.LEFT,
    )
    rep = faulty.repetitions[1]
    console.print(f"   EQI={rep.quality.value:.1f}  gate={rep.quality.gate:.2f}")
    for finding in rep.triggered:
        console.print(f"     - {finding.describe_en()}")

    console.print("\n[bold]3. dataset construction[/bold]")
    dataset = dataset_from_synthetic(n_subjects=subjects, reps_per_condition=2)
    console.print("   " + json.dumps(dataset.summary()))

    console.print("\n[bold]4. models[/bold]")
    train_set, test_set = leave_one_subject_out(dataset)[0]
    for name in ("lstm", "tcn", "hybrid"):
        model, history = train_model(
            train_set, TrainConfig(model=name, epochs=epochs, batch_size=16)
        )
        predictions = predict(model, test_set)
        console.print(
            f"   {name:7s} params={count_parameters(model):>7d}  "
            f"loss {history['train_loss'][0]:.3f} -> {history['train_loss'][-1]:.3f}  "
            f"EQI {predictions['eqi'].min():.0f}-{predictions['eqi'].max():.0f}"
        )

    console.print(
        "\n[green]Pipeline OK.[/green] "
        "[yellow]These numbers are synthetic and must not be reported as results.[/yellow]"
    )


# adapter verification

@app.command()
def verify(
    corpus: str = typer.Argument(..., help="kimore | ui_prmd | intellirehab"),
    root: Path = typer.Option(..., help="Dataset root directory."),
    limit: int = typer.Option(5, help="Trials to inspect in detail."),
) -> None:
    """Validating an adapter against real files before training on them"""
    from .data.adapters.base import verify_adapter

    adapter = _load_adapter(corpus, root)
    console.print(f"[bold]{adapter.name}[/bold] at {adapter.root}")

    table = Table(show_lines=False)
    for column in ("subject", "exercise", "side", "correct", "score", "frames", "problems"):
        table.add_column(column, overflow="fold")

    checked = 0
    total = 0
    failures = 0
    for trial in adapter:
        total += 1
        if checked < limit:
            problems = verify_adapter(trial)
            failures += bool(problems)
            table.add_row(
                trial.subject_id,
                str(trial.exercise_id),
                trial.side.name if trial.side is not None else "-",
                str(trial.correct),
                "-" if trial.clinician_score is None else f"{trial.clinician_score:.1f}",
                str(trial.sequence.n_frames),
                "[green]ok[/green]" if not problems else "[red]" + "; ".join(problems) + "[/red]",
            )
            checked += 1

    console.print(table)
    console.print(f"\ntrials found: {total}")
    if failures:
        console.print(
            f"[red]{failures} of {checked} inspected trials failed geometric checks.[/red]\n"
            "Fix JOINT_MAP / AXIS_TRANSFORM / UNIT_SCALE in the adapter before training."
        )
        raise typer.Exit(code=1)
    if total == 0:
        console.print("[red]No trials parsed. Check the directory layout.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Geometric checks passed on the inspected trials.[/green]")


@app.command()
def inspect(
    corpus: str = typer.Argument(...),
    root: Path = typer.Option(...),
    mirror: bool = typer.Option(False, help="Include mirrored augmentation."),
) -> None:
    """Building the dataset from a corpus and report its composition"""
    from .data.dataset import build_dataset

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, mirror_augment=mirror)
    console.print(json.dumps(dataset.summary(), indent=2))
    console.print(f"\nsubjects ({len(dataset.subjects)}): {', '.join(dataset.subjects[:20])}")


# Training

@app.command()
def train(
    corpus: str = typer.Argument(...),
    root: Path = typer.Option(...),
    model: str = typer.Option("hybrid", help="lstm | tcn | hybrid"),
    epochs: int = typer.Option(60),
    exercise: str = typer.Option(None, help="Restrict to one exercise id."),
    mirror: bool = typer.Option(True, help="Mirror-augment the training data."),
    pooled: bool = typer.Option(
        False,
        "--pooled",
        help="Train ONE model across all exercises instead of one per exercise.",
    ),
    no_aux: bool = typer.Option(
        False,
        "--no-aux",
        help="Disable the per-error and EQI heads. Use when those targets are "
             "rule-derived (e.g. IntelliRehabDS, which labels only correct/incorrect).",
    ),
    self_paced: bool = typer.Option(
        False,
        "--self-paced",
        help="Drop tempo/hold criteria; required for corpora with no prescribed tempo.",
    ),
    out: Path = typer.Option(None, help="Write the JSON report here."),
) -> None:
    """Leave-one-subject-out cross-validation against the rule baseline"""
    from .data.dataset import build_dataset
    from .models.heads import LossWeights
    from .training.train import (
        TrainConfig,
        cross_validate,
        cross_validate_per_exercise,
        save_report,
    )

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, mirror_augment=mirror, self_paced=self_paced)
    if exercise:
        dataset = dataset.filter(exercise_id=exercise)
    if not len(dataset):
        console.print("[red]No samples after filtering.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{len(dataset)} repetitions, {len(dataset.subjects)} subjects[/bold]")
    weights = LossWeights(errors=0.0, eqi=0.0) if no_aux else LossWeights()
    if no_aux:
        console.print(
            "[yellow]--no-aux:[/yellow] training on the human correct/incorrect label "
            "only. The per-error and EQI heads are disabled because their targets "
            "come from the rule engine, not from a clinician."
        )
    cfg = TrainConfig(model=model, epochs=epochs, loss_weights=weights)
    if pooled or exercise:
        report = cross_validate(dataset, cfg)
        summary = report
    else:
        report = cross_validate_per_exercise(dataset, cfg)
        summary = report["overall"]
        table = Table(title="per-exercise (specialist models)", show_lines=False)
        for column in ("exercise", "n", "balanced", "patients", "rule", "rule calib"):
            table.add_column(column, overflow="fold")
        for name, sub in report["per_exercise"].items():
            b = sub["binary_correct"]
            pat = sub.get("by_cohort", {}).get("patient", {})
            patient_bal = pat.get("model", {}).get("balanced_accuracy")
            table.add_row(
                name, str(b["n"]), f"{b['balanced_accuracy']:.3f}",
                "-" if patient_bal is None else f"{patient_bal:.3f}",
                f"{sub['rule_vs_model']['rule_based']['balanced_accuracy']:.3f}",
                f"{sub['rule_calibrated']['balanced_accuracy']:.3f}",
            )
        console.print(table)

    for warning in summary["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    comparison = summary["rule_vs_model"]
    table = Table(title="rule-based vs model", show_lines=True)
    for column in ("metric", "rule (protocol)", "rule (calibrated)", "model", "delta vs calib"):
        table.add_column(column)
    calib = summary.get("rule_calibrated", {})
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        c = calib.get(key)
        table.add_row(
            key,
            f"{comparison['rule_based'][key]:.3f}",
            "-" if c is None else f"{c:.3f}",
            f"{comparison['model'][key]:.3f}",
            "-" if c is None else f"{comparison['model'][key] - c:+.3f}",
        )
    console.print(table)

    intervals = summary.get("intervals", {})
    if intervals:
        it = Table(title="95% bootstrap CI (resampling subjects)")
        for column in ("quantity", "estimate", "95% CI", "width"):
            it.add_column(column)
        for key, d in intervals.items():
            it.add_row(key, f"{d['estimate']:.3f}",
                       f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]", f"{d['ci_width']:.3f}")
        console.print(it)

    cohorts = summary.get("by_cohort", {})
    if cohorts:
        ct = Table(title="by cohort (patients are the clinically meaningful row)")
        for column in ("cohort", "n", "% correct", "model balanced", "rule calib"):
            ct.add_column(column)
        for name, d in cohorts.items():
            note = "  [dim](degenerate)[/dim]" if d.get("degenerate") else ""
            ct.add_row(
                name + note, str(d["n"]), f"{d['positive_rate']:.3f}",
                f"{d['model']['balanced_accuracy']:.3f}",
                f"{d.get('rule_calibrated', d['rule'])['balanced_accuracy']:.3f}",
            )
        console.print(ct)
    console.print(f"McNemar: {json.dumps(summary['mcnemar'])}")
    console.print(f"macro F1 over errors: {summary['errors']['macro_f1']}")
    console.print(f"EQI vs clinician: {json.dumps(summary['eqi_vs_clinician'])}")

    if out:
        console.print(f"\nreport written to {save_report(report, out)}")


@app.command()
def export(
    corpus: str = typer.Argument(...),
    root: Path = typer.Option(..., help="Dataset root directory."),
    out: Path = typer.Option(..., help="Where to write the .pt artifact."),
    model: str = typer.Option("hybrid", help="lstm | tcn | hybrid"),
    epochs: int = typer.Option(60),
    exercise: str = typer.Option(
        None,
        help="Restrict to one exercise id. Recommended: the per-exercise "
             "specialist models beat the pooled one on every exercise.",
    ),
    mirror: bool = typer.Option(True, help="Mirror-augment the training data."),
    no_aux: bool = typer.Option(
        False,
        "--no-aux",
        help="Disable the per-error and EQI heads, for corpora whose only human "
             "label is correct/incorrect.",
    ),
    self_paced: bool = typer.Option(
        False,
        "--self-paced",
        help="Drop tempo/hold criteria; required for corpora with no prescribed tempo.",
    ),
    loso_report: Path = typer.Option(
        None,
        help="The matching `train` report. Recorded in the sidecar as the source "
             "of this artifact's only honest accuracy figure.",
    ),
) -> None:
    """Fit one deployable model on ALL subjects and export it for `serve`"""
    from .data.dataset import build_dataset
    from .models.heads import LossWeights
    from .training.export import save
    from .training.train import TrainConfig, fit_final

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, mirror_augment=mirror, self_paced=self_paced)
    if exercise:
        dataset = dataset.filter(exercise_id=exercise)
    if not len(dataset):
        console.print("[red]No samples after filtering.[/red]")
        raise typer.Exit(code=1)

    weights = LossWeights(errors=0.0, eqi=0.0) if no_aux else LossWeights()
    cfg = TrainConfig(model=model, epochs=epochs, loss_weights=weights)

    trained, provenance = fit_final(dataset, cfg)

    reference = None
    if loso_report is not None:
        if not loso_report.is_file():
            console.print(f"[red]no such report: {loso_report}[/red]")
            raise typer.Exit(code=1)
        report = json.loads(loso_report.read_text(encoding="utf-8"))
        scope = report.get("per_exercise", {}).get(exercise) if exercise else None
        scope = scope or report.get("overall") or report
        patient = scope.get("by_cohort", {}).get("patient", {}).get("model", {})
        reference = {
            "path": str(loso_report),
            "balanced_accuracy_all": scope.get("rule_vs_model", {})
            .get("model", {})
            .get("balanced_accuracy"),
            "balanced_accuracy_patients": patient.get("balanced_accuracy"),
            "n": patient.get("n"),
        }

    provenance = {
        **provenance,
        "corpus": corpus,
        "corpus_root": str(root),
        "exercise_id": exercise,
        "self_paced": self_paced,
        "mirror_augment": mirror,
        "no_aux": no_aux,
        "loso_reference": reference,
    }

    artifact, sidecar = save(
        trained, out, feature_config=dataset.feature_config, provenance=provenance
    )

    for warning in provenance["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    table = Table(title="exported artifact", show_lines=False)
    for column in ("field", "value"):
        table.add_column(column, overflow="fold")
    size_mb = artifact.stat().st_size / 1e6
    table.add_row("artifact", f"{artifact}  ({size_mb:.2f} MB)")
    table.add_row("sidecar", str(sidecar))
    table.add_row("architecture", model)
    table.add_row("exercise", exercise or "ALL (pooled)")
    table.add_row("repetitions", str(provenance["dataset"]["n_samples"]))
    table.add_row("subjects", str(provenance["dataset"]["n_subjects"]))
    table.add_row("epochs run", str(provenance["history"]["epochs_run"]))
    if reference and reference["balanced_accuracy_patients"] is not None:
        table.add_row(
            "LOSO balanced acc (patients)",
            f"{reference['balanced_accuracy_patients']:.3f}  [dim]from {loso_report.name}[/dim]",
        )
    console.print(table)
    console.print(
        "[yellow]This artifact was fitted on every subject and has NO held-out "
        "accuracy of its own.[/yellow] Quote the LOSO figure from `train`."
    )
    console.print("")
    console.print("Serve it with:")
    console.print(f"  python -m rehab_ml serve --model {out}")


@app.command()
def eqi(
    corpus: str = typer.Argument("kimore", help="Corpus with clinician scores."),
    root: Path = typer.Option(..., help="Dataset root directory."),
    self_paced: bool = typer.Option(True, "--self-paced/--prescribed-tempo"),
    cohort: str = typer.Option(
        "patient",
        help="Cohort for the discrimination and gate tests, e.g. 'patient'. "
             "Pass an empty string to pool every cohort.",
    ),
    out: Path = typer.Option(None, help="Write the JSON report here."),
) -> None:
    """Validating the Exercise Quality Index against clinician judgement"""
    from .data.dataset import build_dataset
    from .training.eqi_validation import discriminate, gate_stratified, validate

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, self_paced=self_paced)
    selected = cohort or None

    report = validate(dataset)
    report["discrimination"] = discriminate(dataset, cohort=selected)
    report["gate_test"] = gate_stratified(dataset, cohort=selected)

    _print_discrimination(report["discrimination"])
    _print_gate_test(report["gate_test"])

    if "error" in report:
        console.print(f"\n[yellow]no clinician-score correlation: {report['error']}[/yellow]")
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, default=str))
            console.print(f"report written to {out}")
        return

    console.print(
        f"[bold]{report['n_recordings']} recordings[/bold] from "
        f"{report['n_subjects']} subjects, "
        f"{report['repetitions_per_recording']} repetitions each; "
        f"clinician scores {report['clinician_score_range']}"
    )
    table = Table(title="EQI formulation vs therapist score (per recording)")
    for column in ("formulation", "Spearman", "95% CI", "Kendall", "Pearson", "mean EQI"):
        table.add_column(column)
    for name, d in report["formulations"].items():
        ci = d.get("spearman_ci")
        table.add_row(
            name, f"{d['spearman']:.3f}",
            "-" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{d['kendall_tau']:.3f}", f"{d['pearson']:.3f}", f"{d['mean_eqi']:.1f}",
        )
    console.print(table)
    console.print(f"\n[bold]verdict:[/bold] {report['verdict']}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"report written to {out}")


def _print_discrimination(report: dict) -> None:
    """AUC of each EQI formulation against physiotherapist correct/incorrect labels"""
    if "error" in report:
        console.print(f"[yellow]discrimination: {report['error']}[/yellow]")
        return
    console.print(
        f"\n[bold]discrimination[/bold] - {report['n']} repetitions, "
        f"{report['n_subjects']} subjects"
        + (f", cohort={report['cohort']}" if report.get("cohort") else "")
        + f", {report['positive_rate']:.0%} labelled correct"
    )
    table = Table(title="EQI vs physiotherapist label (higher AUC is better, 0.5 = chance)")
    for column in (
        "formulation", "AUC", "95% CI", "bal.acc LOSO", "95% CI",
        "bal.acc @median", "mean if correct", "mean if incorrect",
    ):
        table.add_column(column)
    for name, d in report["formulations"].items():
        ci = d.get("auc_ci")
        loso_ci = d.get("balanced_accuracy_loso_ci")
        table.add_row(
            name,
            f"{d['auc']:.3f}",
            "-" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{d['balanced_accuracy_loso']:.3f}",
            "-" if not loso_ci else f"[{loso_ci[0]:.3f}, {loso_ci[1]:.3f}]",
            f"{d['balanced_accuracy_at_median']:.3f}",
            f"{d['mean_correct']:.1f}",
            f"{d['mean_incorrect']:.1f}",
        )
    console.print(table)
    console.print(
        "[dim]bal.acc LOSO cuts each subject at the median of the OTHERS - "
        "the figure to compare against a LOSO model. @median cuts at the median of "
        "the same repetitions it grades, so it is not subject-disjoint.[/dim]"
    )


def _print_gate_test(report: dict) -> None:
    """The pre-specified gated-vs-additive test, stratified by divergence"""
    if "error" in report:
        console.print(f"[yellow]gate test: {report['error']}[/yellow]")
        return
    console.print(
        f"\n[bold]gate test[/bold] - stratified on {report['stratified_on']}"
    )
    table = Table(title="gated minus additive, by how much the forms can differ")

    for column in ("stratum", "n", "AUC gated", "AUC additive", "delta AUC", "95% CI", "discordance"):
        table.add_column(column)
    order = ("zero_divergence", "moderate", "high", "pooled")
    for name in order:
        d = report["strata"].get(name) if name != "pooled" else report["pooled"]
        if not d or not d.get("n"):
            continue
        ci = d.get("delta_ci")
        table.add_row(
            name,
            str(d["n"]),
            "-" if d["auc_gated"] is None else f"{d['auc_gated']:.4f}",
            "-" if d["auc_additive"] is None else f"{d['auc_additive']:.4f}",
            "-" if d["delta_auc"] is None else f"{d['delta_auc']:+.4f}",
            "-" if not ci else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
            "-" if d["discordance"] is None else f"{d['discordance']:.1%}",
        )
    console.print(table)
    console.print(f"[bold]gate verdict:[/bold] {report['verdict']}")


# Single-recording assessment
@app.command()
def calibrate(
    corpus: str = typer.Argument(..., help=f"One of: {', '.join(ADAPTERS)}"),
    root: Path = typer.Option(..., help="Dataset root directory."),
    out: Path = typer.Option(..., help="Write the threshold file here."),
    self_paced: bool = typer.Option(
        False,
        "--self-paced",
        help="Drop tempo/hold criteria; required for corpora with no prescribed tempo.",
    ),
    cohort: str = typer.Option(
        None, help="Restrict the reference set to one cohort, e.g. 'patient'."
    ),
) -> None:
    """Fitting rule-engine thresholds from a corpus and save them for `assess`"""
    from .data.dataset import build_dataset
    from .rules.calibration import fit_corpus, save_calibrations

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, mirror_augment=False, self_paced=self_paced)
    if cohort:
        kept = [s for s in dataset.samples if s.meta.get("cohort") == cohort]
        if not kept:
            cohorts = sorted({str(s.meta.get("cohort")) for s in dataset.samples})
            console.print(f"[red]No samples in cohort {cohort!r}. Present: {cohorts}[/red]")
            raise typer.Exit(code=1)
        dataset = replace(dataset, samples=kept)

    console.print(
        f"[bold]{len(dataset)} repetitions, {len(dataset.subjects)} subjects[/bold]"
        + (f", cohort={cohort}" if cohort else "")
    )
    calibrations = fit_corpus(dataset, self_paced=self_paced)

    table = Table(title="fitted thresholds (mild / moderate / severe)", show_lines=False)
    for column in ("exercise", "criterion", "protocol", "fitted", "n"):
        table.add_column(column, overflow="fold")
    for ex_id, cal in calibrations.items():
        protocol_bands = _protocol_bands(ex_id, self_paced)
        for slug, tol in sorted(cal.tolerances.items()):
            if slug in cal.skipped:
                continue
            before = protocol_bands.get(slug)
            table.add_row(
                ex_id,
                slug,
                "-" if before is None else " / ".join(f"{v:g}" for v in before),
                " / ".join(f"{v:g}" for v in (tol.mild, tol.moderate, tol.severe)),
                str(cal.n_reference),
            )
    console.print(table)

    provenance = {
        "corpus": corpus,
        "root": str(root),
        "self_paced": self_paced,
        "cohort": cohort,
        "n_reference_repetitions": len(dataset),
        "n_subjects": len(dataset.subjects),
        "note": (
            "Fitted across ALL subjects of this corpus, so it is not subject-disjoint. "
            "Valid for scoring new recordings; re-scoring this corpus with it is leakage."
        ),
    }
    written = save_calibrations(out, calibrations, provenance)
    console.print(f"thresholds written to {written}")


def _protocol_bands(exercise_id: str, self_paced: bool) -> dict[str, tuple[float, float, float]]:
    """Protocol tolerance bands for the comparison column, keyed by error slug"""
    from .exercises.registry import as_self_paced, get_exercise
    from .rules.calibration import _protocol_tolerances

    exercise = get_exercise(exercise_id)
    if self_paced:
        exercise = as_self_paced(exercise)
    return {
        slug: (t.mild, t.moderate, t.severe)
        for slug, t in _protocol_tolerances(exercise).items()
    }


@app.command()
def assess(
    video: Path = typer.Argument(..., help="Video file to score."),
    exercise: str = typer.Option(..., help="Exercise id; see `protocol`."),
    side: str = typer.Option("right", help="left | right"),
    self_paced: bool = typer.Option(
        False,
        "--self-paced",
        help="Drop the prescribed tempo and end-range hold.",
    ),
    thresholds: Path = typer.Option(
        None,
        help="Threshold file from `rehab-ml calibrate`",
    ),
) -> None:
    """Scoring one recorded video with the rule engine"""
    from .exercises.registry import as_self_paced, get_exercise
    from .pose.mediapipe_estimator import sequence_from_video
    from .pose.normalize import preprocess
    from .rules.engine import RuleEngine
    from .schema import Side

    definition = get_exercise(exercise)
    if self_paced:
        definition = as_self_paced(definition)
        console.print(
            "[yellow]--self-paced:[/yellow] tempo and end-range hold criteria dropped. "
            "Range, return, smoothness and every compensation still apply."
        )
    if thresholds is not None:
        from .rules.calibration import apply_to_exercise, load_calibrations

        calibrations, provenance = load_calibrations(thresholds)
        if exercise not in calibrations:
            console.print(
                f"[red]{thresholds} has no thresholds for {exercise!r}; "
                f"it covers: {sorted(calibrations)}[/red]"
            )
            raise typer.Exit(code=1)
        definition = apply_to_exercise(definition, calibrations[exercise])
        console.print(
            f"[yellow]calibrated thresholds:[/yellow] {provenance.get('corpus', '?')}, "
            f"n={provenance.get('n_reference_repetitions', '?')} repetitions"
            + (", self-paced" if provenance.get("self_paced") else "")
        )
    working = Side.parse(side)

    console.print(f"running pose estimation on {video} ...")
    sequence = preprocess(sequence_from_video(video))
    session = RuleEngine().assess_sequence(sequence, definition, working)

    if not session.usable:
        console.print(f"[red]{session.rejection_reason_uk}[/red]")
        raise typer.Exit(code=1)

    console.print(json.dumps(session.as_dict(), indent=2, ensure_ascii=False))


# Weight audit

@app.command()
def weights(
    corpus: str = typer.Argument("intellirehab", help=f"One of: {', '.join(ADAPTERS)}"),
    root: Path = typer.Option(..., help="Dataset root directory."),
    cohort: str = typer.Option("patient", help="Cohort filter; empty string pools all."),
    self_paced: bool = typer.Option(True, "--self-paced/--prescribed-tempo"),
    out: Path = typer.Option(None, help="Write the JSON report here."),
) -> None:
    """Audit the quality index's component weights against their measured variance"""
    from .data.dataset import build_dataset
    from .training.weight_audit import audit, compare_weightings

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, mirror_augment=False, self_paced=self_paced)
    selected = cohort or None

    a = audit(dataset, cohort=selected)
    if "error" in a:
        console.print(f"[red]{a['error']}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{a['n']} repetitions[/bold]"
                  + (f", cohort={selected}" if selected else ""))
    table = Table(title="component weight vs measured variance")
    for column in ("component", "weight", "mean", "sd", "carries information?"):
        table.add_column(column)
    for c in a["components"]:
        table.add_row(c["component"], f"{c['weight']:.3f}", f"{c['mean']:.3f}",
                      f"{c['sd']:.3f}", "NO - dead weight" if c["dead"] else "yes",
                      style="red" if c["dead"] else "")
    console.print(table)
    if a["dead_components"]:
        console.print(
            f"[red]{a['dead_weight_fraction']:.0%} of the weight is on components with no "
            f"variance:[/red] {', '.join(a['dead_components'])}\n"
            f"effective weights once removed: {a['effective_weights']}"
        )

    r = compare_weightings(dataset, cohort=selected)
    if "error" in r:
        console.print(f"[red]{r['error']}[/red]")
        raise typer.Exit(code=1)
    table = Table(title="weighting arms, scored by AUC against the clinician label")
    for column in ("arm", "AUC", "95% CI", "delta vs clinical", "paired delta 95% CI"):
        table.add_column(column)
    for name, v in r["arms"].items():
        ci, dci = v.get("auc_ci"), v.get("delta_ci")
        table.add_row(
            name, f"{v['auc']:.4f}",
            "-" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{v['delta_auc']:+.4f}",
            "-" if not dci else f"[{dci[0]:+.4f}, {dci[1]:+.4f}]",
        )
    console.print(table)
    console.print(
        "[dim]Read the PAIRED delta interval, not whether the two AUC intervals overlap: "
        "every arm scores the same repetitions, so shared variation cancels in the "
        "difference and the paired test is far more sensitive.[/dim]"
    )
    console.print(f"\n[bold]verdict:[/bold] {r['verdict']}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"audit": a, "comparison": r}, indent=2, default=str))
        console.print(f"report written to {out}")


# cross-sensor agreement
@app.command()
def sensors(
    root: Path = typer.Option(..., help="UI-PRMD root directory."),
    out: Path = typer.Option(None, help="Write the JSON report here."),
) -> None:
    """Measuring angle agreement between UI-PRMD's simultaneous Vicon and Kinect capture"""
    from .data.adapters.ui_prmd import UiPrmdAdapter
    from .training.sensor_agreement import ICC_POOR, report

    console.print("running both sensors over the corpus (a few minutes) ...")
    result = report(
        UiPrmdAdapter(root, sensor="vicon"),
        UiPrmdAdapter(root, sensor="kinect"),
    )
    console.print(
        f"[bold]{result['n_paired']} paired trials[/bold] "
        f"(vicon {result['n_reference_trials']}, kinect {result['n_comparison_trials']})"
    )
    table = Table(title="Kinect vs Vicon, peak per trial (Vicon = reference)")
    for column in ("channel", "n", "bias", "SD", "95% LoA", "ICC(2,1)", "r", "verdict"):
        table.add_column(column, overflow="fold")
    for c in result["channels"]:
        icc = c["icc_2_1"]
        style = "red" if (icc is not None and icc == icc and icc < ICC_POOR) else ""
        lo, hi = c["loa_95"]
        table.add_row(
            c["channel"], str(c["n"]), f"{c['bias']:+.1f}", f"{c['sd']:.1f}",
            f"[{lo:+.1f}, {hi:+.1f}]", f"{icc:.2f}", f"{c['pearson_r']:.2f}",
            c["verdict"], style=style,
        )
    console.print(table)
    if result["poor_agreement"]:
        console.print(
            "[red]POOR agreement (ICC < 0.5):[/red] "
            + ", ".join(result["poor_agreement"])
            + "\nThese channels do not survive a sensor change and should not carry clinical weight."
        )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        console.print(f"report written to {out}")


# Serving

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address. Keep 0.0.0.0 for the phone."),
    port: int = typer.Option(8000),
    thresholds: Path = typer.Option(None, help="Threshold file from `calibrate`."),
    video_dir: Path = typer.Option(
        None,
        help="Directory of pre-recorded clips to expose at /videos, so the app can "
             "score them without the phone handling video at all.",
    ),
    protocol_tempo: bool = typer.Option(
        False,
        "--protocol-tempo",
        help="Enforce the prescribed 3s/2s/3s tempo and end-range hold. Off by "
             "default, because a naturally-paced recording otherwise trips "
             "INSUFFICIENT_HOLD and EXCESSIVE_SPEED on nearly every repetition.",
    ),
    model: Path = typer.Option(
        None,
        help="Exported model artifact from `export`. Its output is returned "
             "BESIDE the rule engine's, never instead of it.",
    ),
    exclude_criteria: list[str] = typer.Option(
        [],
        "--exclude-criteria",
        help="Switch a criterion off by slug, e.g. --exclude-criteria "
             "jerky_movement. For a criterion whose calibrated threshold does "
             "not transfer to this sensor. Removes it from the findings AND from "
             "the quality index.",
    ),
) -> None:
    import socket

    import uvicorn

    from .api.service import create_app

    service = create_app(
        self_paced=not protocol_tempo,
        thresholds=thresholds,
        video_dir=video_dir,
        model=model,
        exclude_criteria=exclude_criteria,
    )

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        lan = probe.getsockname()[0]
        probe.close()
        console.print(f"[bold]this url:  http://{lan}:{port}[/bold]")
    except OSError:
        console.print("[yellow]could not determine the LAN address[/yellow]")

    console.print(
        f"self-paced: {not protocol_tempo} · "
        f"thresholds: {thresholds if thresholds else 'protocol'} · "
        f"videos: {video_dir if video_dir else 'none'} · "
        f"model: {model.name if model else 'rules only'}"
    )

    for warning in service.state.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    uvicorn.run(service, host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()


# Aggregation: graded index versus threshold gate

@app.command()
def aggregate(
    corpus: str = typer.Argument("intellirehab", help=f"One of: {', '.join(ADAPTERS)}"),
    root: Path = typer.Option(..., help="Dataset root directory."),
    cohort: str = typer.Option("patient", help="Cohort filter; empty string pools all."),
    self_paced: bool = typer.Option(True, "--self-paced/--prescribed-tempo"),
    out: Path = typer.Option(None, help="Write the JSON report here."),
) -> None:
    from .data.dataset import build_dataset
    from .training.aggregation import report as aggregation_report

    adapter = _load_adapter(corpus, root)
    dataset = build_dataset(adapter, mirror_augment=False, self_paced=self_paced)

    result = aggregation_report(
        dataset, cohort=cohort or None, self_paced=self_paced
    )

    table = Table(title="balanced accuracy: threshold gate vs graded index")
    for column in ("scope", "n", "rule (binary)", "EQI (graded)", "delta",
                   "95% CI on delta", "resolved"):
        table.add_column(column)
    for scope, r in result.items():
        if "arms" in r:
            ci = r["delta_ci"]
            table.add_row(
                scope, str(r["n"]),
                f"{r['arms']['rule_binary']['balanced_accuracy']:.3f}",
                f"{r['arms']['eqi_graded']['balanced_accuracy']:.3f}",
                f"{r['delta_balanced_accuracy']:+.3f}",
                "-" if not ci else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                "YES" if r["resolved"] else "no",
                style="green" if r["resolved"] else "",
            )
        else:
            table.add_row(scope, str(r.get("n", 0)), "-", "-", "-", "-",
                          r.get("error", "-"))
    console.print(table)
    console.print(
        "[dim]Both arms: same per-fold calibrated tolerances, same stored measurements, "
        "leave-one-subject-out. 'resolved' means the paired interval excludes zero.[/dim]"
    )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        console.print(f"report written to {out}")

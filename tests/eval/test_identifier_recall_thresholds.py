"""Pytest gate over the identifier-recall eval.

Runs the same evaluator as ``python -m tests.eval.identifier_recall``
against the bundled golden set and asserts every acceptance threshold.
This is the CI-friendly entry point — failures show up as a normal
pytest red instead of a separate exit-code dance.
"""

from __future__ import annotations

import importlib

eval_mod = importlib.import_module("tests.eval.identifier_recall")


def test_identifier_extraction_meets_acceptance_thresholds() -> None:
    stats = eval_mod.run_eval(eval_mod.GOLDEN_DIR_DEFAULT)
    violations = eval_mod.check_thresholds(stats)
    assert not violations, (
        "identifier extractor regressed below acceptance thresholds:\n  "
        + "\n  ".join(violations)
        + "\n\nfull report:\n"
        + eval_mod.format_report(stats)
    )


def test_format_report_includes_every_observed_type() -> None:
    stats = eval_mod.run_eval(eval_mod.GOLDEN_DIR_DEFAULT)
    report = eval_mod.format_report(stats)
    for etype in stats:
        assert etype in report

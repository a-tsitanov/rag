"""Stage E — recall/precision eval for the identifier extractor.

Walks every JSON file in a golden directory, runs ``extract_identifiers``
on each ``text`` field, compares the canonical forms returned against
the ``expected`` dictionary, and reports per-type recall/precision.

Acceptance thresholds (from the retrieval-quality plan in
``~/.claude/plans/hashed-rolling-llama.md``):

  * PhoneNumber / Email / INN / OGRN / BIC / DocumentDate — recall ≥ 0.95
  * ContractNumber / Amount — recall ≥ 0.85
  * PostalAddress — recall ≥ 0.75 (rule-based ceiling without libpostal)
  * Precision ≥ 0.90 across all types

Usage::

    python -m tests.eval.identifier_recall                  # informative
    python -m tests.eval.identifier_recall --strict         # CI mode

For ``PostalAddress`` matching is a substring check (any expected token
must appear inside some found canonical) — the rule-based normaliser and
libpostal produce different but both-valid canonical strings, so demanding
exact equality would lock the eval to one backend.  Other types use exact
canonical comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.identifiers import extract_identifiers  # noqa: E402


GOLDEN_DIR_DEFAULT = (
    Path(__file__).resolve().parent / "golden_identifiers"
)


# Per-type recall acceptance thresholds.  Precision threshold is one
# global value applied across every type (see ``PRECISION_THRESHOLD``).
RECALL_THRESHOLDS: dict[str, float] = {
    "PhoneNumber": 0.95,
    "Email": 0.95,
    "INN": 0.95,
    "OGRN": 0.95,
    "BIC": 0.95,
    "DocumentDate": 0.95,
    "ContractNumber": 0.85,
    "Amount": 0.85,
    "PostalAddress": 0.75,
}
PRECISION_THRESHOLD: float = 0.90


# Types whose canonical form is fuzzy enough that we use substring
# containment instead of equality.  At present only addresses qualify.
_SUBSTRING_TYPES: frozenset[str] = frozenset({"PostalAddress"})


@dataclass
class TypeStats:
    """Aggregated counts per identifier type, micro-averaged across docs."""

    expected: int = 0
    found: int = 0
    true_positive: int = 0
    samples_with_expected: int = 0
    miss_examples: list[tuple[str, str]] = field(default_factory=list)
    extra_examples: list[tuple[str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.true_positive / self.expected if self.expected else 1.0

    @property
    def precision(self) -> float:
        return self.true_positive / self.found if self.found else 1.0


def _match(found: set[str], expected: str, etype: str) -> bool:
    """Substring containment for fuzzy types, equality otherwise."""
    if etype in _SUBSTRING_TYPES:
        return any(expected in c for c in found)
    return expected in found


def _is_extra(canonical: str, expected_set: set[str], etype: str) -> bool:
    """A canonical is an extra (false positive) if no expected matches it.

    For substring-types the canonical "matches" if any expected token is
    contained in it — so the SAME canonical can satisfy multiple expected
    entries (we don't punish a single canonical for over-coverage).
    """
    if not expected_set:
        return True
    if etype in _SUBSTRING_TYPES:
        return not any(ex in canonical for ex in expected_set)
    return canonical not in expected_set


def evaluate_case(
    text: str,
    expected: dict[str, list[str]],
    stats: dict[str, TypeStats],
    case_name: str,
) -> None:
    """Update per-type ``stats`` with this case's contribution."""
    found_by_type: dict[str, set[str]] = {}
    for ident in extract_identifiers(text):
        found_by_type.setdefault(ident.entity_type, set()).add(ident.canonical)

    types = set(expected) | set(found_by_type)
    for etype in types:
        s = stats.setdefault(etype, TypeStats())
        exp_list = expected.get(etype, []) or []
        exp_set = set(exp_list)
        f_set = found_by_type.get(etype, set())

        s.expected += len(exp_list)
        s.found += len(f_set)
        if exp_list:
            s.samples_with_expected += 1

        for ex in exp_list:
            if _match(f_set, ex, etype):
                s.true_positive += 1
            else:
                s.miss_examples.append((case_name, ex))
        for c in f_set:
            if _is_extra(c, exp_set, etype):
                s.extra_examples.append((case_name, c))


def load_cases(golden_dir: Path) -> list[dict]:
    """Load every ``*.json`` in ``golden_dir`` (sorted for stable runs)."""
    files = sorted(golden_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no golden cases under {golden_dir}")
    return [json.loads(f.read_text()) for f in files]


def run_eval(golden_dir: Path) -> dict[str, TypeStats]:
    cases = load_cases(golden_dir)
    stats: dict[str, TypeStats] = {}
    for case in cases:
        evaluate_case(
            case["text"], case.get("expected", {}), stats, case["name"],
        )
    return stats


def format_report(stats: dict[str, TypeStats]) -> str:
    lines = [
        f"{'type':18s} {'recall':>8s} {'precision':>10s} "
        f"{'expected':>10s} {'found':>8s} {'TP':>6s}"
    ]
    lines.append("-" * 70)
    for etype in sorted(stats):
        s = stats[etype]
        lines.append(
            f"{etype:18s} "
            f"{s.recall:8.2%} {s.precision:10.2%} "
            f"{s.expected:10d} {s.found:8d} {s.true_positive:6d}"
        )
    return "\n".join(lines)


def check_thresholds(stats: dict[str, TypeStats]) -> list[str]:
    """Return list of human-readable violations.  Empty list = all pass."""
    violations: list[str] = []
    for etype, s in sorted(stats.items()):
        # Only enforce recall when we have expectations to enforce against.
        if s.expected > 0:
            threshold = RECALL_THRESHOLDS.get(etype)
            if threshold is not None and s.recall < threshold:
                violations.append(
                    f"recall {etype}: {s.recall:.2%} < {threshold:.0%}"
                )
        # Precision applies whenever the extractor returned something —
        # otherwise it's vacuously perfect.
        if s.found > 0 and s.precision < PRECISION_THRESHOLD:
            violations.append(
                f"precision {etype}: {s.precision:.2%} < "
                f"{PRECISION_THRESHOLD:.0%}"
            )
    return violations


def stats_to_json(stats: dict[str, TypeStats]) -> dict:
    return {
        etype: {
            "recall": round(s.recall, 4),
            "precision": round(s.precision, 4),
            "expected": s.expected,
            "found": s.found,
            "true_positive": s.true_positive,
            "miss_examples": [
                {"case": c, "value": v} for c, v in s.miss_examples[:5]
            ],
            "extra_examples": [
                {"case": c, "value": v} for c, v in s.extra_examples[:5]
            ],
        }
        for etype, s in stats.items()
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_DIR_DEFAULT,
        help=f"directory of golden JSON cases (default: {GOLDEN_DIR_DEFAULT})",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit code 1 when any threshold is violated (CI mode)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="also write per-type stats to this JSON file",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    stats = run_eval(args.golden)
    print(format_report(stats))
    violations = check_thresholds(stats)
    if violations:
        print("\nTHRESHOLD VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("\nAll thresholds satisfied.")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(stats_to_json(stats), ensure_ascii=False, indent=2)
        )
    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

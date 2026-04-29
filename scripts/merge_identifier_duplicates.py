"""Stage D — periodic merge job for legacy identifier duplicates.

Why this exists
===============

Stages A-C make NEW ingestion produce canonical identifier nodes.  Older
documents (or any LLM-extracted variants that slipped past the system
prompt) still live in Neo4j as fragmented nodes:
``+7 (495) 234-56-78`` / ``8 495 1234567`` / ``+74951234567`` are three
separate nodes that the user perceives as one phone.

This script walks every node of an identifier ``entity_type``, runs the
same canonicaliser used at ingestion time
(``src/ingestion/identifiers.py``), groups nodes that share a canonical
form, and calls ``LightRAG.amerge_entities`` to consolidate each group.

Usage
=====

Dry-run (default — prints plan, modifies nothing)::

    python -m scripts.merge_identifier_duplicates

Apply for real (after a Neo4j backup!)::

    python -m scripts.merge_identifier_duplicates --no-dry-run

Restrict to specific types or cap groups for a partial pass::

    python -m scripts.merge_identifier_duplicates \\
        --types PhoneNumber Email --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.ingestion.identifiers import extract_identifiers  # noqa: E402
from src.retrieval.lightrag_setup import (  # noqa: E402
    close_rag_graph,
    create_rag,
)


# Identifier types that have well-defined canonical forms in
# ``src/ingestion/identifiers.py``.  Person / Organization / Location
# do not — they live in the graph verbatim, the LLM canonicalises them
# at insert time via the system prompt.
DEFAULT_IDENTIFIER_TYPES: frozenset[str] = frozenset({
    "PhoneNumber",
    "Email",
    "INN",
    "OGRN",
    "BIC",
    "ContractNumber",
    "PostalAddress",
    "DocumentDate",
    "Amount",
})


# ── pure helpers (unit-testable) ─────────────────────────────────────


def canonicalize_for_type(name: str, entity_type: str) -> str | None:
    """Run the deterministic detectors on ``name`` and return the
    canonical form for ``entity_type``, or ``None`` if no detector of
    that type fires.

    Idempotent: a name already in canonical form returns itself.
    """
    for ident in extract_identifiers(name):
        if ident.entity_type == entity_type:
            return ident.canonical
    return None


def group_by_canonical(
    nodes: list[tuple[str, str]],
    types: frozenset[str],
) -> dict[tuple[str, str], list[str]]:
    """Group ``[(name, entity_type), ...]`` by ``(entity_type,
    canonical)``.  Skip nodes whose type isn't in ``types`` or that
    don't normalize.

    Resulting groups always need merging — singletons whose canonical
    equals the source name are filtered out.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, etype in nodes:
        if etype not in types:
            continue
        canonical = canonicalize_for_type(name, etype)
        if canonical is None:
            continue
        groups[(etype, canonical)].append(name)
    return {
        key: names
        for key, names in groups.items()
        if len(names) > 1 or (len(names) == 1 and names[0] != key[1])
    }


# ── graph-bound bits ─────────────────────────────────────────────────


async def collect_nodes(rag, batch_size: int = 200) -> list[tuple[str, str]]:
    """Fetch every entity in the graph as ``(entity_id, entity_type)``.

    Uses ``get_all_labels`` + batched ``get_nodes_batch``.  Same call
    pattern works for both Neo4j and NetworkX backends since the storage
    interface is uniform.
    """
    graph = rag.chunk_entity_relation_graph
    labels = await graph.get_all_labels()
    out: list[tuple[str, str]] = []
    for i in range(0, len(labels), batch_size):
        batch = labels[i : i + batch_size]
        nodes = await graph.get_nodes_batch(batch)
        for name, props in nodes.items():
            etype = props.get("entity_type") or ""
            out.append((name, etype))
    logger.info("collected {n} entities from graph", n=len(out))
    return out


async def apply_merges(
    rag,
    groups: dict[tuple[str, str], list[str]],
    *,
    dry_run: bool,
) -> dict[str, int]:
    """For each group call ``amerge_entities`` (or print the plan)."""
    summary = {"groups": len(groups), "merged_sources": 0, "errors": 0}
    for (etype, canonical), names in groups.items():
        if dry_run:
            logger.info(
                "[dry-run] type={t}  target={c!r}  sources={n}",
                t=etype, c=canonical, n=names,
            )
            continue
        try:
            await rag.amerge_entities(
                source_entities=names,
                target_entity=canonical,
                merge_strategy={
                    "description": "concatenate",
                    "source_id": "join_unique",
                    "entity_type": "keep_first",
                },
                target_entity_data={"entity_type": etype},
            )
            summary["merged_sources"] += len(names)
            logger.info(
                "merged  type={t}  target={c!r}  sources={n}",
                t=etype, c=canonical, n=names,
            )
        except Exception as exc:  # noqa: BLE001 — keep job running
            logger.warning(
                "merge failed  type={t}  target={c!r}  err={err}",
                t=etype, c=canonical, err=exc,
            )
            summary["errors"] += 1
    return summary


# ── entry point ──────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preview merges without modifying the graph (default: True). "
        "Pass --no-dry-run to actually merge.",
    )
    p.add_argument(
        "--types",
        nargs="*",
        default=None,
        help=(
            "Restrict to specific entity types "
            f"(default: all of {sorted(DEFAULT_IDENTIFIER_TYPES)})"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of groups processed (debugging / partial passes)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Neo4j get_nodes_batch chunk size (default: 200)",
    )
    return p.parse_args()


async def _amain() -> None:
    args = _parse_args()
    types = (
        frozenset(args.types) if args.types else DEFAULT_IDENTIFIER_TYPES
    )

    rag = await create_rag()
    try:
        nodes = await collect_nodes(rag, batch_size=args.batch_size)
        groups = group_by_canonical(nodes, types)
        if args.limit:
            groups = dict(list(groups.items())[: args.limit])
        logger.info(
            "candidate groups: {n}  types={t}",
            n=len(groups), t=sorted(types),
        )
        if not groups:
            logger.info("nothing to merge — graph is already canonical")
            return
        summary = await apply_merges(rag, groups, dry_run=args.dry_run)
        logger.info(
            "done  dry_run={dr}  summary={s}",
            dr=args.dry_run, s=summary,
        )
    finally:
        await close_rag_graph(rag)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

"""Unit tests for ``scripts/merge_identifier_duplicates.py`` — Stage D.

Graph-bound bits (``collect_nodes``, end-to-end ``_amain``) are exercised
via a fake LightRAG since real Neo4j requires docker compose.  Pure
helpers (canonicalization, grouping) are tested directly.
"""

from __future__ import annotations

import importlib

import pytest

merge_mod = importlib.import_module("scripts.merge_identifier_duplicates")


# ── canonicalize_for_type ────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,etype,expected",
    [
        ("+7 (495) 234-56-78", "PhoneNumber", "+74952345678"),
        ("8 495 234-56-78", "PhoneNumber", "+74952345678"),
        ("+74952345678", "PhoneNumber", "+74952345678"),  # idempotent
        ("Bob@EXAMPLE.com", "Email", "bob@example.com"),
        ("7707083893", "INN", "7707083893"),
        ("15.03.2024", "DocumentDate", "2024-03-15"),
        ("2024-03-15", "DocumentDate", "2024-03-15"),
        ("№ дп-2024/178-К", "ContractNumber", "ДП-2024/178-К"),
    ],
)
def test_canonicalize_for_type_matches(name, etype, expected) -> None:
    assert merge_mod.canonicalize_for_type(name, etype) == expected


def test_canonicalize_for_type_returns_none_for_non_identifier() -> None:
    # A person name doesn't match any identifier detector
    assert merge_mod.canonicalize_for_type("Иванов Иван", "Person") is None
    # Wrong-type request also returns None
    assert merge_mod.canonicalize_for_type("+74952345678", "Email") is None


# ── group_by_canonical ───────────────────────────────────────────────


def test_group_by_canonical_collapses_phone_variants() -> None:
    nodes = [
        ("+7 (495) 234-56-78", "PhoneNumber"),
        ("8 495 234 56 78", "PhoneNumber"),
        ("+74952345678", "PhoneNumber"),  # already canonical
        ("Иванов Иван", "Person"),  # ignored — not an identifier type
    ]
    groups = merge_mod.group_by_canonical(
        nodes, frozenset({"PhoneNumber"}),
    )
    assert len(groups) == 1
    sources = groups[("PhoneNumber", "+74952345678")]
    assert set(sources) == {
        "+7 (495) 234-56-78",
        "8 495 234 56 78",
        "+74952345678",
    }


def test_group_by_canonical_skips_singleton_already_canonical() -> None:
    """A node already in canonical form with no variants must NOT be a
    candidate — there is nothing to merge."""
    nodes = [("+74952345678", "PhoneNumber")]
    groups = merge_mod.group_by_canonical(
        nodes, frozenset({"PhoneNumber"}),
    )
    assert groups == {}


def test_group_by_canonical_keeps_singleton_if_canonical_differs() -> None:
    """Even if a single node exists but its name needs canonicalization,
    we want a one-source group so the merge renames it."""
    nodes = [("+7 (495) 234-56-78", "PhoneNumber")]
    groups = merge_mod.group_by_canonical(
        nodes, frozenset({"PhoneNumber"}),
    )
    assert len(groups) == 1
    assert groups[("PhoneNumber", "+74952345678")] == [
        "+7 (495) 234-56-78",
    ]


def test_group_by_canonical_filters_by_types() -> None:
    nodes = [
        ("Bob@EXAMPLE.com", "Email"),
        ("+7 (495) 234-56-78", "PhoneNumber"),
    ]
    groups = merge_mod.group_by_canonical(
        nodes, frozenset({"Email"}),
    )
    assert set(groups.keys()) == {("Email", "bob@example.com")}


# ── apply_merges ─────────────────────────────────────────────────────


class _FakeRAG:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_next: bool = False

    async def amerge_entities(
        self,
        *,
        source_entities,
        target_entity,
        merge_strategy,
        target_entity_data,
    ):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated merge failure")
        self.calls.append(
            {
                "sources": list(source_entities),
                "target": target_entity,
                "strategy": merge_strategy,
                "data": target_entity_data,
            }
        )


@pytest.mark.asyncio
async def test_apply_merges_dry_run_records_nothing() -> None:
    rag = _FakeRAG()
    groups = {
        ("PhoneNumber", "+74952345678"): [
            "+7 (495) 234-56-78", "8 495 234 56 78",
        ],
    }
    summary = await merge_mod.apply_merges(rag, groups, dry_run=True)
    assert rag.calls == []
    assert summary == {"groups": 1, "merged_sources": 0, "errors": 0}


@pytest.mark.asyncio
async def test_apply_merges_real_run_calls_amerge() -> None:
    rag = _FakeRAG()
    groups = {
        ("Email", "bob@example.com"): ["Bob@EXAMPLE.com", "BOB@example.com"],
        ("INN", "7707083893"): ["7707083893"],  # rename / no-op semantics
    }
    summary = await merge_mod.apply_merges(rag, groups, dry_run=False)
    assert summary["groups"] == 2
    assert summary["merged_sources"] == 3  # 2 + 1
    assert summary["errors"] == 0
    assert len(rag.calls) == 2
    targets = {c["target"] for c in rag.calls}
    assert targets == {"bob@example.com", "7707083893"}


@pytest.mark.asyncio
async def test_apply_merges_continues_on_individual_failure() -> None:
    rag = _FakeRAG()
    rag.fail_next = True
    groups = {
        ("Email", "bob@example.com"): ["Bob@EXAMPLE.com"],
        ("INN", "7707083893"): ["7707083893"],
    }
    summary = await merge_mod.apply_merges(rag, groups, dry_run=False)
    assert summary["groups"] == 2
    assert summary["errors"] == 1
    # second group still merged
    assert summary["merged_sources"] == 1


# ── collect_nodes (with fake graph) ──────────────────────────────────


class _FakeGraph:
    def __init__(self, nodes: dict[str, dict]) -> None:
        self._nodes = nodes

    async def get_all_labels(self) -> list[str]:
        return sorted(self._nodes.keys())

    async def get_nodes_batch(self, ids: list[str]) -> dict[str, dict]:
        return {i: self._nodes[i] for i in ids if i in self._nodes}


class _RAGWithGraph:
    def __init__(self, nodes: dict[str, dict]) -> None:
        self.chunk_entity_relation_graph = _FakeGraph(nodes)


@pytest.mark.asyncio
async def test_collect_nodes_returns_name_type_pairs() -> None:
    rag = _RAGWithGraph(
        {
            "+74952345678": {"entity_type": "PhoneNumber"},
            "bob@example.com": {"entity_type": "Email"},
            "Иванов Иван": {"entity_type": "Person"},
            "no-type-node": {},  # entity_type missing — pass-through with ""
        }
    )
    out = await merge_mod.collect_nodes(rag, batch_size=2)
    assert sorted(out) == sorted(
        [
            ("+74952345678", "PhoneNumber"),
            ("bob@example.com", "Email"),
            ("Иванов Иван", "Person"),
            ("no-type-node", ""),
        ]
    )


@pytest.mark.asyncio
async def test_full_pipeline_dry_run() -> None:
    """End-to-end: graph → collect → group → apply (dry-run)."""
    nodes = {
        "+7 (495) 234-56-78": {"entity_type": "PhoneNumber"},
        "8 495 234-56-78": {"entity_type": "PhoneNumber"},
        "+74952345678": {"entity_type": "PhoneNumber"},
        "bob@example.com": {"entity_type": "Email"},
        "Bob@EXAMPLE.com": {"entity_type": "Email"},
        "Иванов Иван": {"entity_type": "Person"},
    }
    rag = _RAGWithGraph(nodes)
    rag.amerge_entities = _FakeRAG().amerge_entities  # type: ignore[attr-defined]

    collected = await merge_mod.collect_nodes(rag)
    groups = merge_mod.group_by_canonical(
        collected, merge_mod.DEFAULT_IDENTIFIER_TYPES,
    )
    assert ("PhoneNumber", "+74952345678") in groups
    assert ("Email", "bob@example.com") in groups
    assert ("Person", "Иванов Иван") not in groups
    summary = await merge_mod.apply_merges(rag, groups, dry_run=True)
    assert summary["merged_sources"] == 0  # dry-run
    assert summary["groups"] == len(groups)

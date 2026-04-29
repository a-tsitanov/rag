"""Deterministic identifier extraction + canonicalization.

Pre-LLM stage of the retrieval-quality pipeline (Stage B of the plan in
``~/.claude/plans/hashed-rolling-llama.md``).  Detects business
identifiers in raw document text via regex / lib-based parsers and
returns each match with a canonical form suitable for use as an
``entity_name`` in Neo4j.

Why deterministic + canonical:
  Two documents may write the same phone as ``+7 (495) 123-45-67`` and
  ``8 495 1234567``.  If we let the LLM extract both verbatim, Neo4j
  ends up with two separate nodes — graph dedup breaks.  We pre-canon
  to E.164 (``+74951234567``) so identical entities collapse to one
  node regardless of source formatting.

The output of ``extract_identifiers()`` is consumed by Stage C
(``src/ingestion/worker.py``) which:
  1. Calls ``rag.ainsert_custom_kg`` with one entity per canonical
     identifier — guarantees the canonical node exists before LLM
     extraction.
  2. Appends a ``Канонические идентификаторы:`` block to the document
     text so the LLM uses canonical forms when building relationships
     (Stage A taught the LLM this protocol via the system prompt).

``postal`` (libpostal Python bindings) is imported optionally — when
the C library isn't installed locally we fall back to a rule-based
address normalizer.  Phone numbers and dates use pure-Python libs
(``phonenumbers``, ``dateparser``) which install without system deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import dateparser
import phonenumbers

try:  # libpostal is heavy and optional — fall back to rule-based
    from postal.parser import parse_address as _libpostal_parse  # type: ignore[import-not-found]

    _HAS_LIBPOSTAL = True
except ImportError:  # pragma: no cover — depends on system libpostal-dev
    _libpostal_parse = None  # type: ignore[assignment]
    _HAS_LIBPOSTAL = False


IdentifierType = Literal[
    "PhoneNumber",
    "Email",
    "INN",
    "OGRN",
    "BIC",
    "ContractNumber",
    "PostalAddress",
    "DocumentDate",
    "Amount",
]


@dataclass(frozen=True)
class NormalizedIdentifier:
    """One identifier match with both verbatim and canonical forms.

    ``span`` is character offsets in the source text — Stage C uses it
    to build ``Канонические идентификаторы:`` blocks aligned with the
    LLM's view of the document.
    """

    entity_type: IdentifierType
    canonical: str
    original: str
    span: tuple[int, int]


# ── PhoneNumber ──────────────────────────────────────────────────────


def _extract_phones(text: str) -> list[NormalizedIdentifier]:
    """E.164 via google's libphonenumber port. RU as default region."""
    out: list[NormalizedIdentifier] = []
    for match in phonenumbers.PhoneNumberMatcher(text, "RU"):
        canonical = phonenumbers.format_number(
            match.number, phonenumbers.PhoneNumberFormat.E164
        )
        out.append(
            NormalizedIdentifier(
                entity_type="PhoneNumber",
                canonical=canonical,
                original=match.raw_string,
                span=(match.start, match.end),
            )
        )
    return out


# ── Email ────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def _extract_emails(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    for m in _EMAIL_RE.finditer(text):
        out.append(
            NormalizedIdentifier(
                entity_type="Email",
                canonical=m.group(0).lower(),
                original=m.group(0),
                span=m.span(),
            )
        )
    return out


# ── INN (10 or 12 digits with checksum) ──────────────────────────────

_INN_RE = re.compile(r"(?<!\d)(\d{10}|\d{12})(?!\d)")


def _check_inn_10(d: str) -> bool:
    coeffs = (2, 4, 10, 3, 5, 9, 4, 6, 8)
    s = sum(int(d[i]) * coeffs[i] for i in range(9))
    return (s % 11) % 10 == int(d[9])


def _check_inn_12(d: str) -> bool:
    c1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    c2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0)
    n11 = (sum(int(d[i]) * c1[i] for i in range(10)) % 11) % 10
    n12 = (sum(int(d[i]) * c2[i] for i in range(11)) % 11) % 10
    return n11 == int(d[10]) and n12 == int(d[11])


def _extract_inns(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    for m in _INN_RE.finditer(text):
        d = m.group(1)
        valid = (len(d) == 10 and _check_inn_10(d)) or (
            len(d) == 12 and _check_inn_12(d)
        )
        if valid:
            out.append(
                NormalizedIdentifier(
                    entity_type="INN",
                    canonical=d,
                    original=d,
                    span=m.span(1),
                )
            )
    return out


# ── OGRN (13 or 15 digits with checksum) ─────────────────────────────

_OGRN_RE = re.compile(r"(?<!\d)(\d{13}|\d{15})(?!\d)")


def _check_ogrn_13(d: str) -> bool:
    return int(d[12]) == int(d[:12]) % 11 % 10


def _check_ogrn_15(d: str) -> bool:
    return int(d[14]) == int(d[:14]) % 13 % 10


def _extract_ogrn(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    for m in _OGRN_RE.finditer(text):
        d = m.group(1)
        valid = (len(d) == 13 and _check_ogrn_13(d)) or (
            len(d) == 15 and _check_ogrn_15(d)
        )
        if valid:
            out.append(
                NormalizedIdentifier(
                    entity_type="OGRN",
                    canonical=d,
                    original=d,
                    span=m.span(1),
                )
            )
    return out


# ── BIC (РФ: 9 digits, the first two are 04) ────────────────────────

_BIC_RE = re.compile(r"\b04\d{7}\b")


def _extract_bic(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    for m in _BIC_RE.finditer(text):
        out.append(
            NormalizedIdentifier(
                entity_type="BIC",
                canonical=m.group(0),
                original=m.group(0),
                span=m.span(),
            )
        )
    return out


# ── ContractNumber ───────────────────────────────────────────────────
#
# Heuristic: ``№`` (or ``No.`` / ``N``) followed by 2-30 alphanumeric +
# separator characters. The marker is required to avoid extracting
# every dashed alphanumeric token in the document (would catch a lot
# of noise — invoice numbers without context, version strings, etc.).

_CONTRACT_RE = re.compile(
    r"(?:№|\bNo\.?\b)\s*([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9./\-]{1,29})",
    re.IGNORECASE,
)


def _canonicalize_contract(raw: str) -> str:
    """Uppercase + strip whitespace; preserve / - . separators."""
    return raw.upper().replace(" ", "")


def _extract_contracts(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    for m in _CONTRACT_RE.finditer(text):
        original = m.group(1)
        out.append(
            NormalizedIdentifier(
                entity_type="ContractNumber",
                canonical=_canonicalize_contract(original),
                original=original,
                span=m.span(1),
            )
        )
    return out


# ── DocumentDate ─────────────────────────────────────────────────────

_DATE_DMY_RE = re.compile(
    r"(?<!\d)(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})(?!\d)"
)
_DATE_ISO_RE = re.compile(
    r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)"
)
_DATE_VERBAL_RE = re.compile(
    r"\b\d{1,2}\s+(?:янв(?:ар(?:я|ь)|\.)?|фев(?:рал(?:я|ь)|\.)?|"
    r"мар(?:та|т)|апр(?:ел(?:я|ь)|\.)?|мая|май|июн(?:я|ь)|июл(?:я|ь)|"
    r"авг(?:уста|уст|\.)?|сент(?:ябр(?:я|ь)|\.)?|окт(?:ябр(?:я|ь)|\.)?|"
    r"ноя(?:бр(?:я|ь)|\.)?|дек(?:абр(?:я|ь)|\.)?)"
    r"\s+\d{4}(?:\s*(?:г\.?|года))?\b",
    re.IGNORECASE,
)


def _extract_dates(text: str) -> list[NormalizedIdentifier]:
    """Three flavours: ISO (cheap strptime), DMY-numeric, RU verbal.

    ``dateparser`` with ``DATE_ORDER=DMY`` mis-parses ISO YMD strings;
    we strptime ISO directly to side-step that.
    """
    from datetime import datetime

    out: list[NormalizedIdentifier] = []
    seen: set[tuple[int, int]] = set()

    for m in _DATE_ISO_RE.finditer(text):
        span = m.span()
        try:
            parsed = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            continue
        seen.add(span)
        out.append(
            NormalizedIdentifier(
                entity_type="DocumentDate",
                canonical=parsed.strftime("%Y-%m-%d"),
                original=m.group(0),
                span=span,
            )
        )

    for regex, dmy in ((_DATE_DMY_RE, True), (_DATE_VERBAL_RE, False)):
        for m in regex.finditer(text):
            span = m.span()
            if span in seen:
                continue
            try:
                parsed_dt = dateparser.parse(
                    m.group(0),
                    languages=["ru"],
                    settings={"DATE_ORDER": "DMY"} if dmy else None,
                )
            except Exception:  # noqa: BLE001 — dateparser quirks
                continue
            if not parsed_dt:
                continue
            seen.add(span)
            out.append(
                NormalizedIdentifier(
                    entity_type="DocumentDate",
                    canonical=parsed_dt.strftime("%Y-%m-%d"),
                    original=m.group(0),
                    span=span,
                )
            )
    return out


# ── Amount (Russian rubles, RUB, ₽) ──────────────────────────────────

_AMOUNT_RE = re.compile(
    r"(\d[\d\s ]*(?:[.,]\d{1,2})?)\s*"
    r"(?:(млн|тыс|млрд)\.?\s*)?"
    r"(?:руб(?:лей|ля|\.)?|РУБ|RUB|₽)",
    re.IGNORECASE,
)
_AMOUNT_MULT = {"тыс": 1_000, "млн": 1_000_000, "млрд": 1_000_000_000}


def _extract_amounts(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    for m in _AMOUNT_RE.finditer(text):
        raw_num = (
            m.group(1)
            .replace(" ", "")
            .replace(" ", "")
            .replace(",", ".")
        )
        mult_word = (m.group(2) or "").lower()
        try:
            value = float(raw_num)
        except ValueError:
            continue
        if mult_word in _AMOUNT_MULT:
            value *= _AMOUNT_MULT[mult_word]
        out.append(
            NormalizedIdentifier(
                entity_type="Amount",
                canonical=f"{value:.2f} RUB",
                original=m.group(0),
                span=m.span(),
            )
        )
    return out


# ── PostalAddress ────────────────────────────────────────────────────
#
# Detect: 6-digit postal code anchors an address.  Window forward up to
# 200 chars (or to first newline) to capture city/street/house tokens.
# Skip windows that don't contain at least one street/city marker —
# avoids matching random 6-digit numbers (e.g. order numbers).

_POSTAL_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_ADDR_WINDOW = 200
_ADDR_MARKER_RE = re.compile(
    r"\b(г\.|город|Москв|Санкт-Петербург|Питер|ул\.|улица|пр-кт|пер\.|"
    r"проспект|переулок|обл\.|область|край|респ\.|республика|"
    r"д\.|дом\s)",
    re.IGNORECASE,
)

# Rule-based abbreviation expansion for the rule-only fallback path.
# Lowercased, stripped of dots/commas, normalized whitespace.
_ABBR_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bг\.\s*", re.IGNORECASE), ""),
    (re.compile(r"\bгород\s+", re.IGNORECASE), ""),
    (re.compile(r"\bул\.\s*", re.IGNORECASE), "ул "),
    (re.compile(r"\bулица\s+", re.IGNORECASE), "ул "),
    (re.compile(r"\bпр-кт\.?\s*", re.IGNORECASE), "пр "),
    (re.compile(r"\bпроспект\s+", re.IGNORECASE), "пр "),
    (re.compile(r"\bпер\.\s*", re.IGNORECASE), "пер "),
    (re.compile(r"\bпереулок\s+", re.IGNORECASE), "пер "),
    (re.compile(r"\bд\.\s*", re.IGNORECASE), ""),
    (re.compile(r"\bдом\s+", re.IGNORECASE), ""),
    (re.compile(r"\bстр\.\s*", re.IGNORECASE), "стр "),
    (re.compile(r"\bстроение\s+", re.IGNORECASE), "стр "),
    (re.compile(r"\bкорп\.\s*", re.IGNORECASE), "к "),
    (re.compile(r"\bкорпус\s+", re.IGNORECASE), "к "),
    (re.compile(r"\bкв\.\s*", re.IGNORECASE), "кв "),
    (re.compile(r"\bквартира\s+", re.IGNORECASE), "кв "),
    (re.compile(r"\bобл\.\s*", re.IGNORECASE), "обл "),
    (re.compile(r"\bобласть\s+", re.IGNORECASE), "обл "),
    (re.compile(r"\bр-н\.?\s*", re.IGNORECASE), "р-н "),
    (re.compile(r"\bрайон\s+", re.IGNORECASE), "р-н "),
)


def _normalize_address_rule(raw: str) -> str:
    """Lowercase + abbreviation expansion + whitespace cleanup."""
    s = raw.lower()
    for pattern, repl in _ABBR_EXPANSIONS:
        s = pattern.sub(repl, s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;")
    s = re.sub(r"\s*,\s*", ", ", s)
    return s


def _normalize_address(raw: str) -> str:
    """libpostal-based parse → structured fields → canonical assembly.

    Falls back to ``_normalize_address_rule`` when libpostal is missing
    or raises.  When libpostal IS available it transliterates Russian
    addresses by default — we use ``parse_address`` (structured) rather
    than ``expand_address`` (returns transliterated alternatives) to
    keep the canonical in Russian script.
    """
    if not _HAS_LIBPOSTAL or _libpostal_parse is None:
        return _normalize_address_rule(raw)
    try:
        parsed = _libpostal_parse(raw)
        fields: dict[str, str] = {label: value for value, label in parsed}
        parts: list[str] = []
        for label in ("postcode", "city", "road", "house_number", "unit"):
            v = fields.get(label)
            if v:
                parts.append(v.lower())
        if not parts:
            return _normalize_address_rule(raw)
        return ", ".join(parts)
    except Exception:  # noqa: BLE001 — libpostal C errors are opaque
        return _normalize_address_rule(raw)


def _extract_addresses(text: str) -> list[NormalizedIdentifier]:
    out: list[NormalizedIdentifier] = []
    seen_spans: set[tuple[int, int]] = set()
    for m in _POSTAL_CODE_RE.finditer(text):
        start = m.start()
        end = min(len(text), m.end() + _ADDR_WINDOW)
        window = text[start:end]
        nl = window.find("\n")
        if nl >= 0:
            window = window[:nl]
            end = start + nl
        if not _ADDR_MARKER_RE.search(window):
            continue
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        cleaned = window.strip().rstrip(",.;")
        out.append(
            NormalizedIdentifier(
                entity_type="PostalAddress",
                canonical=_normalize_address(cleaned),
                original=cleaned,
                span=span,
            )
        )
    return out


# ── public aggregator ────────────────────────────────────────────────


def extract_identifiers(text: str) -> list[NormalizedIdentifier]:
    """Run every detector on ``text``; return matches sorted by span.

    Multiple occurrences of the same canonical form ARE returned (each
    with its own span). Deduplication for graph injection is the
    integration layer's responsibility (Stage C).
    """
    if not text:
        return []
    found: list[NormalizedIdentifier] = []
    found.extend(_extract_phones(text))
    found.extend(_extract_emails(text))
    found.extend(_extract_inns(text))
    found.extend(_extract_ogrn(text))
    found.extend(_extract_bic(text))
    found.extend(_extract_contracts(text))
    found.extend(_extract_dates(text))
    found.extend(_extract_amounts(text))
    found.extend(_extract_addresses(text))
    found.sort(key=lambda x: x.span)
    return found


# ── Stage-C helpers: payload + augment block builders ───────────────


def dedupe_by_canonical(
    idents: list[NormalizedIdentifier],
) -> list[NormalizedIdentifier]:
    """Keep first occurrence per (entity_type, canonical) pair.

    Preserves source order so the first textual mention wins for
    ``original``/``span`` fields used in descriptions.
    """
    seen: set[tuple[str, str]] = set()
    out: list[NormalizedIdentifier] = []
    for ident in idents:
        key = (ident.entity_type, ident.canonical)
        if key in seen:
            continue
        seen.add(key)
        out.append(ident)
    return out


def build_custom_kg_payload(
    idents: list[NormalizedIdentifier],
    *,
    doc_id: str,
    file_path: str,
    text: str = "",
    snippet_window: int = 80,
) -> dict:
    """Assemble a ``rag.ainsert_custom_kg`` payload from identifier matches.

    One entity per (entity_type, canonical) — duplicates within the doc
    collapse to a single node, but ``ainsert_custom_kg`` is itself
    idempotent across documents (descriptions accumulate when the same
    canonical is inserted from multiple ``source_id``).

    ``description`` includes the verbatim original form and a small
    surrounding snippet so the node carries provenance once it lands in
    Neo4j — useful both for the LLM (when it later relates entities
    via ``ainsert(text)``) and for human auditing.
    """
    entities: list[dict] = []
    for ident in dedupe_by_canonical(idents):
        snippet = ""
        if text:
            start = max(0, ident.span[0] - snippet_window)
            end = min(len(text), ident.span[1] + snippet_window)
            snippet = text[start:end].replace("\n", " ").strip()
        if ident.original != ident.canonical:
            desc = (
                f"{ident.entity_type} извлечён из документа {doc_id}; "
                f"канонический вид: {ident.canonical}; в тексте: "
                f"«{ident.original}»."
            )
        else:
            desc = (
                f"{ident.entity_type} извлечён из документа {doc_id}; "
                f"в тексте: «{ident.original}»."
            )
        if snippet:
            desc += f" Контекст: «…{snippet}…»."
        entities.append(
            {
                "entity_name": ident.canonical,
                "entity_type": ident.entity_type,
                "description": desc,
                "source_id": doc_id,
                "file_path": file_path,
            }
        )
    return {
        "chunks": [],
        "entities": entities,
        "relationships": [],
    }


_AUGMENT_HEADER = (
    "\n\n---\n"
    "Канонические идентификаторы (используй ИМЕННО ТАКУЮ форму в "
    "entity_name):\n"
)


def build_augment_block(idents: list[NormalizedIdentifier]) -> str:
    """Format the canonical-identifiers block appended to document text.

    Produces empty string when there are no identifiers — the caller
    should skip appending altogether in that case.
    """
    deduped = dedupe_by_canonical(idents)
    if not deduped:
        return ""
    lines: list[str] = []
    for ident in deduped:
        if ident.original != ident.canonical:
            lines.append(
                f"- {ident.entity_type}: {ident.canonical} "
                f"(в тексте: «{ident.original}»)"
            )
        else:
            lines.append(f"- {ident.entity_type}: {ident.canonical}")
    return _AUGMENT_HEADER + "\n".join(lines) + "\n"


__all__ = [
    "IdentifierType",
    "NormalizedIdentifier",
    "build_augment_block",
    "build_custom_kg_payload",
    "dedupe_by_canonical",
    "extract_identifiers",
]

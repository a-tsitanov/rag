"""Unit tests for ``src/ingestion/identifiers.py``.

Coverage goals:
  * One happy-path case per identifier type.
  * Edge cases that previously broke real documents (mixed separators,
    unicode quotes, NBSP, RU month names).
  * Validation: invalid INN/OGRN checksums must NOT be returned —
    that filter is the only thing keeping random 10/13-digit numbers
    out of the graph.
  * Integration: a realistic Russian contract excerpt yields the full
    set of canonical identifiers in span order.

Tests do not require libpostal — when it isn't installed, the address
normalizer falls back to the rule layer (which these tests exercise).
"""

from __future__ import annotations

import pytest

from src.ingestion.identifiers import (
    NormalizedIdentifier,
    build_augment_block,
    build_custom_kg_payload,
    dedupe_by_canonical,
    extract_identifiers,
)


# Real-world reference values used across tests.
SBER_INN = "7707083893"           # ИНН Сбербанка (10 digits, valid checksum)
GAZPROM_INN = "7736050003"        # ИНН Газпрома (10 digits, valid checksum)
INDIV_INN = "500400123402"        # 12-digit INN (computed valid checksum)
SBER_OGRN = "1027700132195"       # ОГРН Сбербанка (13, valid)
IP_OGRN = "304500116000157"       # ОГРНИП (15, valid)
SBER_BIC = "044525225"            # БИК Сбербанк ОПЕРУ Москва


def _by_type(
    items: list[NormalizedIdentifier],
    t: str,
) -> list[NormalizedIdentifier]:
    return [x for x in items if x.entity_type == t]


# ── PhoneNumber ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected_canonical",
    [
        ("+7 (495) 234-56-78", "+74952345678"),
        ("+7 495 234 56 78", "+74952345678"),
        ("8 (495) 234-56-78", "+74952345678"),
        ("+7-495-234-56-78", "+74952345678"),
    ],
)
def test_phone_e164(raw: str, expected_canonical: str) -> None:
    found = _by_type(extract_identifiers(f"Контактный телефон: {raw}"), "PhoneNumber")
    assert len(found) == 1
    assert found[0].canonical == expected_canonical
    assert found[0].original.strip().startswith(("+", "8"))


def test_phone_no_match_when_no_phone_in_text() -> None:
    found = _by_type(
        extract_identifiers("В документе нет ни одного телефона."),
        "PhoneNumber",
    )
    assert found == []


# ── Email ────────────────────────────────────────────────────────────


def test_email_lowercased_canonical() -> None:
    text = "Контакт: I.Ivanov@SEVTECH.ru, копия — Bob@Example.COM"
    emails = _by_type(extract_identifiers(text), "Email")
    canonicals = sorted(e.canonical for e in emails)
    assert canonicals == ["bob@example.com", "i.ivanov@sevtech.ru"]
    # original case preserved
    assert emails[0].original == "I.Ivanov@SEVTECH.ru"


# ── INN ──────────────────────────────────────────────────────────────


def test_inn_10_valid_checksum_extracted() -> None:
    text = f"ООО «Тест» (ИНН {SBER_INN}) поставщик."
    inns = _by_type(extract_identifiers(text), "INN")
    assert len(inns) == 1
    assert inns[0].canonical == SBER_INN


def test_inn_10_invalid_checksum_rejected() -> None:
    # 7707083894 — last digit changed to break checksum
    text = "ИНН 7707083894 — неправильная контрольная сумма."
    inns = _by_type(extract_identifiers(text), "INN")
    assert inns == []


def test_inn_12_valid_extracted() -> None:
    text = f"ИП Иванов И.И., ИНН {INDIV_INN}"
    inns = _by_type(extract_identifiers(text), "INN")
    assert len(inns) == 1
    assert inns[0].canonical == INDIV_INN


def test_inn_does_not_match_inside_longer_digit_run() -> None:
    # 14-digit number — neither 10 nor 12 — must not match
    text = "Code 12345678901234 some text."
    inns = _by_type(extract_identifiers(text), "INN")
    assert inns == []


# ── OGRN ─────────────────────────────────────────────────────────────


def test_ogrn_13_valid_extracted() -> None:
    text = f"ПАО Сбербанк, ОГРН {SBER_OGRN}, действует на основании устава."
    ogrns = _by_type(extract_identifiers(text), "OGRN")
    assert len(ogrns) == 1
    assert ogrns[0].canonical == SBER_OGRN


def test_ogrn_15_valid_extracted() -> None:
    text = f"ИП Петров П.П., ОГРНИП {IP_OGRN}"
    ogrns = _by_type(extract_identifiers(text), "OGRN")
    assert len(ogrns) == 1
    assert ogrns[0].canonical == IP_OGRN


def test_ogrn_invalid_checksum_rejected() -> None:
    text = "ОГРН 1027700132190 — последняя цифра неверна."
    ogrns = _by_type(extract_identifiers(text), "OGRN")
    assert ogrns == []


# ── BIC ──────────────────────────────────────────────────────────────


def test_bic_extracted() -> None:
    text = f"Банк получателя: ПАО Сбербанк, БИК {SBER_BIC}"
    bics = _by_type(extract_identifiers(text), "BIC")
    assert len(bics) == 1
    assert bics[0].canonical == SBER_BIC


def test_bic_does_not_match_non_04_prefix() -> None:
    # Russian BICs start with 04 — 12-prefixed 9-digit must not match
    text = "Random number 123456789 — not a BIC."
    bics = _by_type(extract_identifiers(text), "BIC")
    assert bics == []


# ── ContractNumber ───────────────────────────────────────────────────


def test_contract_with_no_marker_extracted_uppercase() -> None:
    text = "Договор поставки № дп-2024/178-К от 15.03.2024."
    contracts = _by_type(extract_identifiers(text), "ContractNumber")
    assert len(contracts) == 1
    assert contracts[0].canonical == "ДП-2024/178-К"
    assert contracts[0].original == "дп-2024/178-К"


def test_contract_no_marker_no_match() -> None:
    # Without №/No prefix nothing should match the contract pattern
    text = "Some random ABC-123 token without a contract marker."
    contracts = _by_type(extract_identifiers(text), "ContractNumber")
    assert contracts == []


# ── DocumentDate ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected_iso",
    [
        ("15.03.2024", "2024-03-15"),
        ("15/03/2024", "2024-03-15"),
        ("15-03-2024", "2024-03-15"),
        ("2024-03-15", "2024-03-15"),
    ],
)
def test_date_numeric(raw: str, expected_iso: str) -> None:
    text = f"Дата подписания: {raw}."
    dates = _by_type(extract_identifiers(text), "DocumentDate")
    assert len(dates) == 1
    assert dates[0].canonical == expected_iso


def test_date_verbal_russian() -> None:
    text = "Заключён 15 марта 2024 года в Москве."
    dates = _by_type(extract_identifiers(text), "DocumentDate")
    assert len(dates) >= 1
    assert any(d.canonical == "2024-03-15" for d in dates)


# ── Amount ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected_canonical",
    [
        ("4 250 000,00 руб.", "4250000.00 RUB"),
        ("1500000 руб", "1500000.00 RUB"),
        ("4,25 млн руб", "4250000.00 RUB"),
        ("125 тыс. руб.", "125000.00 RUB"),
        ("99,99 ₽", "99.99 RUB"),
    ],
)
def test_amount(raw: str, expected_canonical: str) -> None:
    text = f"Сумма договора: {raw}."
    amounts = _by_type(extract_identifiers(text), "Amount")
    assert len(amounts) >= 1
    assert any(a.canonical == expected_canonical for a in amounts)


def test_amount_no_match_without_currency() -> None:
    text = "Просто число 4 250 000 без валюты."
    amounts = _by_type(extract_identifiers(text), "Amount")
    assert amounts == []


# ── PostalAddress ────────────────────────────────────────────────────


def test_postal_address_extracted_with_postcode_and_marker() -> None:
    text = (
        "Юридический адрес: 127015, г. Москва, "
        "ул. Бутырская, д. 76, стр. 1."
    )
    addrs = _by_type(extract_identifiers(text), "PostalAddress")
    assert len(addrs) == 1
    canonical = addrs[0].canonical
    # Rule layer outputs lowercased, abbreviation-expanded form
    assert "127015" in canonical
    assert "москва" in canonical.lower() or "ул бутырская" in canonical


def test_postal_code_alone_not_extracted_without_marker() -> None:
    # 6-digit number with no city/street markers nearby — not an address
    text = "Заказ № 127015 от поставщика на сумму 100 руб."
    addrs = _by_type(extract_identifiers(text), "PostalAddress")
    assert addrs == []


# ── integration ──────────────────────────────────────────────────────


def test_integration_full_contract_excerpt() -> None:
    text = (
        f"Договор поставки № ДП-2024/178-К от 15.03.2024 заключён между "
        f"ООО «Северные технологии» (ИНН {SBER_INN}, ОГРН {SBER_OGRN}, "
        f"юр. адрес: 127015, г. Москва, ул. Бутырская, д. 76, стр. 1) "
        f"и АО «Промсервис».\n"
        f"Контактное лицо: Иванов Иван Петрович, "
        f"телефон +7 (495) 234-56-78, e-mail: i.ivanov@sevtech.ru. "
        f"Сумма договора: 4 250 000,00 руб. "
        f"Банк получателя: ПАО Сбербанк, БИК {SBER_BIC}."
    )

    found = extract_identifiers(text)
    by_type: dict[str, list[NormalizedIdentifier]] = {}
    for f in found:
        by_type.setdefault(f.entity_type, []).append(f)

    canonicals = {t: [x.canonical for x in lst] for t, lst in by_type.items()}

    # Sanity: every type we emitted shows up at least once with the
    # right canonical
    assert "ДП-2024/178-К" in canonicals.get("ContractNumber", [])
    assert "2024-03-15" in canonicals.get("DocumentDate", [])
    assert SBER_INN in canonicals.get("INN", [])
    assert SBER_OGRN in canonicals.get("OGRN", [])
    assert "+74952345678" in canonicals.get("PhoneNumber", [])
    assert "i.ivanov@sevtech.ru" in canonicals.get("Email", [])
    assert "4250000.00 RUB" in canonicals.get("Amount", [])
    assert SBER_BIC in canonicals.get("BIC", [])
    assert any(
        "127015" in c for c in canonicals.get("PostalAddress", [])
    )

    # spans must be sorted (sanity for Stage C augment-block assembly)
    spans = [f.span for f in found]
    assert spans == sorted(spans)


def test_empty_text_returns_empty() -> None:
    assert extract_identifiers("") == []


def test_text_without_identifiers_returns_empty() -> None:
    text = "Это просто абзац без каких-либо структурных данных."
    assert extract_identifiers(text) == []


# ── Stage-C helpers ──────────────────────────────────────────────────


def test_dedupe_by_canonical_keeps_first_occurrence() -> None:
    a = NormalizedIdentifier("PhoneNumber", "+74952345678", "+7 495 234 56 78", (0, 18))
    b = NormalizedIdentifier("PhoneNumber", "+74952345678", "8 495 234 56 78", (50, 65))
    c = NormalizedIdentifier("Email", "x@y.ru", "x@y.ru", (70, 76))
    out = dedupe_by_canonical([a, b, c])
    assert len(out) == 2
    # first occurrence kept
    assert out[0] is a
    assert out[1] is c


def test_build_custom_kg_payload_structure() -> None:
    text = "Тел: +7 495 234 56 78, e-mail: x@y.ru."
    idents = extract_identifiers(text)
    assert idents
    payload = build_custom_kg_payload(
        idents, doc_id="doc-42", file_path="/docs/x.txt", text=text,
    )
    assert payload["chunks"] == []
    assert payload["relationships"] == []
    types = {e["entity_type"] for e in payload["entities"]}
    assert "PhoneNumber" in types
    assert "Email" in types
    for ent in payload["entities"]:
        assert ent["source_id"] == "doc-42"
        assert ent["file_path"] == "/docs/x.txt"
        assert ent["entity_name"]
        assert ent["description"]
        # description mentions doc id + an original or canonical form
        assert "doc-42" in ent["description"]


def test_build_custom_kg_payload_empty_when_no_idents() -> None:
    payload = build_custom_kg_payload([], doc_id="d", file_path="f")
    assert payload == {"chunks": [], "entities": [], "relationships": []}


def test_build_custom_kg_payload_dedupes_within_doc() -> None:
    a = NormalizedIdentifier("INN", "7707083893", "7707083893", (10, 20))
    b = NormalizedIdentifier("INN", "7707083893", "7707083893", (50, 60))  # dup
    payload = build_custom_kg_payload([a, b], doc_id="d", file_path="f")
    assert len(payload["entities"]) == 1


def test_build_augment_block_format() -> None:
    idents = [
        NormalizedIdentifier("PhoneNumber", "+74952345678", "+7 495 234 56 78", (0, 18)),
        NormalizedIdentifier("INN", "7707083893", "7707083893", (20, 30)),
    ]
    block = build_augment_block(idents)
    assert "Канонические идентификаторы" in block
    assert "+74952345678" in block
    assert "7707083893" in block
    # original differs from canonical → annotation present
    assert "в тексте: «+7 495 234 56 78»" in block
    # original equals canonical → no annotation noise
    assert "(в тексте: «7707083893»)" not in block


def test_build_augment_block_empty_input() -> None:
    assert build_augment_block([]) == ""

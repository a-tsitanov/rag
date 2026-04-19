#!/usr/bin/env bash
# Enterprise KB — curl-тесты для всех API endpoints.
#
# Использование:
#   bash scripts/test_api.sh                # все тесты
#   bash scripts/test_api.sh health         # только health
#   bash scripts/test_api.sh ingest         # загрузить тестовый файл
#   bash scripts/test_api.sh search         # все виды поиска
#   bash scripts/test_api.sh search-basic   # только базовый поиск

set -euo pipefail

BASE="http://localhost:8000"
KEY="dev-key-change-me"
H_AUTH="X-API-Key: $KEY"
H_JSON="Content-Type: application/json"

# ── helpers ──────────────────────────────────────────────────────────

sep() { echo -e "\n━━━ $1 ━━━"; }
run() { echo -e "\n\$ $1"; eval "$1"; echo; }

# ── 1. Health ────────────────────────────────────────────────────────

test_health() {
  sep "GET /health"
  run 'curl -s "$BASE/health" | python3 -m json.tool'
}

# ── 2. Ingest ────────────────────────────────────────────────────────

test_ingest() {
  # Создать тестовый файл если нет
  TEST_FILE="/tmp/kb-test-doc.md"
  cat > "$TEST_FILE" << 'DOCEOF'
# Kubernetes Network Policies

## Overview
Kubernetes Network Policies control traffic flow between pods.
The Calico CNI plugin is used in the security department for network isolation.

## Cost
Calico Enterprise license costs $50,000 per year for up to 100 nodes.
The budget is managed by the infrastructure team in collaboration with security.

## Implementation
Network policies are enforced at the node level using eBPF.
Each department has its own namespace with default-deny policies.
DOCEOF

  sep "POST /api/v1/ingest — upload test document"
  RESP=$(curl -s -X POST "$BASE/api/v1/ingest" \
    -H "$H_AUTH" \
    -F "file=@$TEST_FILE" \
    -F "department=security" \
    -F "priority=high")
  echo "$RESP" | python3 -m json.tool

  JOB_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")

  if [ -n "$JOB_ID" ]; then
    sep "GET /api/v1/ingest/$JOB_ID — poll status"
    for i in 1 2 3 4 5; do
      sleep 3
      STATUS_RESP=$(curl -s "$BASE/api/v1/ingest/$JOB_ID" -H "$H_AUTH")
      STATUS=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
      echo "  poll $i: status=$STATUS"
      if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
        echo "$STATUS_RESP" | python3 -m json.tool
        break
      fi
    done
  fi
}

# ── 3. Search ────────────────────────────────────────────────────────

test_search_basic() {
  sep "POST /api/v1/search — basic hybrid"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{"query": "What network plugins are used?"}'"'"' \
    | python3 -m json.tool'
}

test_search_filters() {
  sep "POST /api/v1/search — with metadata filters"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{
      "query": "network policy",
      "department": "security",
      "doc_type_filter": "md",
      "top_k": 5
    }'"'"' \
    | python3 -m json.tool'
}

test_search_modes() {
  sep "POST /api/v1/search — naive mode (vector only, no graph)"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{"query": "Calico CNI cost", "mode": "naive", "top_k": 3}'"'"' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"mode={r[\"mode\"]} sources={len(r[\"sources\"])} answer={r[\"answer\"][:100]}...\")"'

  sep "POST /api/v1/search — local mode (entity-focused)"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{"query": "Calico CNI cost", "mode": "local", "top_k": 3}'"'"' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"mode={r[\"mode\"]} sources={len(r[\"sources\"])} answer={r[\"answer\"][:100]}...\")"'

  sep "POST /api/v1/search — global mode (relationship-focused)"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{"query": "Calico CNI cost", "mode": "global", "top_k": 3}'"'"' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"mode={r[\"mode\"]} sources={len(r[\"sources\"])} answer={r[\"answer\"][:100]}...\")"'
}

test_search_knobs() {
  sep "POST /api/v1/search — LightRAG QueryParam knobs"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{
      "query": "budget allocation",
      "response_type": "List",
      "include_references": true,
      "chunk_top_k": 30,
      "max_entity_tokens": 8000,
      "max_total_tokens": 40000
    }'"'"' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"sources={len(r[\"sources\"])}\\nanswer={r[\"answer\"][:200]}\")"'
}

test_search_decompose() {
  sep "POST /api/v1/search — query decomposition (Phase 2b)"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{
      "query": "How does K8s networking affect security budget across departments?",
      "decompose": true,
      "top_k": 5
    }'"'"' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"sub_queries={r.get(\"sub_queries\")}\\nsources={len(r[\"sources\"])}\\nanswer={r[\"answer\"][:200]}\")"'
}

test_search_agentic() {
  sep "POST /api/v1/search — agentic multi-hop (Phase 4)"
  run 'curl -s -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{
      "query": "What network plugins does the security department use and how much do they cost?",
      "agentic": true,
      "agentic_max_rounds": 3,
      "top_k": 5
    }'"'"' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"rounds={r.get(\"agentic_rounds\")}\\nfollow_ups={r.get(\"follow_up_queries\")}\\nsources={len(r[\"sources\"])}\\nanswer={r[\"answer\"][:300]}\")"'
}

# ── 4. Error cases ───────────────────────────────────────────────────

test_errors() {
  sep "401 — missing API key"
  run 'curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/v1/search" \
    -H "$H_JSON" \
    -d '"'"'{"query": "test"}'"'"''

  sep "403 — invalid API key"
  run 'curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/v1/search" \
    -H "X-API-Key: wrong-key" -H "$H_JSON" \
    -d '"'"'{"query": "test"}'"'"''

  sep "422 — missing required field"
  run 'curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/v1/search" \
    -H "$H_AUTH" -H "$H_JSON" \
    -d '"'"'{"mode": "hybrid"}'"'"''

  sep "404 — unknown job_id"
  run 'curl -s -o /dev/null -w "%{http_code}" \
    "$BASE/api/v1/ingest/00000000-0000-0000-0000-000000000000" \
    -H "$H_AUTH"'
}

# ── dispatcher ───────────────────────────────────────────────────────

case "${1:-all}" in
  health)         test_health ;;
  ingest)         test_ingest ;;
  search-basic)   test_search_basic ;;
  search-filters) test_search_filters ;;
  search-modes)   test_search_modes ;;
  search-knobs)   test_search_knobs ;;
  search-decompose) test_search_decompose ;;
  search-agentic) test_search_agentic ;;
  search)         test_search_basic; test_search_filters; test_search_modes;
                  test_search_knobs; test_search_decompose; test_search_agentic ;;
  errors)         test_errors ;;
  all)            test_health; test_ingest; test_search_basic; test_search_filters;
                  test_search_modes; test_search_knobs; test_search_decompose;
                  test_search_agentic; test_errors ;;
  *)              echo "Usage: $0 {health|ingest|search|search-basic|search-filters|search-modes|search-knobs|search-decompose|search-agentic|errors|all}"; exit 1 ;;
esac

echo -e "\n✓ Done"

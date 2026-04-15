#!/usr/bin/env bash
# Smoke-тест API Enterprise KB через curl.
#
# Запуск:
#   ./scripts/smoke.sh                  # весь сценарий
#   ./scripts/smoke.sh health           # только health
#   ./scripts/smoke.sh ingest           # upload + poll
#   ./scripts/smoke.sh search           # серия поисковых запросов
#   ./scripts/smoke.sh errors           # проверка 401 / 403 / 404
#
# Предусловия:
#   docker compose -f docker-compose.dev.yml up -d
#   python scripts/setup_db.py
#   # в двух других терминалах (или через .vscode/launch.json):
#   uvicorn src.api.main:app --port 8000
#   taskiq worker src.ingestion.tasks:broker

set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-dev-local-key}"
SAMPLE_FILE="${SAMPLE_FILE:-tests/test_ingestion/fixtures/sample.txt}"

# jq доступен? иначе плоский вывод
if command -v jq >/dev/null 2>&1; then
  JQ="jq"
else
  JQ="cat"
fi

# ── helpers ──────────────────────────────────────────────────────────

header() {
  printf "\n\033[1;36m▶ %s\033[0m\n" "$1"
}

dim() {
  printf "  \033[2m%s\033[0m\n" "$1"
}

# ── 1. health ────────────────────────────────────────────────────────

smoke_health() {
  header "GET /health  (public — no auth)"
  dim "curl $BASE_URL/health"
  curl -s "$BASE_URL/health" | $JQ
}

# ── 2. ingest: upload + poll ─────────────────────────────────────────

smoke_ingest() {
  header "POST /api/v1/ingest  (multipart)"
  dim "curl -F file=@$SAMPLE_FILE -F department=demo -F priority=high ..."

  if [[ ! -f "$SAMPLE_FILE" ]]; then
    echo "sample file not found: $SAMPLE_FILE" >&2
    return 1
  fi

  local response
  response=$(curl -s -X POST "$BASE_URL/api/v1/ingest" \
    -H "X-API-Key: $API_KEY" \
    -F "file=@$SAMPLE_FILE" \
    -F "department=demo" \
    -F "priority=high")
  echo "$response" | $JQ

  local job_id
  job_id=$(echo "$response" | sed -E 's/.*"job_id":"([^"]+)".*/\1/')
  if [[ "$job_id" == "$response" ]] || [[ -z "$job_id" ]]; then
    echo "could not extract job_id from response" >&2
    return 1
  fi

  header "GET /api/v1/ingest/$job_id  (poll × 5)"
  dim "curl -H 'X-API-Key: $API_KEY' $BASE_URL/api/v1/ingest/$job_id"

  for i in 1 2 3 4 5; do
    printf "[%d/5] " "$i"
    curl -s "$BASE_URL/api/v1/ingest/$job_id" \
      -H "X-API-Key: $API_KEY" | $JQ -c '{status, error}'
    sleep 2
  done
}

# ── 3. search: разные режимы + фильтры ──────────────────────────────

smoke_search() {
  header "POST /api/v1/search  (mode=hybrid)"
  dim "базовый hybrid, top_k=5"
  curl -s -X POST "$BASE_URL/api/v1/search" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What document formats are supported by the knowledge base?",
      "mode": "hybrid",
      "top_k": 5
    }' | $JQ '{query, mode, answer, latency_ms, sources_count: (.sources|length)}'

  header "POST /api/v1/search  (mode=naive — только векторный поиск)"
  curl -s -X POST "$BASE_URL/api/v1/search" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "How are documents chunked before indexing?",
      "mode": "naive",
      "top_k": 3
    }' | $JQ '{query, mode, answer, latency_ms}'

  header "POST /api/v1/search  (department filter)"
  dim "запрашиваем только документы из department=demo"
  curl -s -X POST "$BASE_URL/api/v1/search" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "query": "Describe the ingestion pipeline",
      "mode": "hybrid",
      "department": "demo",
      "top_k": 5,
      "user_id": "smoke-test"
    }' | $JQ '{query, mode, latency_ms, sources: [.sources[] | {chunk_id, score, department}]}'

  header "POST /api/v1/search  (mode=local — graph)"
  curl -s -X POST "$BASE_URL/api/v1/search" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query": "Which entities are mentioned in ingested documents?", "mode": "local", "top_k": 5}' \
    | $JQ '{query, mode, answer, latency_ms}'

  header "POST /api/v1/search  (mode=global — high-level summary)"
  curl -s -X POST "$BASE_URL/api/v1/search" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query": "Give a high-level summary of the corpus", "mode": "global", "top_k": 5}' \
    | $JQ '{query, mode, answer, latency_ms}'
}

# ── 4. негативные проверки ────────────────────────────────────────────

smoke_errors() {
  header "POST /api/v1/search  без X-API-Key  (ожидается 401)"
  curl -s -o /dev/null -w "status=%{http_code}\n" \
    -X POST "$BASE_URL/api/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query":"ping"}'

  header "POST /api/v1/search  с невалидным ключом  (ожидается 403)"
  curl -s -o /dev/null -w "status=%{http_code}\n" \
    -X POST "$BASE_URL/api/v1/search" \
    -H "X-API-Key: definitely-wrong" \
    -H "Content-Type: application/json" \
    -d '{"query":"ping"}'

  header "GET /api/v1/ingest/00000000-0000-0000-0000-000000000000  (ожидается 404)"
  curl -s -o /dev/null -w "status=%{http_code}\n" \
    -H "X-API-Key: $API_KEY" \
    "$BASE_URL/api/v1/ingest/00000000-0000-0000-0000-000000000000"
}

# ── driver ───────────────────────────────────────────────────────────

case "${1:-all}" in
  health)  smoke_health ;;
  ingest)  smoke_ingest ;;
  search)  smoke_search ;;
  errors)  smoke_errors ;;
  all)     smoke_health && smoke_ingest && smoke_search && smoke_errors ;;
  *)
    echo "usage: $0 [health|ingest|search|errors|all]" >&2
    exit 2
    ;;
esac

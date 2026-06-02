#!/usr/bin/env bash
# check_logprobs.sh — проверяет какие провайдеры поддерживают logprobs

MODELS=(
  "qwen/qwen3-coder"
  "anthropic/claude-haiku-4.5"
  "anthropic/claude-sonnet-4.5"
  "openai/gpt-5-mini"
)

echo "Checking logprobs support across providers..."
echo "============================================"

for model in "${MODELS[@]}"; do
  author="${model%%/*}"
  slug="${model##*/}"

  echo ""
  echo "▶ $model"

  response=$(curl -s \
    "https://openrouter.ai/api/v1/models/${author}/${slug}/endpoints" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY")

  # Провайдеры С logprobs
  with=$(echo "$response" | jq -r '
    .data.endpoints[]
    | select(.supported_parameters | contains(["logprobs"]))
    | "  ✅ \(.provider_name) (\(.tag))  uptime_1d=\(.uptime_last_1d // "?")%"
  ')

  # Провайдеры БЕЗ logprobs
  without=$(echo "$response" | jq -r '
    .data.endpoints[]
    | select(.supported_parameters | contains(["logprobs"]) | not)
    | "  ❌ \(.provider_name) (\(.tag))"
  ')

  if [ -z "$with" ]; then
    echo "  ⚠️  No providers support logprobs for this model"
  else
    echo "$with"
  fi

  echo "  --- no logprobs ---"
  echo "$without"
done
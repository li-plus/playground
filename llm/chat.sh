#!/usr/bin/env bash

set -ex

# load .env
if [[ -f .env ]]; then
  set -o allexport
  source .env
  set +o allexport
fi

model=k2

base_url=${OPENAI_BASE_URL:-"https://api.openai.com/v1"}
api_key=${OPENAI_API_KEY}

curl ${base_url}/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${api_key}" \
  -d '{
    "model": "'${model}'",
    "messages": [
      {
        "role": "user",
        "content": "Count from 1 to 100 in a line separated by comma."
      }
    ]
  }'

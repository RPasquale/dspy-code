#!/bin/sh
set -e

# Generate runtime config.json into the web root so the SPA can read it if desired
API_BASE_URL_ESC=${API_BASE_URL:-}
WS_BASE_URL_ESC=${WS_BASE_URL:-}
cat > /usr/share/nginx/html/config.json <<EOF
{
  "apiBaseUrl": "${API_BASE_URL_ESC}",
  "wsBaseUrl": "${WS_BASE_URL_ESC}",
  "vectorizerEnabled": ${SHOW_VECTORIZER:-0}
}
EOF

exec nginx -g 'daemon off;'

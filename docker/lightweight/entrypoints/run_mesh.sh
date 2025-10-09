#!/bin/bash
set -euo pipefail

NODE_ID=${MESH_NODE_ID:-1001}
DOMAIN=${MESH_DOMAIN:-default}
DOMAIN_ID_ENV=${MESH_DOMAIN_ID:-}
LISTEN=${MESH_LISTEN_ADDR:-0.0.0.0:7000}
GRPC_LISTEN=${MESH_GRPC_LISTEN_ADDR:-0.0.0.0:50051}
PEERS=${MESH_PEERS:-}
METRICS=${MESH_METRICS_ADDR:-0.0.0.0:9100}
ENABLE_GRPC=${MESH_ENABLE_GRPC:-1}
EXTRA_ARGS=${MESH_EXTRA_ARGS:-}

is_uint() {
  [[ $1 =~ ^[0-9]+$ ]]
}

to_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_peer_addr() {
  local peer=$1
  local host=""
  local port=""
  local resolved_host=""
  if [ -z "${peer}" ]; then
    echo "empty peer entry" >&2
    return 1
  fi
  case "${peer}" in
    *:*)
      host=${peer%:*}
      port=${peer##*:}
      ;;
    *)
      echo "invalid peer address (expected host:port): ${peer}" >&2
      return 1
      ;;
  esac

  if [ -z "${host}" ] || [ -z "${port}" ]; then
    echo "invalid peer address (missing host or port): ${peer}" >&2
    return 1
  fi

  resolved_host="${host}"
  if [[ ! ${host} =~ ^[0-9.]+$ ]]; then
    if command -v getent >/dev/null 2>&1; then
      resolved_host=$(getent hosts "${host}" | awk '{print $1; exit}')
    fi
    if [ -z "${resolved_host}" ]; then
      echo "failed to resolve mesh peer host '${host}'" >&2
      return 1
    fi
  fi

  printf '%s:%s\n' "${resolved_host}" "${port}"
}

if [ -n "${DOMAIN_ID_ENV}" ]; then
  if ! is_uint "${DOMAIN_ID_ENV}"; then
    echo "invalid numeric value for MESH_DOMAIN_ID: ${DOMAIN_ID_ENV}" >&2
    exit 1
  fi
  DOMAIN_ID=${DOMAIN_ID_ENV}
elif is_uint "${DOMAIN}"; then
  DOMAIN_ID=${DOMAIN}
else
  # Fallback: deterministic hash for non-numeric domains.
  DOMAIN_ID=$(cksum <<<"${DOMAIN}" | awk '{print $1}')
fi

if [ ! -x /opt/mesh/bin/mesh ]; then
  echo "mesh binary not found" >&2
  exit 1
fi

CMD=(/opt/mesh/bin/mesh \
  --node-id "${NODE_ID}" \
  --listen "${LISTEN}" \
  --grpc-bind "${GRPC_LISTEN}" \
  --domain-id "${DOMAIN_ID}")

if to_bool "${ENABLE_GRPC}"; then
  CMD+=(--enable-grpc)
fi

IFS=','
for peer in $PEERS; do
  if [ -n "${peer}" ]; then
    resolved_peer=$(resolve_peer_addr "${peer}")
    CMD+=(--connect "${resolved_peer}")
  fi
done
unset IFS

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2086
  CMD+=(${EXTRA_ARGS})
fi

exec "${CMD[@]}"

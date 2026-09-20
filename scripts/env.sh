#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT/third_party/RLDP:$ROOT/third_party/offline_rl/src:${PYTHONPATH:-}"

echo "PYTHONPATH configured for RLDP reproduction"
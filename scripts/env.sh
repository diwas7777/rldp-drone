#!/usr/bin/env bash

if [ -n "${ZSH_VERSION:-}" ]; then
    ROOT="$(cd "$(dirname "${(%):-%x}")/.." && pwd)"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

export PYTHONPATH="$ROOT/third_party/RLDP:$ROOT/third_party/offline_rl/src:${PYTHONPATH:-}"

echo "PYTHONPATH configured for RLDP reproduction"

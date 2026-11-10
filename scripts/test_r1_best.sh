#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python scripts/test_r1_best.py "$@"

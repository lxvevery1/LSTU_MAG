#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CORPCOM_TOKEN="${CORPCOM_TOKEN:-corp-secret-change-me}"
exec ./corpcom_exec.sh -m corpcom.server

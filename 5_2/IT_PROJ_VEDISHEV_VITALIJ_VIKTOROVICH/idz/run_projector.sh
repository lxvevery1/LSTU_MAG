#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./corpcom_exec.sh -m corpcom.projector

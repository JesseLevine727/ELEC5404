#!/usr/bin/env bash
# Local driver for the ELEC5404 StrongARM mini-project (GPDK045) on linux.doe.carleton.ca.
#   cf.sh push                                   push local sources to the server
#   cf.sh pull <project> [run]                   pull logs/ data/ plots/ and raw/<run> (all of raw/ if no run)
#   cf.sh spectre <project> <netlist>            push, run Spectre, pull that run's results
#   cf.sh virtuoso <project> <skill> [xvfb]      push, run Virtuoso (batch or virtual display), pull
# Files only move when newer (rsync -u), so neither side's newer results get overwritten.
set -euo pipefail
HOST=jesselevine@linux.doe.carleton.ca
RROOT=Desktop/StrongArm45
LROOT="$(cd "$(dirname "$0")/.." && pwd)"
RSH="ssh -o BatchMode=yes"

push() {
    rsync -azu -e "$RSH" --exclude raw/ --exclude logs/ --exclude __pycache__/ "$LROOT/" "$HOST:$RROOT/"
}
pull() {
    rsync -azu -e "$RSH" --ignore-missing-args \
        "$HOST:$RROOT/$1/logs" "$HOST:$RROOT/$1/data" "$HOST:$RROOT/$1/plots" "$LROOT/$1/"
    if [ -n "${2:-}" ]; then
        # one run only: raw results can be GBs, the rest stays on the server
        mkdir -p "$LROOT/$1/raw"
        rsync -azu -e "$RSH" --ignore-missing-args "$HOST:$RROOT/$1/raw/$2" "$LROOT/$1/raw/"
    else
        rsync -azu -e "$RSH" --ignore-missing-args "$HOST:$RROOT/$1/raw" "$LROOT/$1/"
    fi
}

case "${1:-}" in
    push) push ;;
    pull) pull "$2" "${3:-}" ;;
    # always pull, so a partly failed run still brings back its results and log; keep the exit code
    spectre) push; rc=0; $RSH "$HOST" "tcsh -f $RROOT/common/run_spectre.csh $2 $3" || rc=$?; pull "$2" "$3"; exit $rc ;;
    virtuoso) push; rc=0; $RSH "$HOST" "tcsh -f $RROOT/common/run_virtuoso.csh $2 $3 ${4:-}" || rc=$?; pull "$2" none; exit $rc ;;
    *) sed -n '2,7p' "$0"; exit 1 ;;
esac

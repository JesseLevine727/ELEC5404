#!/bin/tcsh -f
# Runs on the server. Usage: run_spectre.csh <project dir> <netlist name without .scs>
set proj = $1
set name = $2
cd ~/Desktop/StrongArm45
source /CMC/scripts/kit.cadence.gpdk045.v6.0_OA.csh >& /dev/null
cd $proj
mkdir -p raw logs
rm -rf raw/$name
spectre -64 -format psfascii -raw raw/$name netlists/$name.scs =log logs/$name.log < /dev/null >& /dev/null
set rc = $status
echo "spectre ${name}: exit ${rc}"
egrep 'completes with|ERROR' logs/$name.log | head -5
grep -oE 'WARNING \([A-Z]+-[0-9]+\)' logs/$name.log | sort | uniq -c
exit $rc

#!/bin/tcsh -f
# Runs on the server from the GPDK045 workdir ~/Desktop/StrongArm45 (library SAL45).
# Usage: run_virtuoso.csh <project dir> <skill script name without .il> [xvfb]
set proj = $1
set name = $2
cd ~/Desktop/StrongArm45
source /CMC/scripts/kit.cadence.gpdk045.v6.0_OA.csh >& /dev/null
mkdir -p $proj/logs
set log = $proj/logs/${name}.cdslog
rm -f $log
if ( "$3" == "xvfb" ) then
    timeout 600 xvfb-run -a -s "-screen 0 4000x1000x24" virtuoso -replay $proj/skill/${name}.il -log $log < /dev/null >& /dev/null
else
    timeout 600 virtuoso -nograph -replay $proj/skill/${name}.il -log $log < /dev/null >& /dev/null
endif
echo "virtuoso ${name}: exit ${status}"
grep -E '^(\\o )?CF|\*Error\*' $log | sed 's/^\\o //'

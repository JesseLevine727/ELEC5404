; 02 - Netlist SAL45/STRONGARM_TB with ADE (OCEAN). Analyses are appended per experiment
; (scripts/), so the netlist only carries the circuit and the default design variables.
load("common/cf_lib.il")
envSetVal("asimenv.startup" "projectDir" 'string strcat(getWorkingDir() "/latch/ade"))
simulator('spectre)
design("SAL45" "STRONGARM_TB" "schematic")
modelFile('("/CMC/kits/cadence/GPDK045/gpdk045_v_6_0/models/spectre/gpdk045.scs" "tt"))
desVar("vdd" 1.0)  desVar("vcm" 0.5)  desVar("dvin" 1e-3)  desVar("cl" 5e-15)
desVar("tdel" 200e-12)  desVar("trf" 20e-12)  desVar("tpw" 1.2e-9)  desVar("tper" 2e-9)
temp(27)
printf("CF netlist %L\n" createNetlist(?recreateAll t ?display nil))
exit()

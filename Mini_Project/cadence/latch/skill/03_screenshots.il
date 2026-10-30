; 03 - Schematic screenshots (run under xvfb): cf.sh virtuoso latch 03_screenshots xvfb
load("common/cf_lib.il")
cfScreenshots("latch/plots/"
  list(list("SAL45" "STRONGARM" "schematic" "sch_strongarm.png" "win_strongarm.png")
       list("SAL45" "INV45" "schematic" "sch_inv45.png")
       list("SAL45" "STRONGARM" "symbol" "sym_strongarm.png")
       list("SAL45" "STRONGARM_TB" "schematic" "sch_strongarm_tb.png" "win_strongarm_tb.png")))

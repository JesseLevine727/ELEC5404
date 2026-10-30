; 01 - Build the GPDK045 StrongARM latch in library SAL45 (sizes from sa_sizes.il)
;   INV45         output inverter (schematic + triangle symbol)
;   STRONGARM     standard StrongARM: input pair M1/M2, clocked tail M0, NMOS latch M3/M4 on P/Q,
;                 PMOS latch M5/M6 on X/Y, PMOS reset switches S1-S4, output inverters on X/Y
;                 (OUTP = not X: rises when INP > INN)
;   STRONGARM_TB  inputs vcm +/- dvin/2, clock pulse, CL on each output
load("common/cf_lib.il")
load("latch/skill/sa_sizes.il")
cfEnsureLib("SAL45")

; ---------------- INV45 ----------------
let((cv ok)
  cv = dbOpenCellViewByType("SAL45" "INV45" "schematic" "schematic" "w")
  ok = errset({
    cfMos(cv cfPch "MP" 0:0.75 "R0" cfL cfWinvp)
    cfMos(cv cfNch "MN" 0:0 "R0" cfL cfWinvn)
    cfWires(cv list(list(0:0.75 -0.25:0.75 -0.25:0.375) list(0:0 -0.25:0 -0.25:0.375) list(-0.25:0.375 -0.625:0.375)
                    list(0.25:0.5625 0.25:0.375) list(0.25:0.1875 0.25:0.375) list(0.25:0.375 0.875:0.375)
                    list(0.25:0.9375 0.25:1.125) list(0.25:0.75 0.5:0.75 0.5:1.125 0.25:1.125)
                    list(0.25:-0.1875 0.25:-0.375) list(0.25:0 0.5:0 0.5:-0.375 0.25:-0.375)))
    cfPinAt(cv "IN" "input" -0.625:0.375)  cfPinAt(cv "OUT" "output" 0.875:0.375)
    cfPinAt(cv "VDD" "inputOutput" 0.25:1.125)  cfPinAt(cv "VSS" "inputOutput" 0.25:-0.375 t)
    cfCheckSave(cv)
  } t)
  unless(ok printf("CF INV45 FAILED\n"))
  dbPurge(cv)
  cfInvSymbol("SAL45" "INV45"))

; ---------------- STRONGARM ----------------
let((cv ok d inv)
  cv = dbOpenCellViewByType("SAL45" "STRONGARM" "schematic" "schematic" "w")
  ok = errset({
    d = list(
      cfMos(cv cfNch "M1" -1.25:0 "R0" cfL cfWin cfNFin)        ; input INP
      cfMos(cv cfNch "M2" 1.25:0 "MY" cfL cfWin cfNFin)         ; input INN
      cfMos(cv cfNch "M0" -0.25:-0.75 "R0" cfL cfWtail cfNFtail) ; clocked tail
      cfMos(cv cfNch "M3" -0.75:0.75 "MY" cfL cfWlatn cfNFlatn) ; NMOS latch, X over P
      cfMos(cv cfNch "M4" 0.75:0.75 "R0" cfL cfWlatn cfNFlatn)  ; NMOS latch, Y over Q
      cfMos(cv cfPch "M5" -0.75:1.5 "MY" cfL cfWlatp cfNFlatp)  ; PMOS latch on X
      cfMos(cv cfPch "M6" 0.75:1.5 "R0" cfL cfWlatp cfNFlatp)   ; PMOS latch on Y
      cfMos(cv cfPch "S1" -2.75:1.5 "R0" cfL cfWrst cfNFrst)     ; reset X
      cfMos(cv cfPch "S2" 2.75:1.5 "MY" cfL cfWrst cfNFrst)      ; reset Y
      cfMos(cv cfPch "S3" -2.75:0.5625 "R0" cfL cfWrst cfNFrst)  ; reset P
      cfMos(cv cfPch "S4" 2.75:0.5625 "MY" cfL cfWrst cfNFrst))  ; reset Q
    inv = dbOpenCellViewByType("SAL45" "INV45" "symbol")
    schCreateInst(cv inv "I_INVP" -4.5:1.125 "MY")
    schCreateInst(cv inv "I_INVN" 4.5:1.125 "R0")
    cfWires(cv list(
      ; X: M5/M3 drains, reset S1, output inverter, cross link to the right gates
      list(-1.0:1.3125 -1.0:0.9375) list(-2.5:1.3125 -1.0:1.3125)
      list(-1.0:1.125 -4.0:1.125) list(-1.0:1.3125 0.5:1.3125)
      ; Y: mirror
      list(1.0:1.3125 1.0:0.9375) list(2.5:1.3125 1.0:1.3125)
      list(1.0:1.125 4.0:1.125) list(1.0:1.1875 -0.5:1.1875)
      ; gate buses: left gates (M5, M3) on Y, right gates (M6, M4) on X
      list(-0.75:1.5 -0.5:1.5 -0.5:0.75 -0.75:0.75)
      list(0.75:1.5 0.5:1.5 0.5:0.75 0.75:0.75)
      ; P and Q with their reset switches
      list(-1.0:0.5625 -1.0:0.1875) list(-2.5:0.375 -1.0:0.375)
      list(1.0:0.5625 1.0:0.1875) list(2.5:0.375 1.0:0.375)
      ; tail
      list(-1.0:-0.1875 -1.0:-0.375 1.0:-0.375 1.0:-0.1875) list(0:-0.375 0:-0.5625)
      ; VDD rail
      list(-2.5:1.6875 -2.5:1.875) list(-1.0:1.6875 -1.0:1.875) list(1.0:1.6875 1.0:1.875)
      list(2.5:1.6875 2.5:1.875) list(-2.5:1.875 2.5:1.875) list(0:1.875 0:2.0625)
      ; inputs, clock, ground, outputs
      list(-1.25:0 -1.5:0) list(1.25:0 1.5:0) list(-0.25:-0.75 -0.75:-0.75) list(0:-0.9375 0:-1.125)
      list(-5.0:1.125 -5.25:1.125) list(5.0:1.125 5.25:1.125)))
    ; bodies (NMOS to VSS, PMOS to VDD), reset gates on CLK, reset sources on VDD
    ; stubs long enough to clear each device's own parameter text
    foreach(spec list(list("M1" "VSS" 0.5) list("M2" "VSS" -0.5) list("M0" "VSS" 0.5)
                      list("M3" "VSS" -0.375) list("M4" "VSS" 0.375) list("M5" "VDD" -0.375)
                      list("M6" "VDD" 0.375) list("S1" "VDD" 0.625) list("S2" "VDD" -0.625)
                      list("S3" "VDD" 0.625) list("S4" "VDD" -0.625))
      cfStub(cv dbFindAnyInstByName(cv car(spec)) "B" cadr(spec) caddr(spec) 0 "lowerCenter"))
    foreach(spec list(list("S1" -0.1875) list("S2" 0.1875) list("S3" -0.1875) list("S4" 0.1875))
      cfStub(cv dbFindAnyInstByName(cv car(spec)) "G" "CLK" cadr(spec) 0 "lowerCenter"))
    cfStub(cv dbFindAnyInstByName(cv "S3") "S" "VDD" 0 0.1875)
    cfStub(cv dbFindAnyInstByName(cv "S4") "S" "VDD" 0 0.1875)
    foreach(n '("I_INVP" "I_INVN")
      cfStub(cv dbFindAnyInstByName(cv n) "VDD" "VDD" 0 0.1875)
      cfStub(cv dbFindAnyInstByName(cv n) "VSS" "VSS" 0 -0.1875 "upperCenter"))
    cfLabelAt(cv -4.0:1.125 "X" "lowerCenter" 0.03 0.5)
    cfLabelAt(cv 4.0:1.125 "Y" "lowerCenter" 0.03 -0.5)
    cfLabelAt(cv -2.5:0.375 "P" "lowerCenter" 0.03 0.9)
    cfLabelAt(cv 2.5:0.375 "Q" "lowerCenter" 0.03 -0.9)
    cfLabelAt(cv 0:-0.5625 "TAIL" "centerLeft" 0.0 0.08)
    cfPinAt(cv "INP" "input" -1.5:0)
    cfPin(cv "INN" "input" 1.4375:0 "R180")
    cfPinAt(cv "CLK" "input" -0.75:-0.75)
    cfPin(cv "OUTP" "output" -5.1875:1.125 "R180")
    cfPinAt(cv "OUTN" "output" 5.25:1.125)
    cfPinAt(cv "VDD" "inputOutput" 0:2.0625)
    cfPinAt(cv "VSS" "inputOutput" 0:-1.125 t)
    cfCheckSave(cv)
  } t)
  unless(ok printf("CF STRONGARM FAILED\n"))
  dbPurge(cv)
  cfBoxSymbol("SAL45" "STRONGARM" list("INP" "INN" "CLK") list("OUTP" "OUTN") list("VDD") list("VSS")
              list(list("INP" "input") list("INN" "input") list("CLK" "input")
                   list("OUTP" "output") list("OUTN" "output"))))

; ---------------- STRONGARM_TB ----------------
let((cv gnd sa p q src ok)
  cv = dbOpenCellViewByType("SAL45" "STRONGARM_TB" "schematic" "schematic" "w")
  gnd = dbOpenCellViewByType("analogLib" "gnd" "symbol")
  ok = errset({
    sa = schCreateInst(cv dbOpenCellViewByType("SAL45" "STRONGARM" "symbol") "I_SA" 0:0 "R0")
    p = cfInst(cv "analogLib" "vdc" "V0" -3.0:1.0 "R0" list(list("vdc" "vdd")))
    cfLabeled(cv list(cfPinCenter(p "PLUS") -3.0:1.1875) "vdd")
    cfWires(cv list(list(cfPinCenter(p "MINUS") -3.0:0.5)))
    schCreateInst(cv gnd nil -3.0:0.5 "R0")
    cfStub(cv sa "VDD" "vdd" 0 0.1875)
    cfStub(cv sa "VSS" "gnd!" 0 -0.1875 "upperCenter")
    ; inputs vcm +/- dvin/2
    foreach(spec list(list("VINP" "INP" -2.25 "vcm+dvin/2" "inp") list("VINN" "INN" -1.5 "vcm-dvin/2" "inn"))
      q = cfPinCenter(sa nth(1 spec))
      src = cfInst(cv "analogLib" "vdc" nth(0 spec) nth(2 spec):yCoord(q) "R0" list(list("vdc" nth(3 spec))))
      cfWires(cv list(list(cfPinCenter(src "PLUS") q)
                      list(cfPinCenter(src "MINUS") nth(2 spec):(yCoord(q) - 0.5))))
      schCreateInst(cv gnd nil nth(2 spec):(yCoord(q) - 0.5) "R0")
      cfLabelAt(cv q nth(4 spec) "lowerRight" 0.03 -0.05))
    ; clock: 0 -> vdd pulse, reset while low
    q = cfPinCenter(sa "CLK")
    src = cfInst(cv "analogLib" "vpulse" "VCLK" -1.0:-0.75 "R0"
                 list(list("v1" "0") list("v2" "vdd") list("td" "tdel") list("tr" "trf") list("tf" "trf")
                      list("pw" "tpw") list("per" "tper")))
    cfWires(cv list(list(q -1.0:yCoord(q) cfPinCenter(src "PLUS"))
                    list(cfPinCenter(src "MINUS") -1.0:-1.25)))
    schCreateInst(cv gnd nil -1.0:-1.25 "R0")
    cfLabelAt(cv q "clk" "lowerRight" 0.03 -0.05)
    ; output loads: CLP flipped above OUTP, CLN below OUTN
    q = cfPinCenter(sa "OUTP")
    src = cfInst(cv "analogLib" "cap" "CLP" 1.25:yCoord(q) "R180" list(list("c" "cl")))
    cfWires(cv list(list(q cfPinCenter(src "PLUS")) list(cfPinCenter(src "MINUS") 1.25:(yCoord(q) + 0.5))))
    schCreateInst(cv gnd nil 1.25:(yCoord(q) + 0.5) "R180")
    cfLabelAt(cv cfPinCenter(src "PLUS") "outp" "lowerLeft" 0.03 0.05)
    q = cfPinCenter(sa "OUTN")
    src = cfInst(cv "analogLib" "cap" "CLN" 1.25:yCoord(q) "R0" list(list("c" "cl")))
    cfWires(cv list(list(q cfPinCenter(src "PLUS")) list(cfPinCenter(src "MINUS") 1.25:(yCoord(q) - 0.5))))
    schCreateInst(cv gnd nil 1.25:(yCoord(q) - 0.5) "R0")
    cfLabelAt(cv cfPinCenter(src "PLUS") "outn" "upperLeft" -0.03 0.05)
    cfCheckSave(cv)
  } t)
  unless(ok printf("CF STRONGARM_TB FAILED\n"))
  dbPurge(cv))
exit()

; StrongARM sizes. All L = 45n, LVT devices. Widths are totals over nf fingers.
; Chosen 2026-10-29 from the netlist explorations scripts/explore_02..07 (README).
cfNch = "nmos1v_lvt"  cfPch = "pmos1v_lvt"
cfL      = "45n"
cfWin    = "12u"   cfNFin   = "4"     ; input pair M1/M2
cfWtail  = "8u"    cfNFtail = "4"     ; clocked tail M0
cfWlatn  = "2u"    cfNFlatn = "2"     ; NMOS cross-coupled M3/M4
cfWlatp  = "2u"    cfNFlatp = "2"     ; PMOS cross-coupled M5/M6
cfWrst   = "250n"  cfNFrst  = "1"     ; PMOS reset switches S1-S4
cfWinvn  = "250n"  cfWinvp  = "700n"  ; output inverters

; cf_lib.il - SKILL helpers for the ELEC5404 StrongARM mini-project (GPDK045), ported from the ELEC3509 series.
; load("common/cf_lib.il") from ~/Desktop/StrongArm45.

; Centre of an instance terminal's pin, in cellview coordinates
procedure(cfPinCenter(inst termName)
  let((term)
    term = car(setof(tm inst~>master~>terminals tm~>name == termName))
    centerBox(dbTransformBBox(car(term~>pins)~>fig~>bBox inst~>transform))))

; Set CDF parameters the way the Edit Properties form does: PAS init,
; each value followed by its callback, then the done proc.
; params: list of (name value) pairs, applied in order.
procedure(cfSetParams(inst params)
  let((cdf)
    cdf = cdfGetInstCDF(inst)
    cdfgData = cdf
    cdfgForm = nil
    when(cdf->formInitProc evalstring(sprintf(nil "%s(cdfgData)" cdf->formInitProc)))
    foreach(p params
      cdfFindParamByName(cdf car(p))->value = cadr(p)
      when(cdfFindParamByName(cdf car(p))->callback
        evalstring(cdfFindParamByName(cdf car(p))->callback)))
    when(cdf->doneProc evalstring(sprintf(nil "%s(inst)" cdf->doneProc)))
    cdf))

; Place a gpdk045 MOS ("nmos1v"/"pmos1v") and size it, e.g. cfMos(cv "nmos1v" "M1" 0:0 "R0" "45n" "2u" "2")
procedure(cfMos(cv cell name xy orient l w @optional (fingers "1"))
  let((inst)
    inst = schCreateInst(cv dbOpenCellViewByType("gpdk045" cell "symbol") name xy orient)
    cfSetParams(inst list(list("l" l) list("fingers" fingers) list("w" w)))
    inst))

; Place an analogLib/basic instance with plain CDF values (no PAS callbacks).
; Values set through cdfGetInstCDF can carry over to the next instance of the same master,
; so AC magnitude/phase are always written explicitly ("" unless given).
procedure(cfInst(cv lib cell name xy orient @optional params)
  let((inst cdf)
    inst = schCreateInst(cv dbOpenCellViewByType(lib cell "symbol") name xy orient)
    cdf = cdfGetInstCDF(inst)
    foreach(p '("acm" "acp")
      when(cdfFindParamByName(cdf p) && !assoc(p params)
        cdfFindParamByName(cdf p)->value = ""))
    foreach(p params cdfFindParamByName(cdf car(p))->value = cadr(p))
    inst))

; Schematic pin: dir is "input" / "output" / "inputOutput"
procedure(cfPin(cv name dir xy orient)
  let((master)
    master = dbOpenCellViewByType("basic"
      cond((dir == "input" "ipin") (dir == "output" "opin") (t "iopin")) "symbol")
    schCreatePin(cv master name dir nil xy orient)))

; Draw polylines as wires. Every segment is split wherever any polyline vertex
; lies strictly inside it, so T-junctions are always endpoint-to-endpoint.
procedure(cfWires(cv polylines)
  let((verts segs onSeg cuts pts)
    verts = nil
    foreach(pl polylines foreach(p pl verts = cons(p verts)))
    segs = nil
    foreach(pl polylines
      for(i 0 length(pl) - 2
        segs = cons(list(nth(i pl) nth(i + 1 pl)) segs)))
    procedure(onSeg(p a b)
      and(abs((xCoord(b) - xCoord(a)) * (yCoord(p) - yCoord(a))
              - (yCoord(b) - yCoord(a)) * (xCoord(p) - xCoord(a))) < 1e-6
          min(xCoord(a) xCoord(b)) - 1e-6 <= xCoord(p) xCoord(p) <= max(xCoord(a) xCoord(b)) + 1e-6
          min(yCoord(a) yCoord(b)) - 1e-6 <= yCoord(p) yCoord(p) <= max(yCoord(a) yCoord(b)) + 1e-6
          !equal(p a) !equal(p b)))
    foreach(s segs
      cuts = setof(p verts onSeg(p car(s) cadr(s)))
      cuts = sort(cuts lambda((p q)
        (abs(xCoord(p) - xCoord(car(s))) + abs(yCoord(p) - yCoord(car(s))))
        < (abs(xCoord(q) - xCoord(car(s))) + abs(yCoord(q) - yCoord(car(s))))))
      pts = append(cons(car(s) cuts) list(cadr(s)))
      for(i 0 length(pts) - 2
        unless(equal(nth(i pts) nth(i + 1 pts))
          schCreateWire(cv "draw" "full" list(nth(i pts) nth(i + 1 pts)) 0.0625 0.0625 0.0))))))

; Short wire with a net label at its last point
procedure(cfLabeled(cv pts name @optional (just "lowerCenter") (dy 0.03))
  let((w end)
    w = car(schCreateWire(cv "draw" "full" pts 0.0625 0.0625 0.0))
    end = car(last(pts))
    schCreateWireLabel(cv w xCoord(end):(yCoord(end) + dy) name just "R0" "stick" 0.0625 nil)
    w))

; Check, report and save a schematic; returns t on success
procedure(cfCheckSave(cv)
  let((chk)
    chk = schCheck(cv)
    printf("CF %s/%s schCheck (errors warnings) = %L\n" cv~>libName cv~>cellName chk)
    foreach(m cv~>markers printf("CF   marker: %s\n" m~>msg))
    printf("CF %s/%s save %L\n" cv~>libName cv~>cellName dbSave(cv))
    zerop(car(chk))))

; Auto-generate a box symbol from the schematic's pins
procedure(cfMakeSymbol(lib cell)
  printf("CF %s/%s symbol %L\n" lib cell
    schViewToView(lib cell lib cell "schematic" "symbol" "schSchemToPinList" "schPinListToSymbolGen")))

; Create the library (attached to gpdk045) if needed
procedure(cfEnsureLib(lib)
  let((l)
    l = ddGetObj(lib)
    unless(l
      l = ddCreateLib(lib strcat("./" lib))
      techSetTechLibName(l "gpdk045"))
    l))

; Label the wire that has an endpoint at xy (use after cfWires)
procedure(cfLabelAt(cv xy name @optional (just "lowerCenter") (dy 0.03) (dx 0.0))
  let((w)
    w = car(setof(s cv~>shapes
                  s~>objType == "line" && s~>layerName == "wire"
                  && (equal(car(s~>points) xy) || equal(car(last(s~>points)) xy))))
    if(w then
      schCreateWireLabel(cv w (xCoord(xy) + dx):(yCoord(xy) + dy) name just "R0" "stick" 0.0625 nil)
    else
      printf("CF   WARNING no wire ends at %L for label %s\n" xy name)
      nil)))

; Box symbol with pins on chosen sides. Each side: list of pin names.
; dirs: list of (name direction) for every pin (default inputOutput).
procedure(cfBoxSymbol(lib cell left right top bottom @optional dirs)
  let((cv w h n pinAt lbl)
    n = max(length(left) length(right) 1)
    h = (n + 1) * 0.25
    w = max(1.0 (max(length(top) length(bottom)) + 1) * 0.25)
    cv = dbOpenCellViewByType(lib cell "symbol" "schematicSymbol" "w")
    procedure(pinAt(name xy stubTo labelXY just)
      let((net dir)
        dir = or(cadr(assoc(name dirs)) "inputOutput")
        net = dbCreateNet(cv name)
        dbCreateTerm(net name dir)
        dbCreatePin(net dbCreateRect(cv list("pin" "drawing")
          list((xCoord(xy) - 0.025):(yCoord(xy) - 0.025) (xCoord(xy) + 0.025):(yCoord(xy) + 0.025))) name)
        dbCreateLine(cv list("device" "drawing") list(xy stubTo))
        dbCreateLabel(cv list("pin" "drawing") labelXY name just "R0" "stick" 0.05)))
    dbCreateRect(cv list("device" "drawing") list((-w / 2):(-h / 2) (w / 2):(h / 2)))
    for(k 0 length(left) - 1
      pinAt(nth(k left) (-w / 2 - 0.125):(h / 2 - (k + 1) * 0.25) (-w / 2):(h / 2 - (k + 1) * 0.25)
            (-w / 2 + 0.04):(h / 2 - (k + 1) * 0.25) "centerLeft"))
    for(k 0 length(right) - 1
      pinAt(nth(k right) (w / 2 + 0.125):(h / 2 - (k + 1) * 0.25) (w / 2):(h / 2 - (k + 1) * 0.25)
            (w / 2 - 0.04):(h / 2 - (k + 1) * 0.25) "centerRight"))
    for(k 0 length(top) - 1
      pinAt(nth(k top) (-w / 2 + (k + 1) * w / (length(top) + 1)):(h / 2 + 0.125)
            (-w / 2 + (k + 1) * w / (length(top) + 1)):(h / 2)
            (-w / 2 + (k + 1) * w / (length(top) + 1)):(h / 2 - 0.04) "upperCenter"))
    for(k 0 length(bottom) - 1
      pinAt(nth(k bottom) (-w / 2 + (k + 1) * w / (length(bottom) + 1)):(-h / 2 - 0.125)
            (-w / 2 + (k + 1) * w / (length(bottom) + 1)):(-h / 2)
            (-w / 2 + (k + 1) * w / (length(bottom) + 1)):(-h / 2 + 0.04) "lowerCenter"))
    lbl = dbCreateLabel(cv list("annotate" "drawing7") (-w / 2):(h / 2 + 0.06) "[@instanceName]"
            "lowerLeft" "R0" "stick" 0.05)
    lbl~>labelType = "NLPLabel"
    lbl = dbCreateLabel(cv list("annotate" "drawing8") 0:0 "[@partName]" "centerCenter" "R0" "stick" 0.05)
    lbl~>labelType = "NLPLabel"
    dbCreateRect(cv list("instance" "drawing")
      list((-w / 2 - 0.125):(-h / 2 - 0.125) (w / 2 + 0.125):(h / 2 + 0.125)))
    printf("CF %s/%s box symbol terms %L save %L\n" lib cell sort(cv~>terminals~>name nil) dbSave(cv))
    dbClose(cv)))

; Write waveforms sharing one x axis to a text table at full precision
; (ocnPrint keeps only ~7 significant digits whatever ?precision says).
procedure(cfWriteWaves(file header waves)
  let((port xv)
    port = outfile(file "w")
    fprintf(port "# %s\n" header)
    xv = drGetWaveformXVec(car(waves))
    for(k 0 drVectorLength(xv) - 1
      fprintf(port "%.12e" drGetElem(xv k))
      foreach(w waves fprintf(port " %.12e" drGetElem(drGetWaveformYVec(w) k)))
      fprintf(port "\n"))
    close(port)
    file))

; Screenshots on a virtual display (run_virtuoso.csh ... xvfb). views: list of
; (lib cell view canvasPng windowPng-or-nil); files go to dir. Exits Virtuoso when done.
; The virtual screen is wider than the 1600x1000 window so the pointer sits off-canvas.
procedure(cfScreenshots(dir views)
  cfShotDir = dir
  cfShotQueue = views
  cfShotNext())

procedure(cfShotNext()
  let((v)
    if(cfShotQueue then
      v = car(cfShotQueue)
      cfShotWin = geOpen(?lib car(v) ?cell cadr(v) ?view caddr(v) ?mode "r")
      hiResizeWindow(cfShotWin list(0:0 1600:1000))
      hiZoomAbsoluteScale(cfShotWin 0.9)
      hiRegTimer("cfShotTake()" 50)
    else
      exit())))

procedure(cfShotTake()
  let((v)
    v = car(cfShotQueue)
    cfShotQueue = cdr(cfShotQueue)
    printf("CF shot %s %L\n" nth(3 v)
      or(errset(hiExportImage(?fileName strcat(cfShotDir nth(3 v)) ?window cfShotWin
                              ?exportRegion 'cellView ?width 2400 ?keepAspect t) t)
         errset(hiExportImage(?fileName strcat(cfShotDir nth(3 v)) ?window cfShotWin) t)))
    when(nth(4 v)
      system(sprintf(nil "import -window root -crop 1600x1000+0+0 +repage %s%s" cfShotDir nth(4 v))))
    hiCloseWindow(cfShotWin)
    cfShotNext()))

; Pin whose connection point is exactly xy. input -> ipin pointing left of xy,
; output -> opin to the right of xy, inputOutput -> iopin (graphic up, or down if down = t).
procedure(cfPinAt(cv name dir xy @optional down)
  cond(
    (dir == "input"
      cfPin(cv name dir (xCoord(xy) + 0.0625):yCoord(xy) "R0"))
    (dir == "output"
      cfPin(cv name dir (xCoord(xy) - 0.0625):yCoord(xy) "R0"))
    (t
      cfPin(cv name dir xy if(down "R180" "R0")))))

; Inverter symbol: triangle + bubble, IN left, OUT right, VDD top, VSS bottom
procedure(cfInvSymbol(lib cell)
  let((cv lbl pin)
    cv = dbOpenCellViewByType(lib cell "symbol" "schematicSymbol" "w")
    procedure(pin(name dir xy labelXY just)
      let((net)
        net = dbCreateNet(cv name)
        dbCreateTerm(net name dir)
        dbCreatePin(net dbCreateRect(cv list("pin" "drawing")
          list((xCoord(xy) - 0.025):(yCoord(xy) - 0.025) (xCoord(xy) + 0.025):(yCoord(xy) + 0.025))) name)
        dbCreateLabel(cv list("pin" "drawing") labelXY name just "R0" "stick" 0.045)))
    dbCreatePolygon(cv list("device" "drawing") list(-0.25:0.25 -0.25:-0.25 0.25:0))
    dbCreateEllipse(cv list("device" "drawing") list(0.25:-0.0625 0.375:0.0625))
    dbCreateLine(cv list("device" "drawing") list(-0.5:0 -0.25:0))
    dbCreateLine(cv list("device" "drawing") list(0.375:0 0.5:0))
    dbCreateLine(cv list("device" "drawing") list(0:0.125 0:0.375))
    dbCreateLine(cv list("device" "drawing") list(0:-0.125 0:-0.375))
    pin("IN" "input" -0.5:0 -0.48:0.04 "lowerLeft")
    pin("OUT" "output" 0.5:0 0.5:0.08 "lowerRight")
    pin("VDD" "inputOutput" 0:0.375 -0.04:0.33 "upperRight")
    pin("VSS" "inputOutput" 0:-0.375 -0.04:-0.33 "lowerRight")
    lbl = dbCreateLabel(cv list("annotate" "drawing7") 0.06:0.2 "[@instanceName]" "lowerLeft" "R0" "stick" 0.05)
    lbl~>labelType = "NLPLabel"
    dbCreateRect(cv list("instance" "drawing") list(-0.5:-0.375 0.5:0.375))
    printf("CF %s/%s inverter symbol terms %L save %L\n" lib cell sort(cv~>terminals~>name nil) dbSave(cv))
    dbClose(cv)))

; Short wire from an instance pin, with a net label at its end (dx, dy: stub direction)
procedure(cfStub(cv inst term name dx dy @optional (just "lowerCenter"))
  let((p)
    p = cfPinCenter(inst term)
    cfLabeled(cv list(p (xCoord(p) + dx):(yCoord(p) + dy)) name just
              if(dy < 0 -0.03 if(dy > 0 0.03 0.03)))))

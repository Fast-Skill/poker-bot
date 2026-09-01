<#
.SYNOPSIS
  Assembles the three position-specific CoinPoker template files from a
  manifest of known-good (source frame, card index, rank, suit) crops.

.DESCRIPTION
  A board card and the two hole-card positions (the fanned-under "back" card
  and the fanned-over "front" card) render their corner index differently
  enough that they need separate template sets - see coinpoker.rs's module
  docs for why. This script builds all three from one manifest, writing
  card_templates_coinpoker_board.bin, _hole_back.bin and _hole_front.bin.

  Card index is 1-based, in the same top-to-bottom/left-to-right discovery
  order `harvest-coinpoker.ps1` prints as "card N". Re-runs the exact same
  detection used there against the source frame to find that card's
  top-left corner, crops the rank and (position-appropriate) suit region at
  native resolution, converts to greyscale and contrast-stretches it exactly
  like Gray::normalised() does (best_match compares a normalised live patch
  against the template as loaded, so a raw, un-stretched template is being
  compared on a different scale every time - this cost a long debugging
  session to track down once already), and packs everything into the PKVT
  format `poker_vision::Templates` expects.

  A position is only written once all 13 ranks and all 4 suits are present
  for it - a partial rank set is worse than no file at all, because an
  untemplated rank does not refuse to match, it confidently matches the
  nearest wrong template.
#>
param(
  [string]$OutDir = "D:\poker-bot\data"
)

Add-Type -AssemblyName System.Drawing

# --- the manifest: edit this as new cards get harvested -----------------------
# Suit=$null means "already have this suit for this position, rank is what's
# new here".
$Manifest = @(
  # --- board (complete: 13/13 ranks, 4/4 suits) --------------------------------
  @{ Frame="h181631-0-1280x960.png"; Card=1; Rank="A"; Suit="d"; Position="Board" }
  @{ Frame="h181631-0-1280x960.png"; Card=2; Rank="2"; Suit="s"; Position="Board" }
  @{ Frame="h181631-0-1280x960.png"; Card=3; Rank="9"; Suit=$null; Position="Board" }
  @{ Frame="h181631-0-1280x960.png"; Card=4; Rank="J"; Suit="h"; Position="Board" }
  @{ Frame="h181631-0-1280x960.png"; Card=5; Rank="K"; Suit="c"; Position="Board" }
  @{ Frame="h191248-0-1280x960.png"; Card=4; Rank="3"; Suit=$null; Position="Board" }
  @{ Frame="h205555-0-1280x960.png"; Card=6; Rank="4"; Suit=$null; Position="Board" }
  @{ Frame="h191248-0-1280x960.png"; Card=3; Rank="5"; Suit=$null; Position="Board" }
  @{ Frame="h115119-0-1280x960.png"; Card=5; Rank="6"; Suit=$null; Position="Board" }
  @{ Frame="h172738-0-1280x960.png"; Card=2; Rank="7"; Suit=$null; Position="Board" }
  @{ Frame="h172738-0-1280x960.png"; Card=5; Rank="8"; Suit=$null; Position="Board" }
  @{ Frame="h191248-0-1280x960.png"; Card=5; Rank="T"; Suit=$null; Position="Board" }
  @{ Frame="h172408-0-1280x960.png"; Card=1; Rank="Q"; Suit=$null; Position="Board" }

  # --- hole-front (complete: 13/13 ranks, 4/4 suits) ---------------------------
  @{ Frame="h120454-0-1280x960.png"; Card=2; Rank="A"; Suit=$null; Position="HoleFront" }
  @{ Frame="h230407-0-1280x960.png"; Card=2; Rank="2"; Suit=$null; Position="HoleFront" }
  @{ Frame="h205314-0-1280x960.png"; Card=6; Rank="3"; Suit="d";   Position="HoleFront" }
  @{ Frame="h214351-0-1280x960.png"; Card=2; Rank="4"; Suit="s";   Position="HoleFront" }
  @{ Frame="h213915-0-1280x960.png"; Card=2; Rank="5"; Suit=$null; Position="HoleFront" }
  @{ Frame="h181631-0-1280x960.png"; Card=7; Rank="6"; Suit="d";   Position="HoleFront" }
  @{ Frame="h172408-0-1280x960.png"; Card=5; Rank="7"; Suit=$null; Position="HoleFront" }
  @{ Frame="h214128-0-1280x960.png"; Card=2; Rank="8"; Suit=$null; Position="HoleFront" }
  @{ Frame="h210207-0-1280x960.png"; Card=5; Rank="9"; Suit="c";   Position="HoleFront" }
  @{ Frame="h121826-0-1280x960.png"; Card=5; Rank="T"; Suit="h";   Position="HoleFront" }
  @{ Frame="h215445-0-1280x960.png"; Card=3; Rank="J"; Suit="d";   Position="HoleFront" }
  @{ Frame="h205510-0-1280x960.png"; Card=2; Rank="Q"; Suit="c";   Position="HoleFront" }
  @{ Frame="h215247-0-1280x960.png"; Card=2; Rank="K"; Suit=$null; Position="HoleFront" }

  # --- hole-back (incomplete: 11/13 - still missing 2 and 4) -------------------
  @{ Frame="h181506-0-1280x960.png"; Card=1; Rank="A"; Suit="d";   Position="HoleBack" }
  @{ Frame="h170646-0-1280x960.png"; Card=1; Rank="3"; Suit=$null; Position="HoleBack" }
  @{ Frame="h214351-0-1280x960.png"; Card=1; Rank="5"; Suit=$null; Position="HoleBack" }
  @{ Frame="h181631-0-1280x960.png"; Card=6; Rank="6"; Suit="h";   Position="HoleBack" }
  @{ Frame="h214905-0-1280x960.png"; Card=3; Rank="7"; Suit="h";   Position="HoleBack" }
  @{ Frame="h210323-0-1280x960.png"; Card=5; Rank="8"; Suit="s";   Position="HoleBack" }
  @{ Frame="h214128-0-1280x960.png"; Card=1; Rank="9"; Suit=$null; Position="HoleBack" }
  @{ Frame="h172408-0-1280x960.png"; Card=4; Rank="T"; Suit="c";   Position="HoleBack" }
  @{ Frame="h171130-0-1280x960.png"; Card=4; Rank="J"; Suit=$null; Position="HoleBack" }
  @{ Frame="h115742-0-1280x960.png"; Card=1; Rank="Q"; Suit=$null; Position="HoleBack" }
  @{ Frame="h205510-0-1280x960.png"; Card=1; Rank="K"; Suit=$null; Position="HoleBack" }
  @{ Frame="h205749-0-1280x960.png"; Card=4; Rank="K"; Suit="d";   Position="HoleBack" } # suit-only: diamonds
  # 2 and 4 still needed - add lines here once harvested, e.g.:
  # @{ Frame="hXXXXXX-0-1280x960.png"; Card=N; Rank="2"; Suit=$null; Position="HoleBack" }
  # @{ Frame="hXXXXXX-0-1280x960.png"; Card=N; Rank="4"; Suit=$null; Position="HoleBack" }
)

$SrcDir = "D:\poker-bot\captures-coinpoker"
$RANK_TOP=8; $RANK_W=38; $RANK_H=31
$SUIT_W=34; $SUIT_H=27
# Matches coinpoker.rs's Position::suit_offset() exactly - (dx, top).
$SuitOffset = @{
  Board     = @(0, 43)
  HoleBack  = @(7, 49)
  HoleFront = @(-5, 44)
}

# --- detect card boxes, tagging each with its Position ------------------------
function Get-CardBoxes($bmp) {
    $w = $bmp.Width; $h = $bmp.Height
    $mask = New-Object 'bool[,]' $w,$h
    for ($y = 0; $y -lt $h; $y++) {
        for ($x = 0; $x -lt $w; $x++) {
            $p = $bmp.GetPixel($x, $y)
            $lo = [Math]::Min($p.R, [Math]::Min($p.G, $p.B))
            $hi = [Math]::Max($p.R, [Math]::Max($p.G, $p.B))
            $mask[$x,$y] = ($lo -gt 110) -and (($hi - $lo) -lt 60)
        }
    }
    $seen = New-Object 'bool[,]' $w,$h
    $boxes = New-Object System.Collections.Generic.List[object]
    for ($sy = 0; $sy -lt $h; $sy++) {
        for ($sx = 0; $sx -lt $w; $sx++) {
            if (-not $mask[$sx,$sy] -or $seen[$sx,$sy]) { continue }
            $stack = New-Object System.Collections.Generic.Stack[object]
            $stack.Push(@($sx,$sy)); $seen[$sx,$sy] = $true
            $x0=$sx; $x1=$sx; $y0=$sy; $y1=$sy
            while ($stack.Count -gt 0) {
                $cur = $stack.Pop(); $cx = $cur[0]; $cy = $cur[1]
                if ($cx -lt $x0) { $x0 = $cx }; if ($cx -gt $x1) { $x1 = $cx }
                if ($cy -lt $y0) { $y0 = $cy }; if ($cy -gt $y1) { $y1 = $cy }
                foreach ($d in @(@(1,0),@(-1,0),@(0,1),@(0,-1))) {
                    $nx = $cx + $d[0]; $ny = $cy + $d[1]
                    if ($nx -ge 0 -and $nx -lt $w -and $ny -ge 0 -and $ny -lt $h -and $mask[$nx,$ny] -and -not $seen[$nx,$ny]) {
                        $seen[$nx,$ny] = $true; $stack.Push(@($nx,$ny))
                    }
                }
            }
            $bw = $x1-$x0+1; $bh = $y1-$y0+1
            if ($bh -ge 90 -and $bh -le 200 -and $bw -ge 60) {
                $boxes.Add([PSCustomObject]@{X=$x0; Y=$y0; W=$bw; H=$bh})
            }
        }
    }
    $CARD_W = 86
    $cards = New-Object System.Collections.Generic.List[object]
    foreach ($b in $boxes) {
        if ($b.W -le ($CARD_W + 15)) {
            $cards.Add([PSCustomObject]@{X=$b.X; Y=$b.Y; Position="Board"})
        } else {
            $cards.Add([PSCustomObject]@{X=$b.X; Y=$b.Y; Position="HoleBack"})
            $cards.Add([PSCustomObject]@{X=($b.X + $b.W - $CARD_W); Y=$b.Y; Position="HoleFront"})
        }
    }
    return $cards
}

# --- greyscale + contrast-stretch a region, matching Gray::normalised() -------
function Get-GreyBytes($bmp, $x, $y, $w, $h) {
    $raw = New-Object int[] ($w * $h)
    $i = 0
    for ($yy = 0; $yy -lt $h; $yy++) {
        for ($xx = 0; $xx -lt $w; $xx++) {
            $p = $bmp.GetPixel($x + $xx, $y + $yy)
            $luma = 0.299 * $p.R + 0.587 * $p.G + 0.114 * $p.B
            $v = [Math]::Round($luma)
            if ($v -lt 0) { $v = 0 }; if ($v -gt 255) { $v = 255 }
            $raw[$i] = [int]$v
            $i++
        }
    }
    $lo = ($raw | Measure-Object -Minimum).Minimum
    $hi = ($raw | Measure-Object -Maximum).Maximum
    $bytes = New-Object byte[] ($w * $h)
    if (($hi - $lo) -lt 30) {
        for ($j = 0; $j -lt $bytes.Length; $j++) { $bytes[$j] = 255 }
        return ,$bytes
    }
    $scale = 255.0 / ($hi - $lo)
    for ($j = 0; $j -lt $raw.Length; $j++) {
        # Rust's `as u8` truncates toward zero, it does not round.
        $stretched = [Math]::Truncate(($raw[$j] - $lo) * $scale)
        if ($stretched -lt 0) { $stretched = 0 }; if ($stretched -gt 255) { $stretched = 255 }
        $bytes[$j] = [byte]$stretched
    }
    return ,$bytes
}

# --- resolve the manifest into position -> rank/suit -> bytes -----------------
$positions = @("Board", "HoleBack", "HoleFront")
$rankBytes = @{}; $suitBytes = @{}
foreach ($pos in $positions) { $rankBytes[$pos] = @{}; $suitBytes[$pos] = @{} }
$bmpCache = @{}

foreach ($entry in $Manifest) {
    $path = Join-Path $SrcDir $entry.Frame
    if (-not (Test-Path $path)) { Write-Host "missing source frame: $path" -ForegroundColor Red; continue }
    if (-not $bmpCache.ContainsKey($path)) { $bmpCache[$path] = New-Object System.Drawing.Bitmap($path) }
    $bmp = $bmpCache[$path]
    # Card indices in the manifest are positions in the *full* detected list,
    # matching harvest-coinpoker.ps1's "card N" numbering - not positions
    # within just this entry's Board/HoleBack/HoleFront subset.
    $allCards = Get-CardBoxes $bmp
    if ($entry.Card -gt $allCards.Count) {
        Write-Host "frame $($entry.Frame) only has $($allCards.Count) card(s), asked for #$($entry.Card)" -ForegroundColor Red
        continue
    }
    $c = $allCards[$entry.Card - 1]
    if ($c.Position -ne $entry.Position) {
        Write-Host "frame $($entry.Frame) card $($entry.Card) is $($c.Position), not $($entry.Position) as the manifest says" -ForegroundColor Red
        continue
    }

    if ($entry.Rank -and -not $rankBytes[$entry.Position].ContainsKey($entry.Rank)) {
        $rankBytes[$entry.Position][$entry.Rank] = Get-GreyBytes $bmp ($c.X) ($c.Y + $RANK_TOP) $RANK_W $RANK_H
    }
    if ($entry.Suit -and -not $suitBytes[$entry.Position].ContainsKey($entry.Suit)) {
        $offset = $SuitOffset[$entry.Position]
        $sx = [Math]::Max(0, $c.X + $offset[0])
        $suitBytes[$entry.Position][$entry.Suit] = Get-GreyBytes $bmp $sx ($c.Y + $offset[1]) $SUIT_W $SUIT_H
    }
}

# --- write whichever positions are actually complete ---------------------------
$wantedRanks = @("2","3","4","5","6","7","8","9","T","J","Q","K","A")
$wantedSuits = @("c","d","h","s")
$fileNames = @{ Board="card_templates_coinpoker_board.bin"; HoleBack="card_templates_coinpoker_hole_back.bin"; HoleFront="card_templates_coinpoker_hole_front.bin" }

foreach ($pos in $positions) {
    $missingRanks = $wantedRanks | Where-Object { -not $rankBytes[$pos].ContainsKey($_) }
    $missingSuits = $wantedSuits | Where-Object { -not $suitBytes[$pos].ContainsKey($_) }
    Write-Host "$pos - ranks: $($rankBytes[$pos].Keys.Count)/13, suits: $($suitBytes[$pos].Keys.Count)/4"
    if ($missingRanks.Count -gt 0) { Write-Host "  missing ranks: $($missingRanks -join ', ')" -ForegroundColor Yellow }
    if ($missingSuits.Count -gt 0) { Write-Host "  missing suits: $($missingSuits -join ', ')" -ForegroundColor Yellow }
    if ($missingRanks.Count -gt 0 -or $missingSuits.Count -gt 0) {
        Write-Host "  not writing $($fileNames[$pos]) - incomplete." -ForegroundColor Yellow
        continue
    }

    $outFile = Join-Path $OutDir $fileNames[$pos]
    $fs = [System.IO.File]::Create($outFile)
    $bw = New-Object System.IO.BinaryWriter($fs)
    $bw.Write([byte[]][char[]]"PKVT")
    $bw.Write([uint32]1)

    $bw.Write([uint32]$wantedRanks.Count)
    $bw.Write([uint32]$RANK_H)
    $bw.Write([uint32]$RANK_W)
    foreach ($r in $wantedRanks) {
        $label = [byte[]][char[]]$r
        $bw.Write([byte]$label.Length); $bw.Write($label)
        $bw.Write($rankBytes[$pos][$r])
    }

    $bw.Write([uint32]$wantedSuits.Count)
    $bw.Write([uint32]$SUIT_H)
    $bw.Write([uint32]$SUIT_W)
    foreach ($s in $wantedSuits) {
        $label = [byte[]][char[]]$s
        $bw.Write([byte]$label.Length); $bw.Write($label)
        $bw.Write($suitBytes[$pos][$s])
    }

    $bw.Close(); $fs.Close()
    Write-Host "  wrote $outFile" -ForegroundColor Green
}

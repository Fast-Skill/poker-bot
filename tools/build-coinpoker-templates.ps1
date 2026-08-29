<#
.SYNOPSIS
  Assembles data/card_templates_coinpoker.bin from a manifest of known-good
  (source frame, card index, rank, suit) crops.

.DESCRIPTION
  Card index is 1-based, in the same top-to-bottom/left-to-right discovery
  order `harvest-coinpoker.ps1` prints as "card N". Re-runs the exact same
  detection used there against the source frame to find that card's top-left
  corner, then crops the rank and suit regions at native resolution (no
  zoom), converts to greyscale with the same luma weights the Rust reader
  uses, and packs everything into the PKVT format `poker_vision::Templates`
  expects.

  Refuses to write the file unless all 13 ranks and all 4 suits are present -
  a partial rank set is worse than no file at all, because an untemplated
  rank does not refuse to match, it confidently matches the nearest wrong
  template. See coinpoker.rs's module docs.
#>
param(
  [string]$OutFile = "D:\poker-bot\data\card_templates_coinpoker.bin"
)

Add-Type -AssemblyName System.Drawing

# --- the manifest: edit this as new cards get harvested -----------------------
$Manifest = @(
  @{ Frame="h181631-0-1280x960.png"; Card=1; Rank="A"; Suit="d" }  # A of diamonds
  @{ Frame="h181631-0-1280x960.png"; Card=2; Rank="2"; Suit="s" }  # 2 of spades
  @{ Frame="h181631-0-1280x960.png"; Card=3; Rank="9"; Suit="s" }  # 9 of spades
  @{ Frame="h181631-0-1280x960.png"; Card=4; Rank="J"; Suit="h" }  # J of hearts
  @{ Frame="h181631-0-1280x960.png"; Card=5; Rank="K"; Suit="c" }  # K of clubs
  @{ Frame="h172408-0-1280x960.png"; Card=1; Rank="Q"; Suit=$null } # Q of diamonds (suit already have)
  @{ Frame="h172408-0-1280x960.png"; Card=4; Rank="T"; Suit=$null } # 10 of clubs
  @{ Frame="h172408-0-1280x960.png"; Card=5; Rank="7"; Suit=$null } # 7 of hearts
  @{ Frame="h172738-0-1280x960.png"; Card=5; Rank="8"; Suit=$null } # 8 of hearts
  @{ Frame="h172738-0-1280x960.png"; Card=7; Rank="5"; Suit=$null } # 5 of clubs
  @{ Frame="h181616-0-1280x960.png"; Card=4; Rank="6"; Suit=$null } # 6 (red)
  @{ Frame="flop1b-0-1280x960.png";  Card=5; Rank="4"; Suit=$null } # 4 of diamonds
  # 3 is still missing - add a line here once harvested, e.g.:
  # @{ Frame="hXXXXXX-0-1280x960.png"; Card=N; Rank="3"; Suit=$null }
)

$SrcDir = "D:\poker-bot\captures-coinpoker"
$RANK_TOP=8; $RANK_W=38; $RANK_H=31
$SUIT_TOP=38; $SUIT_W=34; $SUIT_H=36

# --- detect card boxes in a frame (same rule as coinpoker.rs) -----------------
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
            $cards.Add([PSCustomObject]@{X=$b.X; Y=$b.Y})
        } else {
            $cards.Add([PSCustomObject]@{X=$b.X; Y=$b.Y})
            $cards.Add([PSCustomObject]@{X=($b.X + $b.W - $CARD_W); Y=$b.Y})
        }
    }
    return $cards
}

# --- greyscale a region with the same luma weights the Rust reader uses -------
function Get-GreyBytes($bmp, $x, $y, $w, $h) {
    $bytes = New-Object byte[] ($w * $h)
    $i = 0
    for ($yy = 0; $yy -lt $h; $yy++) {
        for ($xx = 0; $xx -lt $w; $xx++) {
            $p = $bmp.GetPixel($x + $xx, $y + $yy)
            $luma = 0.299 * $p.R + 0.587 * $p.G + 0.114 * $p.B
            $v = [Math]::Round($luma)
            if ($v -lt 0) { $v = 0 }; if ($v -gt 255) { $v = 255 }
            $bytes[$i] = [byte]$v
            $i++
        }
    }
    return ,$bytes
}

# --- resolve the manifest into rank/suit -> bytes -----------------------------
$rankBytes = @{}
$suitBytes = @{}
$bmpCache = @{}
foreach ($entry in $Manifest) {
    $path = Join-Path $SrcDir $entry.Frame
    if (-not (Test-Path $path)) { Write-Host "missing source frame: $path" -ForegroundColor Red; continue }
    if (-not $bmpCache.ContainsKey($path)) { $bmpCache[$path] = New-Object System.Drawing.Bitmap($path) }
    $bmp = $bmpCache[$path]
    $cards = Get-CardBoxes $bmp
    if ($entry.Card -gt $cards.Count) {
        Write-Host "frame $($entry.Frame) only has $($cards.Count) card(s), asked for #$($entry.Card)" -ForegroundColor Red
        continue
    }
    $c = $cards[$entry.Card - 1]

    if ($entry.Rank -and -not $rankBytes.ContainsKey($entry.Rank)) {
        $rankBytes[$entry.Rank] = Get-GreyBytes $bmp ($c.X) ($c.Y + $RANK_TOP) $RANK_W $RANK_H
    }
    if ($entry.Suit -and -not $suitBytes.ContainsKey($entry.Suit)) {
        $suitBytes[$entry.Suit] = Get-GreyBytes $bmp ($c.X) ($c.Y + $SUIT_TOP) $SUIT_W $SUIT_H
    }
}

$wantedRanks = @("2","3","4","5","6","7","8","9","T","J","Q","K","A")
$wantedSuits = @("c","d","h","s")
$missingRanks = $wantedRanks | Where-Object { -not $rankBytes.ContainsKey($_) }
$missingSuits = $wantedSuits | Where-Object { -not $suitBytes.ContainsKey($_) }

Write-Host "ranks collected: $($rankBytes.Keys.Count) / 13"
Write-Host "suits collected: $($suitBytes.Keys.Count) / 4"
if ($missingRanks.Count -gt 0) { Write-Host "missing ranks: $($missingRanks -join ', ')" -ForegroundColor Yellow }
if ($missingSuits.Count -gt 0) { Write-Host "missing suits: $($missingSuits -join ', ')" -ForegroundColor Yellow }

if ($missingRanks.Count -gt 0 -or $missingSuits.Count -gt 0) {
    Write-Host "`nNot writing $OutFile - the loader refuses a file that isn't all 13 ranks and all 4 suits." -ForegroundColor Yellow
    exit 1
}

# --- write the PKVT binary -----------------------------------------------------
$fs = [System.IO.File]::Create($OutFile)
$bw = New-Object System.IO.BinaryWriter($fs)
$bw.Write([byte[]][char[]]"PKVT")
$bw.Write([uint32]1)  # version

$bw.Write([uint32]$wantedRanks.Count)
$bw.Write([uint32]$RANK_H)
$bw.Write([uint32]$RANK_W)
foreach ($r in $wantedRanks) {
    $label = [byte[]][char[]]$r
    $bw.Write([byte]$label.Length)
    $bw.Write($label)
    $bw.Write($rankBytes[$r])
}

$bw.Write([uint32]$wantedSuits.Count)
$bw.Write([uint32]$SUIT_H)
$bw.Write([uint32]$SUIT_W)
foreach ($s in $wantedSuits) {
    $label = [byte[]][char[]]$s
    $bw.Write([byte]$label.Length)
    $bw.Write($label)
    $bw.Write($suitBytes[$s])
}

$bw.Close(); $fs.Close()
Write-Host "`nwrote $OutFile" -ForegroundColor Green

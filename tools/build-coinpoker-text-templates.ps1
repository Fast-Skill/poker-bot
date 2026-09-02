<#
.SYNOPSIS
  Assembles data/digit_templates_coinpoker.bin from a manifest of known-good
  (source frame, x, y, w, h, ink, label) glyph crops.

.DESCRIPTION
  Unlike the card templates, these are simple binary masks (0 or 255 per
  pixel) rather than contrast-stretched greyscale - that's what
  coinpoker_text.rs's Ink::matches() produces, and what its matcher compares
  against. Each (frame, x, y, w, h) is read straight from a
  harvest-coinpoker-text.ps1 crop's filename, which already carries a
  glyph's exact native-resolution bounding box - no re-detection needed.

  Known gap: white 5 and 9 are only sourced from the Bet/Raise button
  caption (~20px tall), not from the pot or bet-chip (~19px tall). A pot or
  bet-chip reading containing a 5 or 9 will therefore refuse rather than
  read - safe, but incomplete. Backfill by finding a pot- or bet-chip-sized
  5 or 9 and pointing its manifest entry there instead.
#>
param(
  [string]$SrcDir = "D:\poker-bot\captures-coinpoker",
  [string]$OutFile = "D:\poker-bot\data\digit_templates_coinpoker.bin"
)

Add-Type -AssemblyName System.Drawing

function IsGold($r, $g, $b) { return ($r -gt 200 -and $g -gt 150 -and $b -lt 100 -and ($r-$b) -gt 100) }
function IsWhite($r, $g, $b) { return ($r -gt 200 -and $g -gt 200 -and $b -gt 200 -and [Math]::Abs($r-$b) -lt 20 -and [Math]::Abs($g-$b) -lt 20) }

# --- the manifest: edit this as new glyphs get harvested ----------------------
$Manifest = @(
  # white, ~19px - pot and bet-chip readouts
  @{ Frame="t211644-0-1280x960.png"; X=641; Y=347; W=12; H=19; Ink="white"; Label="0" }
  @{ Frame="t002432-0-1280x960.png"; X=663; Y=347; W=5;  H=19; Ink="white"; Label="1" }
  @{ Frame="t211739-0-1280x960.png"; X=661; Y=347; W=10; H=19; Ink="white"; Label="2" }
  @{ Frame="t194058-0-1280x960.png"; X=675; Y=347; W=11; H=19; Ink="white"; Label="3" }
  @{ Frame="t194020-0-1280x960.png"; X=674; Y=347; W=13; H=19; Ink="white"; Label="4" }
  @{ Frame="t094307-0-1280x960.png"; X=675; Y=347; W=11; H=19; Ink="white"; Label="6" }
  @{ Frame="t193436-0-1280x960.png"; X=675; Y=347; W=11; H=19; Ink="white"; Label="7" }
  @{ Frame="t194143-0-1280x960.png"; X=675; Y=347; W=11; H=19; Ink="white"; Label="8" }
  @{ Frame="h181631-0-1280x960.png"; X=656; Y=362; W=3;  H=4;  Ink="white"; Label="." }
  # white, ~20px - Bet/Raise button caption only (see the known-gap note above)
  @{ Frame="t194257-0-1280x960.png"; X=1204; Y=926; W=10; H=20; Ink="white"; Label="5" }
  @{ Frame="t223454-0-1280x960.png"; X=1195; Y=925; W=11; H=20; Ink="white"; Label="9" }
  # gold, ~22-23px - hero/villain stacks
  @{ Frame="t225658-0-1280x960.png"; X=614; Y=222; W=14; H=23; Ink="gold"; Label="0" }
  @{ Frame="t211726-0-1280x960.png"; X=615; Y=222; W=8;  H=22; Ink="gold"; Label="1" }
  @{ Frame="t094457-0-1280x960.png"; X=616; Y=222; W=12; H=22; Ink="gold"; Label="2" }
  @{ Frame="t002432-0-1280x960.png"; X=654; Y=860; W=13; H=23; Ink="gold"; Label="3" }
  @{ Frame="t172646-0-1280x960.png"; X=655; Y=222; W=12; H=23; Ink="gold"; Label="4" }
  @{ Frame="t172537-0-1280x960.png"; X=650; Y=222; W=15; H=22; Ink="gold"; Label="5" }
  @{ Frame="t172706-0-1280x960.png"; X=651; Y=222; W=13; H=23; Ink="gold"; Label="6" }
  @{ Frame="t094307-0-1280x960.png"; X=651; Y=861; W=13; H=22; Ink="gold"; Label="7" }
  @{ Frame="t172604-0-1280x960.png"; X=651; Y=860; W=14; H=23; Ink="gold"; Label="8" }
  @{ Frame="t223454-0-1280x960.png"; X=636; Y=861; W=13; H=22; Ink="gold"; Label="9" }
  @{ Frame="h181631-0-1280x960.png"; X=628; Y=879; W=4;  H=4;  Ink="gold"; Label="." }
)

$bmpCache = @{}
$glyphs = New-Object System.Collections.Generic.List[object]
$missing = $false
foreach ($entry in $Manifest) {
    $path = Join-Path $SrcDir $entry.Frame
    if (-not (Test-Path $path)) {
        Write-Host "missing source frame: $path" -ForegroundColor Red
        $missing = $true
        continue
    }
    if (-not $bmpCache.ContainsKey($path)) { $bmpCache[$path] = New-Object System.Drawing.Bitmap($path) }
    $bmp = $bmpCache[$path]

    $w = $entry.W; $h = $entry.H
    $mask = New-Object byte[] ($w * $h)
    $i = 0
    for ($y = 0; $y -lt $h; $y++) {
        for ($x = 0; $x -lt $w; $x++) {
            $p = $bmp.GetPixel($entry.X + $x, $entry.Y + $y)
            $hit = if ($entry.Ink -eq "gold") { IsGold $p.R $p.G $p.B } else { IsWhite $p.R $p.G $p.B }
            $mask[$i] = if ($hit) { 255 } else { 0 }
            $i++
        }
    }
    $glyphs.Add([PSCustomObject]@{ Ink=$entry.Ink; Label=$entry.Label; W=$w; H=$h; Mask=$mask })
}

if ($missing) {
    Write-Host "`nNot writing $OutFile - fix the missing source frame(s) above first." -ForegroundColor Yellow
    exit 1
}

$wantedDigits = @("0","1","2","3","4","5","6","7","8","9",".")
foreach ($ink in @("gold","white")) {
    $have = $glyphs | Where-Object { $_.Ink -eq $ink } | ForEach-Object { $_.Label }
    $missingLabels = $wantedDigits | Where-Object { $_ -notin $have }
    Write-Host "$ink : $($have.Count) / $($wantedDigits.Count) glyph(s)"
    if ($missingLabels.Count -gt 0) {
        Write-Host "  missing: $($missingLabels -join ', ')" -ForegroundColor Yellow
        $missing = $true
    }
}
if ($missing) {
    Write-Host "`nNot writing $OutFile - incomplete." -ForegroundColor Yellow
    exit 1
}

# --- write the PKCT binary -----------------------------------------------------
$fs = [System.IO.File]::Create($OutFile)
$bw = New-Object System.IO.BinaryWriter($fs)
$bw.Write([byte[]][char[]]"PKCT")
$bw.Write([uint32]1)  # version
$bw.Write([uint32]$glyphs.Count)
foreach ($g in $glyphs) {
    $inkLabel = [byte[]][char[]]$g.Ink
    $bw.Write([byte]$inkLabel.Length); $bw.Write($inkLabel)
    $charLabel = [byte[]][char[]]$g.Label
    $bw.Write([byte]$charLabel.Length); $bw.Write($charLabel)
    $bw.Write([uint32]$g.H)
    $bw.Write([uint32]$g.W)
    $bw.Write($g.Mask)
}
$bw.Close(); $fs.Close()
Write-Host "`nwrote $OutFile ($($glyphs.Count) glyphs)" -ForegroundColor Green

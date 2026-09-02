<#
.SYNOPSIS
  Captures the CoinPoker table and crops every gold or white glyph-sized
  blob, for building the digit template set.

.DESCRIPTION
  Same idea as harvest-coinpoker.ps1, for numeric text instead of cards:
  resize the table to the canonical 1280x960, capture it, then scan a
  handful of known regions (pot, hero/villain stacks, the bet-chip amount,
  the bet/raise button caption) for connected blobs matching the two
  measured inks (gold = stacks, white = everything else) that are about the
  size of one character. Each candidate gets cropped and zoomed into its own
  small PNG for a human to identify and label.

  Scanning known regions rather than the whole frame is deliberate - a first
  pass scanned everything and mostly caught icons, the "Max" button label,
  and "%" symbols on the sizing presets, none of them digits. Restricting to
  where amounts actually render cuts that noise out before it ever reaches
  a human to review.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\harvest-coinpoker-text.ps1
#>
param(
  [string]$RepoDir = "D:\poker-bot",
  [string]$OutDir = "D:\poker-bot\captures-coinpoker\harvest-text"
)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public class HarvestTextWin {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    public static List<IntPtr> Handles = new List<IntPtr>();
    public static void Scan() {
        Handles.Clear();
        EnumWindows((hWnd, lParam) => {
            if (!IsWindowVisible(hWnd)) return true;
            var sb = new StringBuilder(256);
            GetWindowText(hWnd, sb, 256);
            if (sb.ToString() != "CoinPoker") return true;
            RECT r; GetWindowRect(hWnd, out r);
            int w = r.Right-r.Left, h = r.Bottom-r.Top;
            if (w == 1280 && h == 778) return true;
            if (w < 100 || h < 100) return true;
            Handles.Add(hWnd);
            return true;
        }, IntPtr.Zero);
    }
}
"@

# --- 1. resize the table window (excluding the lobby) ------------------------
[HarvestTextWin]::Scan()
if ([HarvestTextWin]::Handles.Count -eq 0) {
    Write-Host "No CoinPoker table window found. Open a table with a hand showing and retry." -ForegroundColor Yellow
    exit 1
}
foreach ($h in [HarvestTextWin]::Handles) {
    [HarvestTextWin]::SetWindowPos($h, [IntPtr]::Zero, 100, 50, 1280, 960, 0x0004) | Out-Null
}
Start-Sleep -Milliseconds 250

# --- 2. capture via the project's own tool ------------------------------------
Push-Location $RepoDir
$stamp = Get-Date -Format "HHmmss"
& ".\target\debug\poker.exe" grab --process CoinPoker --out captures-coinpoker --label "t$stamp" | Out-Null
Pop-Location
$frame = Get-ChildItem (Join-Path $RepoDir "captures-coinpoker\t$stamp-0-1280x960.png") -ErrorAction SilentlyContinue
if (-not $frame) {
    Write-Host "No 1280x960 capture came back - is a table actually open and showing a hand?" -ForegroundColor Yellow
    exit 1
}
$src = $frame.FullName
Write-Host "captured: $src"

# --- 3. scan known amount regions for gold/white glyph-sized blobs ------------
$bmp = New-Object System.Drawing.Bitmap($src)
$w = $bmp.Width; $h = $bmp.Height

function IsGold($r, $g, $b) { return ($r -gt 200 -and $g -gt 150 -and $b -lt 100 -and ($r-$b) -gt 100) }
function IsWhite($r, $g, $b) { return ($r -gt 200 -and $g -gt 200 -and $b -gt 200 -and [Math]::Abs($r-$b) -lt 20 -and [Math]::Abs($g-$b) -lt 20) }

# Generous regions, measured from captures-coinpoker/h181631-0-1280x960.png,
# where an amount actually renders - not the whole frame. (x0,y0,x1,y1)
$Regions = @(
  @{Name="pot";           Ink="white"; Box=@(560,335,740,375)}
  @{Name="hero-stack";    Ink="gold";  Box=@(560,845,730,895)}
  @{Name="villain-stack"; Ink="gold";  Box=@(555,205,730,250)}
  @{Name="bet-chip";      Ink="white"; Box=@(560,535,700,580)}
  @{Name="aggressive-caption"; Ink="white"; Box=@(1105,905,1275,955)}
  @{Name="passive-caption";    Ink="white"; Box=@(930,905,1100,955)}
)

function FindBlobsInRegion($bmp, $box, $inkName) {
    $rx0=$box[0]; $ry0=$box[1]; $rx1=$box[2]; $ry1=$box[3]
    $rw = $rx1-$rx0; $rh = $ry1-$ry0
    $mask = New-Object 'bool[,]' $rw,$rh
    for ($y=0; $y -lt $rh; $y++) {
        for ($x=0; $x -lt $rw; $x++) {
            $p = $bmp.GetPixel($rx0+$x,$ry0+$y)
            $mask[$x,$y] = if ($inkName -eq "gold") { IsGold $p.R $p.G $p.B } else { IsWhite $p.R $p.G $p.B }
        }
    }
    $seen = New-Object 'bool[,]' $rw,$rh
    $found = New-Object System.Collections.Generic.List[object]
    for ($sy=0; $sy -lt $rh; $sy++) {
        for ($sx=0; $sx -lt $rw; $sx++) {
            if (-not $mask[$sx,$sy] -or $seen[$sx,$sy]) { continue }
            $stack = New-Object System.Collections.Generic.Stack[object]
            $stack.Push(@($sx,$sy)); $seen[$sx,$sy]=$true
            $bx0=$sx;$bx1=$sx;$by0=$sy;$by1=$sy
            while ($stack.Count -gt 0) {
                $cur=$stack.Pop(); $cx=$cur[0]; $cy=$cur[1]
                if ($cx -lt $bx0){$bx0=$cx}; if($cx -gt $bx1){$bx1=$cx}
                if ($cy -lt $by0){$by0=$cy}; if($cy -gt $by1){$by1=$cy}
                foreach ($d in @(@(1,0),@(-1,0),@(0,1),@(0,-1))) {
                    $nx=$cx+$d[0]; $ny=$cy+$d[1]
                    if ($nx -ge 0 -and $nx -lt $rw -and $ny -ge 0 -and $ny -lt $rh -and $mask[$nx,$ny] -and -not $seen[$nx,$ny]) {
                        $seen[$nx,$ny]=$true; $stack.Push(@($nx,$ny))
                    }
                }
            }
            $bw=$bx1-$bx0+1; $bh=$by1-$by0+1
            # Roughly glyph-sized: narrow, and short - a digit or a point,
            # not a button, a card, or the felt. Bounds converted back to
            # absolute frame coordinates before returning.
            if ($bw -ge 2 -and $bw -le 30 -and $bh -ge 2 -and $bh -le 30) {
                $found.Add([PSCustomObject]@{X0=($rx0+$bx0);Y0=($ry0+$by0);W=$bw;H=$bh})
            }
        }
    }
    return $found
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$i = 0
foreach ($region in $Regions) {
    $blobs = FindBlobsInRegion $bmp $region.Box $region.Ink
    foreach ($b in $blobs) {
        $i++
        $pad = 3
        $cx0 = [Math]::Max(0, $b.X0-$pad); $cy0 = [Math]::Max(0, $b.Y0-$pad)
        $cw = [Math]::Min($w-$cx0, $b.W+2*$pad); $ch = [Math]::Min($h-$cy0, $b.H+2*$pad)
        $rect = New-Object System.Drawing.Rectangle($cx0, $cy0, $cw, $ch)
        $crop = $bmp.Clone($rect, $bmp.PixelFormat)
        $big = New-Object System.Drawing.Bitmap($crop, ($cw*8), ($ch*8))
        $path = Join-Path $OutDir ("{0}-{1}-{2}-x{3}-y{4}-{5}x{6}.png" -f $stamp, $i, $region.Name, $b.X0, $b.Y0, $b.W, $b.H)
        $big.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    Write-Host "  $($region.Name): $($blobs.Count) blob(s)"
}
Write-Host "found $i glyph-sized blob(s) total, crops in $OutDir"

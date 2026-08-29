<#
.SYNOPSIS
  Captures the CoinPoker table and crops every visible card's rank+suit corner,
  for building the from-scratch template set.

.DESCRIPTION
  One shot: resize the table window to the canonical 1280x960, capture it with
  the project's own `poker grab`, find every card face (same bright-rectangle
  rule as poker-vision's detect_cards), split the overlapping hole-card pair
  into two, and crop+zoom each card's corner index (rank glyph, then suit pip)
  into its own small PNG for a human to identify and file into the template
  set.

  The window resets to a smaller default size on every new hand, so this
  re-resizes every time it runs rather than assuming a previous resize stuck.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\harvest-coinpoker.ps1
#>
param(
  [string]$RepoDir = "D:\poker-bot",
  [string]$OutDir = "D:\poker-bot\captures-coinpoker\harvest"
)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public class HarvestWin {
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
            if (w == 1280 && h == 778) return true; // the lobby
            if (w < 100 || h < 100) return true;    // stray tiny windows
            Handles.Add(hWnd);
            return true;
        }, IntPtr.Zero);
    }
}
"@

# --- 1. resize the table window (excluding the lobby) ------------------------
[HarvestWin]::Scan()
if ([HarvestWin]::Handles.Count -eq 0) {
    Write-Host "No CoinPoker table window found. Open a table with a hand showing and retry." -ForegroundColor Yellow
    exit 1
}
foreach ($h in [HarvestWin]::Handles) {
    [HarvestWin]::SetWindowPos($h, [IntPtr]::Zero, 100, 50, 1280, 960, 0x0004) | Out-Null
}
Start-Sleep -Milliseconds 250

# --- 2. capture via the project's own tool ------------------------------------
Push-Location $RepoDir
$stamp = Get-Date -Format "HHmmss"
& ".\target\debug\poker.exe" grab --process CoinPoker --out captures-coinpoker --label "h$stamp" | Out-Null
Pop-Location
$frame = Get-ChildItem (Join-Path $RepoDir "captures-coinpoker\h$stamp-0-1280x960.png") -ErrorAction SilentlyContinue
if (-not $frame) {
    Write-Host "No 1280x960 capture came back - is a table actually open and showing a hand?" -ForegroundColor Yellow
    exit 1
}
$src = $frame.FullName
Write-Host "captured: $src"

# --- 3. detect card boxes (same rule as poker-vision's detect_cards) ----------
$bmp = New-Object System.Drawing.Bitmap($src)
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

# Split any merged (overlapping) blob wider than one card into two 86-wide cards.
$CARD_W = 86
$cards = New-Object System.Collections.Generic.List[object]
foreach ($b in $boxes) {
    if ($b.W -le ($CARD_W + 10)) {
        $cards.Add([PSCustomObject]@{X=$b.X; Y=$b.Y})
    } else {
        $cards.Add([PSCustomObject]@{X=$b.X; Y=$b.Y})
        $cards.Add([PSCustomObject]@{X=($b.X + $b.W - $CARD_W); Y=$b.Y})
    }
}
Write-Host "found $($cards.Count) card(s)"
if ($cards.Count -eq 0) {
    Write-Host "No cards detected in this frame." -ForegroundColor Yellow
    exit 0
}

# --- 4. crop rank+suit corner from each card, scaled 5x -----------------------
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$RANK_TOP=8; $RANK_W=38; $RANK_H=31
$SUIT_TOP=38; $SUIT_W=34; $SUIT_H=36
$i = 0
foreach ($c in $cards) {
    $i++
    $rankRect = New-Object System.Drawing.Rectangle($c.X, ($c.Y+$RANK_TOP), $RANK_W, $RANK_H)
    $rank = $bmp.Clone($rankRect, $bmp.PixelFormat)
    $rankBig = New-Object System.Drawing.Bitmap($rank, ($RANK_W*5), ($RANK_H*5))
    $rankPath = Join-Path $OutDir "$stamp-card$i-rank.png"
    $rankBig.Save($rankPath, [System.Drawing.Imaging.ImageFormat]::Png)

    $suitRect = New-Object System.Drawing.Rectangle($c.X, ($c.Y+$SUIT_TOP), $SUIT_W, $SUIT_H)
    $suit = $bmp.Clone($suitRect, $bmp.PixelFormat)
    $suitBig = New-Object System.Drawing.Bitmap($suit, ($SUIT_W*5), ($SUIT_H*5))
    $suitPath = Join-Path $OutDir "$stamp-card$i-suit.png"
    $suitBig.Save($suitPath, [System.Drawing.Imaging.ImageFormat]::Png)

    Write-Host ("card {0} -> {1} | {2}" -f $i, (Split-Path $rankPath -Leaf), (Split-Path $suitPath -Leaf))
}
Write-Host "`nAll crops saved in $OutDir"

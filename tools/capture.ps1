<#
.SYNOPSIS
  Captures a window to PNG, for building the bot's card and label templates.

.DESCRIPTION
  Two jobs. Right now it collects training frames without manual keystrokes.
  Longer term it answers a question nothing else has: whether this app permits
  programmatic screen capture at all. An app that blocks it returns a flat
  rectangle, and that is a hard blocker for the whole vision approach - far
  better to learn it in ten seconds than after the recogniser is built.

  Windows can be selected three ways. Process name is the most reliable: a
  client renames its window as you navigate, but the executable behind it does
  not change.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File capture.ps1 -List
  powershell -NoProfile -ExecutionPolicy Bypass -File capture.ps1 -Index 4 -Count 1
  powershell -NoProfile -ExecutionPolicy Bypass -File capture.ps1 -Process ClubGG -Count 30
#>
[CmdletBinding()]
param(
  [switch]$List,
  [string]$Title,
  [string]$Process,
  [int]$Index = -1,
  [int]$Count = 30,
  [double]$Interval = 3.0,
  [int]$MinSize = 150,
  [string]$OutDir = "c:\poker\captures"
)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Win {
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr h);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

function Get-Windows {
  param([int]$Minimum)
  $script:found = @()
  $callback = [Win+EnumProc]{
    param($handle, $lparam)
    if (-not [Win]::IsWindowVisible($handle)) { return $true }

    $rect = New-Object Win+RECT
    [void][Win]::GetWindowRect($handle, [ref]$rect)
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    if ($width -lt $Minimum -or $height -lt $Minimum) { return $true }

    # An empty title is common in game clients, so it is listed rather than
    # skipped. Filtering on it is what hid the target last time.
    $text = New-Object System.Text.StringBuilder 512
    [void][Win]::GetWindowText($handle, $text, $text.Capacity)
    $caption = $text.ToString()
    if ([string]::IsNullOrWhiteSpace($caption)) { $caption = "(no title)" }

    $class = New-Object System.Text.StringBuilder 256
    [void][Win]::GetClassName($handle, $class, $class.Capacity)

    $procId = 0
    [void][Win]::GetWindowThreadProcessId($handle, [ref]$procId)
    $name = "?"
    try { $name = (Get-Process -Id $procId -ErrorAction Stop).ProcessName } catch { }

    $script:found += [pscustomobject]@{
      Handle  = $handle
      Process = $name
      Title   = $caption
      Class   = $class.ToString()
      Width   = $width
      Height  = $height
      Left    = $rect.Left
      Top     = $rect.Top
    }
    return $true
  }
  [void][Win]::EnumWindows($callback, [IntPtr]::Zero)
  return $script:found
}

$windows = @(Get-Windows -Minimum $MinSize | Sort-Object Process, Title)

if ($List -or (-not $Title -and -not $Process -and $Index -lt 0)) {
  Write-Host "`nVisible windows at least ${MinSize}px on both sides:`n"
  $n = 0
  $windows | ForEach-Object {
    Write-Host ("{0,3}  {1,-18} {2,5} x {3,-5}  {4}" -f $n, $_.Process, $_.Width, $_.Height, $_.Title)
    $n++
  }
  Write-Host "`nPick a row, then capture one frame to test:"
  Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File capture.ps1 -Index <n> -Count 1"
  Write-Host "`nOr select by executable, which survives the app renaming its window:"
  Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File capture.ps1 -Process ClubGG -Count 30`n"
  Write-Host "Nothing listed? Lower the threshold: -MinSize 50`n"
  return
}

$target = $null
if ($Index -ge 0) {
  if ($Index -ge $windows.Count) {
    Write-Error "No window at index $Index. There are $($windows.Count); run -List again."
    return
  }
  $target = $windows[$Index]
} elseif ($Process) {
  # Largest window belonging to the process: a client often owns small hidden
  # helpers alongside the one being played.
  $target = $windows | Where-Object { $_.Process -like "*$Process*" } |
            Sort-Object { $_.Width * $_.Height } -Descending | Select-Object -First 1
  if (-not $target) { Write-Error "No visible window from a process matching '*$Process*'."; return }
} else {
  $target = $windows | Where-Object { $_.Title -like "*$Title*" } |
            Sort-Object { $_.Width * $_.Height } -Descending | Select-Object -First 1
  if (-not $target) { Write-Error "No visible window titled like '*$Title*'. Run -List."; return }
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

# Display scaling changes template pixel sizes, so it is recorded alongside.
$dpi = (Get-ItemProperty -Path "HKCU:\Control Panel\Desktop\WindowMetrics" -Name AppliedDPI -ErrorAction SilentlyContinue).AppliedDPI
if (-not $dpi) { $dpi = 96 }
$scaling = [math]::Round($dpi / 96 * 100)

Write-Host "process  : $($target.Process)   (class $($target.Class))"
Write-Host "window   : $($target.Title)"
Write-Host "size     : $($target.Width) x $($target.Height) at ($($target.Left), $($target.Top))"
Write-Host "scaling  : $scaling%"
Write-Host "saving   : $OutDir`n"

[void][Win]::SetForegroundWindow($target.Handle)
Start-Sleep -Milliseconds 600

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$blank = 0
for ($i = 1; $i -le $Count; $i++) {
  $rect = New-Object Win+RECT
  [void][Win]::GetWindowRect($target.Handle, [ref]$rect)
  $w = $rect.Right - $rect.Left
  $h = $rect.Bottom - $rect.Top
  if ($w -le 0 -or $h -le 0) { Write-Warning "Window has closed."; break }

  $bitmap = New-Object System.Drawing.Bitmap $w, $h
  $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
  $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)

  # A window that blocks capture comes back one flat colour. Sampling a spread
  # of points rather than the whole frame keeps this cheap.
  $probes = @(
    $bitmap.GetPixel(5, 5), $bitmap.GetPixel($w - 6, 5),
    $bitmap.GetPixel(5, $h - 6), $bitmap.GetPixel($w - 6, $h - 6),
    $bitmap.GetPixel([int]($w / 2), [int]($h / 2)),
    $bitmap.GetPixel([int]($w / 3), [int]($h / 3))
  )
  $flat = ($probes | ForEach-Object { $_.ToArgb() } | Select-Object -Unique).Count -eq 1
  if ($flat) { $blank++ }

  $path = Join-Path $OutDir ("{0}-{1:d3}.png" -f $stamp, $i)
  $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
  $graphics.Dispose(); $bitmap.Dispose()

  Write-Host ("{0,3}/{1}  {2}  {3}x{4}{5}" -f $i, $Count, (Split-Path $path -Leaf), $w, $h, $(if ($flat) { "   <-- BLANK" } else { "" }))
  if ($i -lt $Count) { Start-Sleep -Seconds $Interval }
}

Write-Host ""
if ($Count -gt 0 -and $blank -eq $Count) {
  Write-Warning "Every frame was one flat colour. Either the window was hidden behind"
  Write-Warning "another, or the app blocks screen capture. Bring it to the front and"
  Write-Warning "retry; if it stays blank, the vision approach needs rethinking."
} elseif ($blank -gt 0) {
  Write-Warning "$blank of $Count frames were flat - probably captured mid-transition."
} else {
  Write-Host "$Count frames captured, none blank. Screen capture works on this app."
}

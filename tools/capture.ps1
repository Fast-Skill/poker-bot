<#
.SYNOPSIS
  Captures a window to PNG, for building the bot's card and label templates.

.DESCRIPTION
  Two jobs. Right now it collects training frames without manual keystrokes.
  Longer term it answers a question nothing else has: whether this app permits
  programmatic screen capture at all. An app that blocks it returns a black
  rectangle, and that is a hard blocker for the whole vision approach — far
  better to learn it in ten seconds than after the recogniser is built.

.EXAMPLE
  .\capture.ps1 -List
  .\capture.ps1 -Title "PokerApp" -Count 30 -Interval 3
  .\capture.ps1 -Title "PokerApp" -Count 1        # one frame, to test
#>
[CmdletBinding()]
param(
  [switch]$List,
  [string]$Title,
  [int]$Count = 30,
  [double]$Interval = 3.0,
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
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

function Get-Windows {
  $found = @()
  $callback = [Win+EnumProc]{
    param($handle, $lparam)
    if ([Win]::IsWindowVisible($handle)) {
      $length = [Win]::GetWindowTextLength($handle)
      if ($length -gt 0) {
        $text = New-Object System.Text.StringBuilder ($length + 1)
        [void][Win]::GetWindowText($handle, $text, $text.Capacity)
        $rect = New-Object Win+RECT
        [void][Win]::GetWindowRect($handle, [ref]$rect)
        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        # Skip tool windows and tray helpers; a poker table is never tiny.
        if ($width -gt 400 -and $height -gt 300) {
          $script:found += [pscustomobject]@{
            Handle = $handle; Title = $text.ToString()
            Width = $width; Height = $height
            Left = $rect.Left; Top = $rect.Top
          }
        }
      }
    }
    return $true
  }
  $script:found = @()
  [void][Win]::EnumWindows($callback, [IntPtr]::Zero)
  return $script:found
}

if ($List -or -not $Title) {
  Write-Host "`nVisible windows larger than 400x300:`n"
  Get-Windows | Sort-Object Title | Format-Table Title, Width, Height -AutoSize
  Write-Host "Pick the game's title, then run:"
  Write-Host "  .\capture.ps1 -Title `"<part of the title>`" -Count 30 -Interval 3`n"
  return
}

$target = Get-Windows | Where-Object { $_.Title -like "*$Title*" } | Select-Object -First 1
if (-not $target) {
  Write-Error "No visible window matching '*$Title*'. Run with -List to see what is open."
  return
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

# Display scaling changes template pixel sizes, so it is recorded alongside.
$dpi = (Get-ItemProperty -Path "HKCU:\Control Panel\Desktop\WindowMetrics" -Name AppliedDPI -ErrorAction SilentlyContinue).AppliedDPI
if (-not $dpi) { $dpi = 96 }
$scaling = [math]::Round($dpi / 96 * 100)

Write-Host "window   : $($target.Title)"
Write-Host "size     : $($target.Width) x $($target.Height) at ($($target.Left), $($target.Top))"
Write-Host "scaling  : $scaling%"
Write-Host "saving   : $OutDir`n"

[void][Win]::SetForegroundWindow($target.Handle)
Start-Sleep -Milliseconds 400

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$blank = 0
for ($i = 1; $i -le $Count; $i++) {
  # Re-read the rectangle each frame in case the window moved.
  $rect = New-Object Win+RECT
  [void][Win]::GetWindowRect($target.Handle, [ref]$rect)
  $w = $rect.Right - $rect.Left
  $h = $rect.Bottom - $rect.Top

  $bitmap = New-Object System.Drawing.Bitmap $w, $h
  $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
  $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)

  # A window that blocks capture comes back a single flat colour.
  $corners = @(
    $bitmap.GetPixel(5, 5), $bitmap.GetPixel($w - 6, 5),
    $bitmap.GetPixel(5, $h - 6), $bitmap.GetPixel([int]($w / 2), [int]($h / 2))
  )
  $identical = ($corners | Select-Object -Unique).Count -eq 1
  if ($identical) { $blank++ }

  $path = Join-Path $OutDir ("{0}-{1:d3}.png" -f $stamp, $i)
  $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
  $graphics.Dispose(); $bitmap.Dispose()

  Write-Host ("{0,3}/{1}  {2}{3}" -f $i, $Count, (Split-Path $path -Leaf), $(if ($identical) { "   <-- BLANK" } else { "" }))
  if ($i -lt $Count) { Start-Sleep -Seconds $Interval }
}

Write-Host ""
if ($blank -eq $Count) {
  Write-Warning "Every frame came back blank. This app is blocking screen capture, which"
  Write-Warning "rules out the vision approach as designed. Tell Claude before going further."
} elseif ($blank -gt 0) {
  Write-Warning "$blank of $Count frames were blank - likely captured mid-transition."
} else {
  Write-Host "$Count frames captured, none blank. Screen capture works on this app."
}

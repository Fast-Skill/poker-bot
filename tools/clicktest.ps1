<#
.SYNOPSIS
  Tests whether the app accepts mouse clicks injected by a program.

.DESCRIPTION
  This is the last unverified assumption before a bot can act. Some
  applications filter synthetic input, in which case the bot could read the
  table perfectly and still be unable to do anything about it.

  The test clicks a harmless target and compares the screen before and after.
  A visible change means the click landed. Nothing changing means either the
  click missed the target or the app ignored it, and the two are separated by
  also checking whether the cursor physically moved.

  Two injection methods are tried in order. SendInput is what a bot would
  actually use and enters the same queue as real hardware. mouse_event is the
  older call, kept as a fallback so a failure of one does not read as a failure
  of both.

.EXAMPLE
  # Pick a harmless target from the lobby: the PLO tab.
  .\clicktest.ps1 -Process ClubGG -X 199 -Y 413

.NOTES
  Choose a target that changes something visible and costs nothing - a view
  tab, not a poker action. Do not run this while sitting in a hand.
#>
[CmdletBinding()]
param(
  [string]$Process = "ClubGG",
  [Parameter(Mandatory=$true)][int]$X,      # relative to the window's top-left
  [Parameter(Mandatory=$true)][int]$Y,
  [string]$OutDir = "c:\poker\captures\clicktest"
)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Click {
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [StructLayout(LayoutKind.Sequential)] public struct MOUSEINPUT {
    public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public IntPtr dwExtraInfo;
  }
  [StructLayout(LayoutKind.Sequential)] public struct INPUT {
    public uint type; public MOUSEINPUT mi;
  }
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT p);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern uint SendInput(uint n, INPUT[] inputs, int size);
  [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, IntPtr extra);
  public const uint MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004;

  public static uint SendInputClick() {
    INPUT[] inputs = new INPUT[2];
    inputs[0].type = 0; inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    inputs[1].type = 0; inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
    return SendInput(2, inputs, Marshal.SizeOf(typeof(INPUT)));
  }
  public static void LegacyClick() {
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, IntPtr.Zero);
    System.Threading.Thread.Sleep(40);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, IntPtr.Zero);
  }
}
"@

$app = Get-Process -Name $Process -ErrorAction SilentlyContinue |
       Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
if (-not $app) { Write-Error "No running process named '$Process' with a window."; return }

$handle = $app.MainWindowHandle
$rect = New-Object Click+RECT
[void][Click]::GetWindowRect($handle, [ref]$rect)
$screenX = $rect.Left + $X
$screenY = $rect.Top + $Y

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

function Grab($label) {
  $r = New-Object Click+RECT
  [void][Click]::GetWindowRect($handle, [ref]$r)
  $w = $r.Right - $r.Left; $h = $r.Bottom - $r.Top
  $bmp = New-Object System.Drawing.Bitmap $w, $h
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($r.Left, $r.Top, 0, 0, $bmp.Size)
  $g.Dispose()
  $bmp.Save((Join-Path $OutDir "$label.png"), [System.Drawing.Imaging.ImageFormat]::Png)
  return $bmp
}

function Difference($a, $b) {
  # Sample a grid rather than every pixel: enough to detect a view change,
  # fast enough to run between clicks.
  $changed = 0; $total = 0
  for ($y = 0; $y -lt $a.Height; $y += 4) {
    for ($x = 0; $x -lt $a.Width; $x += 4) {
      $total++
      if ($a.GetPixel($x,$y).ToArgb() -ne $b.GetPixel($x,$y).ToArgb()) { $changed++ }
    }
  }
  return [math]::Round($changed / $total * 100, 2)
}

Write-Host "window : $($app.MainWindowTitle)"
Write-Host "size   : $($rect.Right-$rect.Left) x $($rect.Bottom-$rect.Top) at ($($rect.Left), $($rect.Top))"
Write-Host "target : window ($X, $Y) -> screen ($screenX, $screenY)`n"

[void][Click]::SetForegroundWindow($handle)
Start-Sleep -Milliseconds 700

$before = Grab "1-before"

# Baseline: how much does the screen change on its own? Animations, timers and
# a ticking jackpot all move without anyone clicking, and that noise has to be
# measured or it will be mistaken for a successful click.
Start-Sleep -Milliseconds 900
$idle = Grab "2-idle"
$noise = Difference $before $idle
Write-Host ("idle drift     : {0,6}%  (animation and timers, with nothing clicked)" -f $noise)

$origin = New-Object Click+POINT
[void][Click]::GetCursorPos([ref]$origin)

[void][Click]::SetCursorPos($screenX, $screenY)
Start-Sleep -Milliseconds 250
$moved = New-Object Click+POINT
[void][Click]::GetCursorPos([ref]$moved)
$cursorOk = ($moved.X -eq $screenX -and $moved.Y -eq $screenY)
Write-Host ("cursor moved   : {0}" -f $(if ($cursorOk) { "yes, to ($($moved.X), $($moved.Y))" } else { "NO - blocked at ($($moved.X), $($moved.Y))" }))

$sent = [Click]::SendInputClick()
Start-Sleep -Milliseconds 900
$afterSend = Grab "3-after-sendinput"
$sendChange = Difference $idle $afterSend
Write-Host ("SendInput      : accepted {0}/2 events, screen changed {1}%" -f $sent, $sendChange)

$legacyChange = 0
if ($sendChange -le [math]::Max($noise * 2, 1.0)) {
  Write-Host "                 no clear change - trying the legacy call"
  [void][Click]::SetCursorPos($screenX, $screenY)
  Start-Sleep -Milliseconds 200
  [Click]::LegacyClick()
  Start-Sleep -Milliseconds 900
  $afterLegacy = Grab "4-after-mouseevent"
  $legacyChange = Difference $afterSend $afterLegacy
  Write-Host ("mouse_event    : screen changed {0}%" -f $legacyChange)
}

[void][Click]::SetCursorPos($origin.X, $origin.Y)

Write-Host "`n--- verdict ---"
$threshold = [math]::Max($noise * 3, 1.5)
if (-not $cursorOk) {
  Write-Host "The cursor could not be moved. Something is blocking pointer control" -ForegroundColor Yellow
  Write-Host "system-wide - check for an overlay, a remote session, or elevation mismatch." -ForegroundColor Yellow
} elseif ($sendChange -gt $threshold) {
  Write-Host "CLICKS WORK. SendInput was accepted and the app responded." -ForegroundColor Green
} elseif ($legacyChange -gt $threshold) {
  Write-Host "CLICKS WORK via mouse_event, but SendInput was ignored." -ForegroundColor Green
  Write-Host "The bot should use the legacy call." -ForegroundColor Green
} else {
  Write-Host "NO RESPONSE from either method." -ForegroundColor Yellow
  Write-Host "Either the coordinates missed the target, or the app filters" -ForegroundColor Yellow
  Write-Host "synthetic input. Check the saved frames in $OutDir - if 3-after" -ForegroundColor Yellow
  Write-Host "shows the cursor sitting on the right control, it is being ignored." -ForegroundColor Yellow
}
Write-Host "`nframes saved to $OutDir for inspection."

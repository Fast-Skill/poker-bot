@echo off
REM Collects the frames the card reader cannot yet name.
REM
REM Run it from anywhere: it sets its own working directory, because the
REM template files are found relative to the project root and running from
REM somewhere else is the one way this fails.
REM
REM It resizes the ClubGG table to 1430x1040 itself and refuses to run if the
REM client will not accept that size, so frames cannot come out unusable.
REM It only watches - it will not click anything.

cd /d "%~dp0"

if not exist "target\release\poker.exe" (
  echo Build it first:  cargo build --release
  exit /b 1
)

echo Collecting for 30 minutes. Play normally - just let the cards be dealt.
echo Press Ctrl+C to stop early.
echo.
target\release\poker.exe live --seconds 1800 --keep-unread captures\unread %*

@echo off
setlocal
cd /d "%~dp0"
title CricketLens
echo.
echo  CricketLens - Indian cricketer face identification
echo  ---------------------------------------------------
echo.

REM ---- 1. find a compatible Python (3.10 - 3.12) ----
set PY=
py -3.11 --version >nul 2>&1
if not errorlevel 1 set PY=py -3.11
if not defined PY (
  py -3.10 --version >nul 2>&1
  if not errorlevel 1 set PY=py -3.10
)
if not defined PY (
  py -3.12 --version >nul 2>&1
  if not errorlevel 1 set PY=py -3.12
)
if not defined PY (
  python --version >nul 2>&1
  if not errorlevel 1 set PY=python
)
if not defined PY (
  echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup. Then run this file again.
  goto :fail
)

REM ---- 2. virtual environment ----
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"

REM ---- 3. install libraries (first run only, several minutes) ----
if not exist ".venv\installed.flag" (
  echo [2/4] Installing libraries. This takes a few minutes the first time...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 goto :fail
  echo ok> ".venv\installed.flag"
)

REM ---- 4. build the face gallery if there is none yet (optional: you can also add players inside the app) ----
if not exist "models\gallery.npz" (
  echo [3/4] Building the face gallery from the dataset folder...
  python build_gallery.py
  echo.
  echo If the gallery is empty, no problem: open the app and use "Add a player" in the sidebar.
)

REM ---- 5. start the app and open the browser ----
echo [4/4] Starting CricketLens at http://127.0.0.1:5000  (close this window to stop)
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:5000"
python app.py
goto :end

:fail
echo.
echo Something went wrong. Read the message above, fix it, and run this file again.
:end
pause

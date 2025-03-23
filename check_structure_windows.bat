@echo off
REM Script to verify the wake word engine's file structure on Windows

echo Checking wake word engine file structure...
echo ===========================================

REM Variables to track missing items
set missing_dirs=0
set missing_files=0

REM Check main directories
echo Checking directories...
set directories=. .\audio .\model .\ui .\utils

for %%d in (%directories%) do (
    if exist "%%d\" (
        echo [OK] Directory exists: %%d
    ) else (
        echo [MISSING] Directory not found: %%d
        set /a missing_dirs+=1
    )
)

REM Check files
echo.
echo Checking files...
set files=^
    .\__init__.py ^
    .\main.py ^
    .\setup.py ^
    .\README.md ^
    .\audio\__init__.py ^
    .\audio\capture.py ^
    .\audio\features.py ^
    .\audio\vad.py ^
    .\model\__init__.py ^
    .\model\architecture.py ^
    .\model\inference.py ^
    .\model\training.py ^
    .\ui\__init__.py ^
    .\ui\tray.py ^
    .\ui\config.py ^
    .\ui\training_ui.py ^
    .\utils\__init__.py ^
    .\utils\config.py ^
    .\utils\actions.py

for %%f in (%files%) do (
    if exist "%%f" (
        echo [OK] File exists: %%f
    ) else (
        echo [MISSING] File not found: %%f
        set /a missing_files+=1
    )
)

REM Summary
echo.
echo ===========================================
if %missing_dirs% EQU 0 if %missing_files% EQU 0 (
    echo All directories and files are present.
    echo The wake word engine structure is complete!
) else (
    echo Missing items detected:
    echo   - Missing directories: %missing_dirs%
    echo   - Missing files: %missing_files%
    echo Please create the missing items to complete the structure.
)

REM Check Python dependencies
echo.
echo Checking Python dependencies...
echo ===========================================

set dependencies=numpy pyaudio torch librosa PySimpleGUI pystray PIL sklearn

set missing_deps=0
for %%d in (%dependencies%) do (
    python -c "import %%d" 2>NUL
    if errorlevel 1 (
        echo [MISSING] Dependency not installed: %%d
        set /a missing_deps+=1
    ) else (
        echo [OK] Dependency installed: %%d
    )
)

if %missing_deps% EQU 0 (
    echo All dependencies are installed.
) else (
    echo Missing dependencies: %missing_deps%
    echo Run: pip install -e . to install missing dependencies
)

REM Pause to view results
echo.
pause

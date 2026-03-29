@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo       AutoSWD Launcher for Windows
echo ===================================================
echo.

:: -------- Create Desktop Shortcut --------
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\AutoSWD.lnk"

if not exist "%SHORTCUT_PATH%" (
    echo Creating Desktop shortcut...
    powershell -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\AutoSWD.lnk'); $s.TargetPath = '%~f0'; $s.WorkingDirectory = '%~dp0.'; $s.Save()"
    if exist "%SHORTCUT_PATH%" echo Desktop shortcut created successfully!
    echo.
)
:: -----------------------------------------

:: Check for Conda first
call conda --version >nul 2>&1
if not errorlevel 1 goto use_conda

:: ============ PYTHON CHECK ============
set "PYTHON_CMD="

:: First try the standard python command
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto python_found
)

:: Try py launcher
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto python_found
)

:: Try common installation paths
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto python_found
)

if exist "%PROGRAMFILES%\Python311\python.exe" (
    set "PYTHON_CMD=%PROGRAMFILES%\Python311\python.exe"
    goto python_found
)

:: Python not found - need to install
goto install_python

:use_conda
echo Conda detected.
echo Creating/Updating Conda environment from environment_windows.yml...
call conda env update -f environment_windows.yml --prune
echo Activating Conda environment...
call conda activate autoswd_env
echo.
echo Launching AutoSWD...
python AutoSWD.py 2>nul
echo.
pause
exit /b 0

:install_python
echo.
echo [INFO] Python is not installed or not in PATH.
echo.

:: Check if we already tried installing
if exist "%TEMP%\autoswd_python_installing.tmp" (
    echo [NOTE] Python was recently installed but PATH may not be updated.
    echo Please close ALL command prompt windows and run this script again.
    echo.
    del "%TEMP%\autoswd_python_installing.tmp" >nul 2>&1
    pause
    exit /b 1
)

echo Attempting to download Python 3.11.5 installer...
echo.

powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe' -OutFile 'python_installer.exe'"

if not exist python_installer.exe (
    echo [ERROR] Failed to download Python installer.
    echo Please manually install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo =====================================================
echo           PYTHON INSTALLATION
echo =====================================================
echo.
echo [IMPORTANT] CHECK "Add python.exe to PATH" at the bottom!
echo.
echo Press any key to launch the Python installer...
pause >nul

echo installing > "%TEMP%\autoswd_python_installing.tmp"
start /wait python_installer.exe
del python_installer.exe >nul 2>&1

echo.
echo Python installation completed.
echo Please close this window and run the script again.
echo.
del "%TEMP%\autoswd_python_installing.tmp" >nul 2>&1
pause
exit /b 0

:python_found
echo Python detected: %PYTHON_CMD%
echo.

:: Check/create virtual environment
if exist env goto env_exists

echo First time setup - Creating virtual environment...
echo.

"%PYTHON_CMD%" -m venv env
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

call env\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo =====================================================
echo Installing dependencies with exact versions...
echo This may take several minutes on first run.
echo =====================================================
echo.

:: Install all packages with exact versions from environment_windows.yml
echo Installing numpy==1.26.4...
pip install numpy==1.26.4

echo Installing pandas==2.2.0...
pip install pandas==2.2.0

echo Installing matplotlib==3.8.0...
pip install matplotlib==3.8.0

echo Installing scipy==1.13.1...
pip install scipy==1.13.1

echo Installing mne==1.6.1...
pip install mne==1.6.1

echo Installing PyQt5==5.15.10...
pip install PyQt5==5.15.10

echo Installing tensorflow==2.15.0...
pip install tensorflow==2.15.0

echo Installing torch==2.2.0...
pip install torch==2.2.0

echo Installing h5py==3.10.0...
pip install h5py==3.10.0

echo Installing openpyxl==3.1.2...
pip install openpyxl==3.1.2

echo Installing joblib==1.3.2...
pip install joblib==1.3.2

echo.
echo =====================================================
echo All dependencies installed successfully!
echo =====================================================
echo.
goto launch

:env_exists
echo Virtual environment found.
call env\Scripts\activate
goto launch

:launch
echo ===================================================
echo           Launching AutoSWD...
echo ===================================================
echo.
python AutoSWD.py 2>nul
echo.
echo ===================================================
echo AutoSWD closed. Press any key to exit.
echo ===================================================
pause >nul

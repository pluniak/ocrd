@echo off
REM Rune this file in Windows from a Conda terminal

REM Name of the Conda environment
set ENV_NAME=ocrd

REM Path to the requirements.txt file
set REQUIREMENTS_FILE=requirements.txt

REM Check if the Conda environment exists
conda info --envs | findstr /C:"%ENV_NAME%" > nul
if errorlevel 1 (
    echo Environment %ENV_NAME% does not exist, creating...
    conda create --name %ENV_NAME% python=3.10 --yes
    echo Activating environment and installing packages from requirements.txt using pip...
    call activate %ENV_NAME%
    pip install -r %REQUIREMENTS_FILE%
) else (
    echo Environment %ENV_NAME% exists, updating...
    call activate %ENV_NAME%
    pip install -r %REQUIREMENTS_FILE%
)

echo Operation completed.


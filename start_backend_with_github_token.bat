@echo off
REM Windows batch script to start backend with GitHub token

echo ========================================
echo Resume Reviewer Backend Starter
echo ========================================
echo.

REM Check if GITHUB_TOKEN is provided as argument
if "%1"=="" (
    echo Usage: start_backend_with_github_token.bat YOUR_GITHUB_TOKEN
    echo.
    echo Or set it first:
    echo   set GITHUB_TOKEN=ghp_your_token_here
    echo   start_backend_with_github_token.bat
    echo.

    REM Check if already set in environment
    if "%GITHUB_TOKEN%"=="" (
        echo [ERROR] GITHUB_TOKEN not set!
        echo.
        echo Please either:
        echo   1. Run: start_backend_with_github_token.bat ghp_your_token
        echo   2. Set environment variable first: set GITHUB_TOKEN=ghp_your_token
        echo.
        pause
        exit /b 1
    ) else (
        echo [OK] Using GITHUB_TOKEN from environment
        echo Token: %GITHUB_TOKEN:~0,10%...
    )
) else (
    REM Set token from argument
    set GITHUB_TOKEN=%1
    echo [OK] GITHUB_TOKEN set from argument
    echo Token: %GITHUB_TOKEN:~0,10%...
)

echo.
echo Starting backend server...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Start uvicorn
uvicorn backend.app.main:app --reload

pause

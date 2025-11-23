@echo off
REM Quick start script for Resume Reviewer Docker setup (Windows)

echo =======================================================================
echo Resume Reviewer - Docker Quick Start
echo =======================================================================
echo.

REM Check if docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    exit /b 1
)

REM Check if docker-compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] docker-compose is not installed!
    echo Please install docker-compose
    exit /b 1
)

echo [OK] Docker is installed
echo.

REM Check if .env file exists
if not exist .env (
    echo [INFO] No .env file found. Creating from .env.example...
    copy .env.example .env >nul
    echo [OK] Created .env file
    echo.
    echo [NOTE] To enable GitHub analysis, edit .env and add your GITHUB_TOKEN
    echo.
) else (
    echo [OK] .env file exists
    echo.
)

REM Stop any existing containers
echo Stopping existing containers...
docker-compose down 2>nul
echo.

REM Pull latest images
echo Pulling Docker images...
docker-compose pull
echo.

REM Build services
echo Building application containers...
docker-compose build
echo.

REM Start services
echo Starting all services...
docker-compose up -d
echo.

REM Wait for services to start
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo =======================================================================
echo Service Status
echo =======================================================================
docker-compose ps
echo.

REM Check if backend is healthy
echo Checking backend health...
for /L %%i in (1,1,30) do (
    curl -f http://localhost:8000/api/health >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] Backend is healthy!
        goto :backend_ready
    )
    echo Waiting for backend... (%%i/30)
    timeout /t 2 /nobreak >nul
)

:backend_ready
echo.
echo =======================================================================
echo Resume Reviewer is Ready!
echo =======================================================================
echo.
echo Access the application:
echo   * Frontend:  http://localhost:3000
echo   * Backend:   http://localhost:8000
echo   * API Docs:  http://localhost:8000/docs
echo.
echo View logs:
echo   docker-compose logs -f
echo.
echo Stop services:
echo   docker-compose down
echo.
echo =======================================================================
pause

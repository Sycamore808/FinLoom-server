@echo off
REM FinLoom Vue3前端构建脚本 (Windows版本)
REM 用于构建Vue3前端并部署到web/dist目录

setlocal enabledelayedexpansion

echo ================================
echo FinLoom Vue3 前端构建脚本
echo ================================
echo.

REM 检查Node.js是否安装
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] Node.js未安装
    echo 请先安装Node.js: https://nodejs.org/
    pause
    exit /b 1
)

echo [成功] Node.js已安装
node --version

REM 检查npm是否安装
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] npm未安装
    pause
    exit /b 1
)

echo [成功] npm已安装
npm --version
echo.

REM 进入Vue项目目录
cd web-vue

REM 检查package.json是否存在
if not exist "package.json" (
    echo [错误] package.json不存在
    pause
    exit /b 1
)

REM 安装依赖
echo [进行中] 安装依赖...
if not exist "node_modules" (
    call npm install --registry=https://registry.npmmirror.com
) else (
    echo [成功] 依赖已存在，跳过安装
)
echo.

REM 构建生产版本
echo [进行中] 构建生产版本...
call npm run build

REM 检查构建是否成功
if exist "..\web\dist\index.html" (
    echo.
    echo ================================
    echo [成功] 构建成功！
    echo ================================
    echo.
    echo 构建产物位置: web\dist\
    echo.
    echo 下一步：
    echo 1. 启动后端服务: python main.py
    echo 2. 访问 http://localhost:8000
    echo.
) else (
    echo [错误] 构建失败
    pause
    exit /b 1
)

pause


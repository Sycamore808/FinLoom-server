#!/bin/bash

# FinLoom Vue3前端构建脚本
# 用于构建Vue3前端并部署到web/dist目录

set -e  # 遇到错误立即退出

echo "================================"
echo "FinLoom Vue3 前端构建脚本"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ 错误: Node.js未安装${NC}"
    echo "请先安装Node.js: https://nodejs.org/"
    exit 1
fi

echo -e "${GREEN}✓ Node.js 版本: $(node --version)${NC}"

# 检查npm是否安装
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ 错误: npm未安装${NC}"
    exit 1
fi

echo -e "${GREEN}✓ npm 版本: $(npm --version)${NC}"
echo ""

# 进入Vue项目目录
cd web-vue

# 检查package.json是否存在
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ 错误: package.json不存在${NC}"
    exit 1
fi

# 安装依赖
echo -e "${YELLOW}📦 安装依赖...${NC}"
if [ ! -d "node_modules" ]; then
    npm install --registry=https://registry.npmmirror.com
else
    echo -e "${GREEN}✓ 依赖已存在，跳过安装${NC}"
fi
echo ""

# 构建生产版本
echo -e "${YELLOW}🔨 构建生产版本...${NC}"
npm run build

# 检查构建是否成功
if [ -f "../web/dist/index.html" ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}✅ 构建成功！${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "构建产物位置: web/dist/"
    echo ""
    echo "下一步："
    echo "1. 启动后端服务: python main.py"
    echo "2. 访问 http://localhost:8000"
    echo ""
else
    echo -e "${RED}❌ 构建失败${NC}"
    exit 1
fi


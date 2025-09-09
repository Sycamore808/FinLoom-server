#!/usr/bin/env python3
"""
FinLoom Web应用启动脚本
启动完整的Web应用，包括前端界面和后端API
"""

import os
import sys
import asyncio
import webbrowser
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{project_root}"

from main import FinLoomEngine

async def start_web_app():
    """启动Web应用"""
    print("🚀 启动FinLoom Web应用...")
    print("=" * 50)
    
    # 创建引擎实例
    engine = FinLoomEngine()
    
    try:
        # 初始化引擎
        print("📋 初始化系统...")
        await engine.initialize()
        print("✅ 系统初始化完成")
        
        # 启动API服务器
        print("🌐 启动Web服务器...")
        print("📍 访问地址: http://localhost:8000")
        print("🔧 API文档: http://localhost:8000/docs")
        print("💡 按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        # 自动打开浏览器
        try:
            webbrowser.open('http://localhost:8000')
            print("🌍 已自动打开浏览器")
        except Exception as e:
            print(f"⚠️  无法自动打开浏览器: {e}")
            print("请手动访问: http://localhost:8000")
        
        # 启动服务器
        await engine.start_api_server(host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    try:
        asyncio.run(start_web_app())
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

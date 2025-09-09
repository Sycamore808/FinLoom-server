#!/usr/bin/env python3
"""
FinLoom Web应用启动脚本
启动完整的Web应用，包括前端界面和后端API
"""

import os
import sys
import asyncio
import webbrowser
import subprocess
import socket
import time
import requests
from pathlib import Path

# 添加项目根目录到Python路径3
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:{project_root}"

# 虚拟环境路径
venv_path = project_root / ".venv"

def setup_virtual_environment():
    """设置虚拟环境，优先使用uv"""
    print("🔧 设置虚拟环境...")
    
    # 检查uv是否可用
    uv_available = False
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            uv_available = True
            print(f"✅ 找到 uv: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  uv 不可用，将使用标准 venv")
    
    # 创建虚拟环境
    if not venv_path.exists():
        print("📦 创建虚拟环境...")
        try:
            if uv_available:
                # 使用uv创建虚拟环境，确保包含pip
                cmd = ["uv", "venv", str(venv_path), "--python", "python3"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ 使用 uv 创建虚拟环境成功")
                    # 确保pip已安装
                    pip_cmd = [str(venv_path / "bin" / "python"), "-m", "ensurepip", "--upgrade"]
                    subprocess.run(pip_cmd, capture_output=True, text=True, timeout=30)
                else:
                    print(f"❌ uv 创建虚拟环境失败: {result.stderr}")
                    raise Exception("uv failed")
            else:
                # 使用标准venv创建虚拟环境
                import venv
                venv.create(venv_path, with_pip=True)
                print("✅ 使用标准 venv 创建虚拟环境成功")
        except Exception as e:
            print(f"❌ 创建虚拟环境失败: {e}")
            return False
    else:
        print("✅ 虚拟环境已存在")
    
    # 确定Python可执行文件路径
    if os.name == 'nt':  # Windows
        python_executable = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        python_executable = venv_path / "bin" / "python"
    
    if not python_executable.exists():
        print(f"❌ 虚拟环境中找不到Python可执行文件: {python_executable}")
        print("🔄 尝试重新创建虚拟环境...")
        try:
            # 删除损坏的虚拟环境
            import shutil
            if venv_path.exists():
                shutil.rmtree(venv_path)
            
            # 重新创建虚拟环境
            import venv
            venv.create(venv_path, with_pip=True)
            print("✅ 重新创建虚拟环境成功")
            
            # 再次检查Python可执行文件
            if not python_executable.exists():
                print(f"❌ 重新创建后仍找不到Python可执行文件: {python_executable}")
                return False
        except Exception as e:
            print(f"❌ 重新创建虚拟环境失败: {e}")
            return False
    
    # 更新sys.executable
    sys.executable = str(python_executable)
    print(f"🐍 使用虚拟环境Python: {python_executable}")
    
    # 检查并安装依赖
    if not install_dependencies(python_executable):
        print("⚠️  依赖安装失败，但继续运行...")
    
    return True

def install_dependencies(python_executable):
    """安装项目依赖"""
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        print("⚠️  未找到 requirements.txt 文件")
        return False
    
    print("📦 安装项目依赖（使用清华源）...")
    try:
        # 使用虚拟环境的pip安装依赖，指定清华源
        cmd = [
            str(python_executable), "-m", "pip", "install", 
            "-r", str(requirements_file),
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✅ 依赖安装成功")
            return True
        else:
            print(f"❌ 依赖安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 依赖安装异常: {e}")
        return False

def check_port_available(port):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8000, max_port=8010):
    """查找可用端口"""
    for port in range(start_port, max_port + 1):
        if check_port_available(port):
            return port
    return None

def kill_process_on_port(port):
    """终止占用指定端口的进程"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"🔪 终止占用端口{port}的进程 PID: {pid}")
                    subprocess.run(['kill', pid], capture_output=True)
            return True
    except Exception as e:
        print(f"⚠️  无法终止进程: {e}")
    return False

async def wait_for_system_ready(port, max_wait_time=30):
    """等待系统完全启动并准备就绪"""
    print("⏳ 等待系统完全启动...")
    start_time = time.time()
    check_count = 0
    max_checks = max_wait_time // 2  # 每2秒检查一次
    
    while time.time() - start_time < max_wait_time and check_count < max_checks:
        check_count += 1
        try:
            # 简化检查：只检查基本的健康状态
            health_response = requests.get(f"http://localhost:{port}/health", timeout=3)
            if health_response.status_code == 200:
                health_data = health_response.json()
                status = health_data.get('status', 'unknown')
                print(f"📊 系统健康状态: {status}")
                
                # 如果系统健康，就认为可以继续
                if status in ['healthy', 'initializing']:
                    print("✅ 系统已启动，可以访问")
                    return True
                else:
                    print(f"⏳ 系统状态: {status}")
            else:
                print(f"⏳ 等待服务器启动... (状态码: {health_response.status_code})")
                
        except requests.exceptions.ConnectionError:
            print("⏳ 等待服务器启动...")
        except requests.exceptions.Timeout:
            print("⏳ 服务器响应超时，继续等待...")
        except Exception as e:
            print(f"⚠️  检查系统状态时出错: {e}")
        
        # 增加等待时间，避免过于频繁的检查
        await asyncio.sleep(3)
    
    print("⚠️  系统启动超时，但继续运行...")
    return False

# 在导入其他模块之前设置虚拟环境
if not setup_virtual_environment():
    print("❌ 虚拟环境设置失败，退出程序")
    sys.exit(1)

from main import FinLoomEngine

async def start_web_app():
    """启动Web应用"""
    print("🚀 启动FinLoom Web应用...")
    print("=" * 50)
    
    # 检查并处理端口冲突
    preferred_port = 8000
    if not check_port_available(preferred_port):
        print(f"⚠️  端口 {preferred_port} 被占用，尝试释放...")
        if kill_process_on_port(preferred_port):
            # 等待一下让进程完全终止
            await asyncio.sleep(2)
            if check_port_available(preferred_port):
                print(f"✅ 端口 {preferred_port} 已释放")
            else:
                print(f"❌ 无法释放端口 {preferred_port}，寻找其他可用端口...")
                preferred_port = find_available_port()
                if preferred_port is None:
                    print("❌ 无法找到可用端口，请手动终止占用端口的进程")
                    sys.exit(1)
                print(f"✅ 找到可用端口: {preferred_port}")
        else:
            print(f"❌ 无法释放端口 {preferred_port}，寻找其他可用端口...")
            preferred_port = find_available_port()
            if preferred_port is None:
                print("❌ 无法找到可用端口，请手动终止占用端口的进程")
                sys.exit(1)
            print(f"✅ 找到可用端口: {preferred_port}")
    else:
        print(f"✅ 端口 {preferred_port} 可用")
    
    # 创建引擎实例
    engine = FinLoomEngine()
    
    try:
        # 初始化引擎
        print("📋 初始化系统...")
        await engine.initialize()
        print("✅ 系统初始化完成")
        
        # 启动API服务器
        print("🌐 启动Web服务器...")
        print(f"📍 访问地址: http://localhost:{preferred_port}")
        print(f"🔧 API文档: http://localhost:{preferred_port}/docs")
        print("💡 按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        # 启动服务器
        print("🚀 正在启动服务器...")
        
        # 创建一个简单的启动检查
        async def simple_startup_check():
            """简单的启动检查"""
            for i in range(10):  # 最多检查10次
                try:
                    response = requests.get(f"http://localhost:{preferred_port}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ 服务器已启动")
                        return True
                except:
                    pass
                await asyncio.sleep(1)
            return False
        
        # 在后台启动服务器
        server_task = asyncio.create_task(
            engine.start_api_server(host="0.0.0.0", port=preferred_port)
        )
        
        # 等待服务器基本启动
        print("⏳ 等待服务器启动...")
        server_started = await simple_startup_check()
        
        # 打开浏览器
        print("🌍 正在打开浏览器...")
        try:
            webbrowser.open(f'http://localhost:{preferred_port}')
            print("✅ 浏览器已打开")
        except Exception as e:
            print(f"⚠️  无法自动打开浏览器: {e}")
            print(f"请手动访问: http://localhost:{preferred_port}")
        
        # 等待服务器任务完成
        await server_task
        
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

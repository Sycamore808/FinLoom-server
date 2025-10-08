#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinLoom系统诊断工具
快速检测系统所有功能是否正常
"""

import sys
import os
import json
import time
from datetime import datetime

# Windows UTF-8编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def print_header(title):
    """打印标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_step(step_num, title):
    """打印步骤"""
    print(f"\n[步骤 {step_num}] {title}")
    print("-" * 70)

def print_result(success, message):
    """打印结果"""
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    return success

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    required = (3, 8)
    success = version >= required
    
    if success:
        print_result(True, f"Python版本: {version.major}.{version.minor}.{version.micro} (满足要求)")
    else:
        print_result(False, f"Python版本: {version.major}.{version.minor}.{version.micro} (需要3.8+)")
    
    return success

def check_dependencies():
    """检查关键依赖"""
    dependencies = {
        'fastapi': 'FastAPI Web框架',
        'uvicorn': 'ASGI服务器',
        'torch': 'PyTorch深度学习',
        'transformers': 'Hugging Face Transformers',
        'pandas': '数据处理',
        'numpy': '数值计算',
        'akshare': 'A股数据获取',
        'psutil': '系统信息',
        'yaml': 'YAML配置'
    }
    
    results = []
    for module, name in dependencies.items():
        try:
            if module == 'yaml':
                __import__('pyyaml')
            else:
                __import__(module)
            print_result(True, f"{name} ({module})")
            results.append(True)
        except ImportError:
            print_result(False, f"{name} ({module}) - 未安装")
            results.append(False)
    
    return all(results)

def check_project_structure():
    """检查项目结构"""
    critical_dirs = [
        'module_00_environment',
        'module_01_data_pipeline',
        'module_10_ai_interaction',
        'web',
        'config',
        'logs'
    ]
    
    critical_files = [
        'main.py',
        'requirements.txt',
        'config/system_config.yaml',
        'web/index_upgraded.html',
        'web/js/model-manager.js',
        'web/pages/model-manager.html'
    ]
    
    results = []
    
    # 检查目录
    for dir_name in critical_dirs:
        if os.path.exists(dir_name):
            print_result(True, f"目录: {dir_name}")
            results.append(True)
        else:
            print_result(False, f"目录: {dir_name} - 不存在")
            results.append(False)
    
    # 检查文件
    for file_name in critical_files:
        if os.path.exists(file_name):
            print_result(True, f"文件: {file_name}")
            results.append(True)
        else:
            print_result(False, f"文件: {file_name} - 不存在")
            results.append(False)
    
    return all(results)

def check_fin_r1_model():
    """检查FIN-R1模型"""
    try:
        from module_00_environment.model_manager import ModelManager
        
        print("⏳ 初始化模型管理器...")
        manager = ModelManager()
        
        print("⏳ 检查模型状态...")
        status = manager.get_model_status()
        
        print(f"\n模型配置状态:")
        print(f"  - 是否配置: {'是' if status.get('configured') else '否'}")
        print(f"  - 模型路径: {status.get('path', '未配置')}")
        print(f"  - 模型存在: {'是' if status.get('exists') else '否'}")
        if status.get('exists'):
            print(f"  - 模型大小: {status.get('size_mb', 0):.2f} MB")
        
        if status.get('configured') and status.get('exists'):
            print_result(True, "FIN-R1模型已配置且可用")
            return True
        elif status.get('configured') and not status.get('exists'):
            print_result(False, "FIN-R1模型已配置但文件不存在")
            print("💡 建议: 重新下载模型或检查路径")
            return False
        else:
            print_result(False, "FIN-R1模型未配置")
            print("💡 建议: 访问 http://localhost:8000/web/pages/model-manager.html 配置模型")
            return False
            
    except Exception as e:
        print_result(False, f"检查模型失败: {e}")
        return False

def check_system_requirements():
    """检查系统配置"""
    try:
        from module_00_environment.model_manager import ModelManager
        
        print("⏳ 检测系统配置...")
        manager = ModelManager()
        requirements = manager.check_system_requirements()
        
        system_info = requirements.get('system_info', {})
        print(f"\n系统信息:")
        print(f"  - CPU核心数: {system_info.get('cpu_count', 0)}")
        print(f"  - CPU频率: {system_info.get('cpu_freq_mhz', 0):.0f} MHz")
        print(f"  - 内存大小: {system_info.get('memory_gb', 0):.1f} GB")
        print(f"  - 可用内存: {system_info.get('memory_available_gb', 0):.1f} GB")
        print(f"  - 磁盘空间: {system_info.get('disk_free_gb', 0):.1f} GB")
        print(f"  - GPU可用: {'是' if system_info.get('gpu_available') else '否'}")
        print(f"  - CUDA可用: {'是' if system_info.get('cuda_available') else '否'}")
        print(f"  - Python版本: {system_info.get('python_version', 'Unknown')}")
        
        meets = requirements.get('meets_requirements', False)
        issues = requirements.get('issues', [])
        
        if meets:
            print_result(True, "系统配置满足要求")
        else:
            print_result(False, "系统配置不满足要求")
            
        if issues:
            print("\n⚠️ 配置问题:")
            for issue in issues:
                print(f"  - {issue}")
        
        return meets
        
    except Exception as e:
        print_result(False, f"检查系统配置失败: {e}")
        return False

def check_server_running():
    """检查服务器是否运行"""
    try:
        import requests
        
        print("⏳ 尝试连接服务器...")
        response = requests.get('http://localhost:8000/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n服务器信息:")
            print(f"  - 状态: {data.get('status', 'unknown')}")
            print(f"  - 版本: {data.get('version', 'unknown')}")
            print(f"  - 时间: {data.get('timestamp', 'unknown')}")
            
            print_result(True, "服务器正在运行")
            return True
        else:
            print_result(False, f"服务器响应异常: {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, "服务器未运行或无法连接")
        print("💡 建议: 运行 python main.py 启动服务器")
        return False

def check_model_api():
    """检查模型管理API"""
    try:
        import requests
        
        print("⏳ 测试模型管理API...")
        
        # 测试模型状态API
        response = requests.get('http://localhost:8000/api/v1/model/status', timeout=5)
        if response.status_code == 200:
            print_result(True, "模型状态API正常")
        else:
            print_result(False, f"模型状态API异常: {response.status_code}")
            return False
        
        # 测试系统配置API
        response = requests.get('http://localhost:8000/api/v1/model/system-requirements', timeout=5)
        if response.status_code == 200:
            print_result(True, "系统配置API正常")
        else:
            print_result(False, f"系统配置API异常: {response.status_code}")
            return False
        
        # 测试磁盘列表API
        response = requests.get('http://localhost:8000/api/v1/model/available-disks', timeout=5)
        if response.status_code == 200:
            print_result(True, "磁盘列表API正常")
        else:
            print_result(False, f"磁盘列表API异常: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print_result(False, f"API测试失败: {e}")
        return False

def check_chat_api():
    """检查对话API"""
    try:
        import requests
        
        print("⏳ 测试对话API...")
        
        response = requests.post(
            'http://localhost:8000/api/chat',
            json={'message': '你好', 'conversation_id': 'test'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print_result(True, "对话API正常（FIN-R1已启用）")
                print(f"  回复: {data.get('response', '')[:100]}...")
                return True
            else:
                print_result(True, "对话API可用但使用降级模式（FIN-R1未启用）")
                return True
        else:
            print_result(False, f"对话API异常: {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"对话API测试失败: {e}")
        return False

def check_data_api():
    """检查数据API"""
    try:
        import requests
        
        print("⏳ 测试数据API...")
        
        response = requests.get('http://localhost:8000/api/v1/data/overview', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n数据概览:")
            print(f"  - 股票数量: {data.get('total_symbols', 0)}")
            print(f"  - 记录总数: {data.get('total_records', 0)}")
            print(f"  - 最后更新: {data.get('last_update', '未知')}")
            
            print_result(True, "数据API正常")
            return True
        else:
            print_result(False, f"数据API异常: {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"数据API测试失败: {e}")
        return False

def generate_report(results):
    """生成诊断报告"""
    print_header("诊断报告")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"总计检查项: {total}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"通过率: {passed/total*100:.1f}%\n")
    
    print("详细结果:")
    for name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} - {name}")
    
    print("\n" + "="*70)
    
    if failed == 0:
        print("🎉 恭喜！所有检查项都通过了！")
        print("✅ 您的系统已准备就绪，可以开始使用FinLoom。")
    elif passed == 0:
        print("❌ 严重错误：所有检查项都失败了！")
        print("💡 建议：")
        print("  1. 确保在正确的目录中运行此脚本")
        print("  2. 运行 pip install -r requirements.txt 安装依赖")
        print("  3. 运行 python main.py 启动服务器")
    else:
        print("⚠️ 部分检查项失败，请查看上面的详细结果。")
        print("💡 建议：")
        
        if not results.get('服务器运行检查'):
            print("  1. 运行 python main.py 启动服务器")
        
        if not results.get('FIN-R1模型检查'):
            print("  2. 访问 http://localhost:8000/web/pages/model-manager.html 配置FIN-R1模型")
        
        if not results.get('Python版本检查') or not results.get('依赖包检查'):
            print("  3. 运行 pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
    
    print("\n详细文档: 快速启动指南.md")
    print("="*70)

def main():
    """主函数"""
    print_header("FinLoom 系统诊断工具")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    
    results = {}
    
    # 步骤1: 检查Python版本
    print_step(1, "Python版本检查")
    results['Python版本检查'] = check_python_version()
    
    # 步骤2: 检查依赖包
    print_step(2, "依赖包检查")
    results['依赖包检查'] = check_dependencies()
    
    # 步骤3: 检查项目结构
    print_step(3, "项目结构检查")
    results['项目结构检查'] = check_project_structure()
    
    # 步骤4: 检查系统配置
    print_step(4, "系统配置检查")
    results['系统配置检查'] = check_system_requirements()
    
    # 步骤5: 检查FIN-R1模型
    print_step(5, "FIN-R1模型检查")
    results['FIN-R1模型检查'] = check_fin_r1_model()
    
    # 步骤6: 检查服务器
    print_step(6, "服务器运行检查")
    server_running = check_server_running()
    results['服务器运行检查'] = server_running
    
    # 如果服务器运行，进行API测试
    if server_running:
        print_step(7, "模型管理API检查")
        results['模型管理API检查'] = check_model_api()
        
        print_step(8, "对话API检查")
        results['对话API检查'] = check_chat_api()
        
        print_step(9, "数据API检查")
        results['数据API检查'] = check_data_api()
    else:
        print("\n⚠️ 服务器未运行，跳过API测试")
        print("💡 运行 python main.py 启动服务器后重新测试")
    
    # 生成报告
    generate_report(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断测试")
    except Exception as e:
        print(f"\n\n❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()








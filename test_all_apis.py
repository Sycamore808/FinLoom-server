#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试脚本 - 验证所有功能模块与后端API的连接
"""

import requests
import json
import sys
import os
from datetime import datetime

# Windows控制台UTF-8编码设置
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 服务器地址
BASE_URL = "http://localhost:8000"

# 测试结果
test_results = {
    "passed": 0,
    "failed": 0,
    "tests": []
}

def test_api(name, method, endpoint, data=None, expected_status=200):
    """测试单个API端点"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"端点: {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            print(f"❌ 不支持的HTTP方法: {method}")
            test_results["failed"] += 1
            test_results["tests"].append({"name": name, "status": "FAILED", "error": "Unsupported method"})
            return
        
        # 检查状态码
        if response.status_code == expected_status:
            print(f"[PASS] 状态码: {response.status_code}")
            
            # 尝试解析JSON
            try:
                result = response.json()
                print(f"[PASS] 响应格式: JSON")
                
                # 显示部分响应内容
                if isinstance(result, dict):
                    if 'status' in result:
                        print(f"   状态: {result['status']}")
                    if 'message' in result:
                        print(f"   消息: {result['message']}")
                    if 'data' in result:
                        print(f"   数据: {type(result['data']).__name__}")
                
                print(f"[PASS] 测试通过")
                test_results["passed"] += 1
                test_results["tests"].append({"name": name, "status": "PASSED", "response_time": f"{response.elapsed.total_seconds():.2f}s"})
            except json.JSONDecodeError:
                print(f"[WARN] 响应不是JSON格式")
                print(f"   内容: {response.text[:200]}")
                test_results["passed"] += 1
                test_results["tests"].append({"name": name, "status": "PASSED", "note": "Non-JSON response"})
        else:
            print(f"[FAIL] 状态码错误: {response.status_code} (期望: {expected_status})")
            print(f"   响应: {response.text[:200]}")
            test_results["failed"] += 1
            test_results["tests"].append({"name": name, "status": "FAILED", "error": f"Status {response.status_code}"})
            
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] 连接失败: 无法连接到服务器")
        print(f"   请确保服务器正在运行: python main.py")
        test_results["failed"] += 1
        test_results["tests"].append({"name": name, "status": "FAILED", "error": "Connection refused"})
    except requests.exceptions.Timeout:
        print(f"[FAIL] 请求超时")
        test_results["failed"] += 1
        test_results["tests"].append({"name": name, "status": "FAILED", "error": "Timeout"})
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        test_results["failed"] += 1
        test_results["tests"].append({"name": name, "status": "FAILED", "error": str(e)})

def main():
    """运行所有API测试"""
    print("="*60)
    print("FinLoom API 测试套件")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"服务器: {BASE_URL}")
    print("="*60)
    
    # 1. 健康检查
    test_api("健康检查", "GET", "/health")
    
    # 2. API根路径
    test_api("API根路径", "GET", "/api")
    
    # 3. 就绪检查
    test_api("就绪检查", "GET", "/api/v1/ready")
    
    # 4. 智能对话API（简化版）
    test_api(
        "智能对话API",
        "POST",
        "/api/chat",
        {
            "message": "请帮我分析一下平安银行这只股票",
            "conversation_id": "test_001"
        }
    )
    
    # 5. FIN-R1智能分析API
    test_api(
        "FIN-R1智能分析",
        "POST",
        "/api/v1/ai/chat",
        {
            "text": "请帮我制定一个稳健型投资策略，初始资金100万",
            "amount": 1000000,
            "risk_tolerance": "moderate"
        }
    )
    
    # 6. 数据概览API
    test_api("数据概览", "GET", "/api/v1/data/overview")
    
    # 7. 仪表板指标API
    test_api("仪表板指标", "GET", "/api/v1/dashboard/metrics")
    
    # 8. 投资组合持仓API
    test_api("投资组合持仓", "GET", "/api/v1/portfolio/positions")
    
    # 9. 最近交易API
    test_api("最近交易", "GET", "/api/v1/trades/recent")
    
    # 10. 市场概览API
    test_api("市场概览", "GET", "/api/v1/market/overview")
    
    # 11. 回测API
    test_api(
        "策略回测",
        "POST",
        "/api/v1/backtest/run",
        {
            "strategy": "sma",
            "symbol": "000001",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 1000000
        }
    )
    
    # 12. 模型状态API
    test_api("模型状态", "GET", "/api/v1/model/status")
    
    # 13. 投资组合概览API
    test_api("投资组合概览", "GET", "/api/v1/portfolio/overview")
    
    # 14. 报告列表API
    test_api("报告列表", "GET", "/api/v1/reports/list")
    
    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"[PASS] 通过: {test_results['passed']}")
    print(f"[FAIL] 失败: {test_results['failed']}")
    print(f"[INFO] 总计: {test_results['passed'] + test_results['failed']}")
    if test_results['passed'] + test_results['failed'] > 0:
        print(f"[INFO] 通过率: {test_results['passed'] / (test_results['passed'] + test_results['failed']) * 100:.1f}%")
    
    # 详细结果
    print("\n详细结果:")
    for i, test in enumerate(test_results["tests"], 1):
        status_icon = "[PASS]" if test["status"] == "PASSED" else "[FAIL]"
        print(f"{i}. {status_icon} {test['name']}")
        if test["status"] == "FAILED":
            print(f"   错误: {test.get('error', 'Unknown')}")
        elif "response_time" in test:
            print(f"   响应时间: {test['response_time']}")
    
    print("\n" + "="*60)
    
    # 返回退出码
    if test_results["failed"] > 0:
        print("[WARN] 部分测试失败，请检查服务器日志")
        sys.exit(1)
    else:
        print("[SUCCESS] 所有测试通过！")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)


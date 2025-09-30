#!/usr/bin/env python3
"""
FinLoom API使用示例
演示如何使用FinLoom系统的API进行投资需求分析
"""

import requests
import json
from typing import Dict, Any

class FinLoomAPIClient:
    """FinLoom API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """初始化API客户端
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            健康状态信息
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_investment_requirement(self, text: str) -> Dict[str, Any]:
        """分析投资需求
        
        Args:
            text: 用户输入的投资需求文本
            
        Returns:
            分析结果
        """
        try:
            payload = {"text": text}
            response = requests.post(
                f"{self.base_url}/api/v1/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """主函数 - 演示API使用"""
    print("FinLoom API 使用示例")
    print("=" * 50)
    
    # 创建API客户端
    client = FinLoomAPIClient()
    
    # 1. 健康检查
    print("1. 检查系统健康状态...")
    health = client.health_check()
    if "error" not in health:
        print(f"   系统状态: {health['status']}")
        print(f"   健康分数: {health['health_score']}")
    else:
        print(f"   健康检查失败: {health['error']}")
        return
    
    print()
    
    # 2. 投资需求分析示例
    test_cases = [
        "我想找一些高成长性的中小盘股票，风险承受能力中等，投资期限1-3年",
        "我需要一个保守的投资组合，主要投资大盘蓝筹股，风险要低",
        "我想投资科技股，追求高收益，可以承受较高风险",
        "我需要一个平衡的投资组合，包含股票和债券，投资期限5年"
    ]
    
    for i, requirement in enumerate(test_cases, 1):
        print(f"{i}. 分析投资需求: {requirement}")
        result = client.analyze_investment_requirement(requirement)
        
        if "error" not in result:
            # 显示解析结果
            parsed = result.get("parsed_requirement", {})
            print(f"   投资期限: {parsed.get('investment_horizon', '未指定')}")
            print(f"   风险偏好: {parsed.get('risk_tolerance', '未指定')}")
            print(f"   投资目标: {[goal['goal_type'] for goal in parsed.get('investment_goals', [])]}")
            
            # 显示策略参数
            strategy = result.get("strategy_params", {})
            print(f"   策略组合: {strategy.get('strategy_mix', {})}")
            print(f"   调仓频率: {strategy.get('rebalance_frequency', '未指定')}")
            
            # 显示风险参数
            risk = result.get("risk_params", {})
            print(f"   最大回撤: {risk.get('max_drawdown', 0):.1%}")
            print(f"   仓位限制: {risk.get('position_limit', 0):.1%}")
            
        else:
            print(f"   分析失败: {result['error']}")
        
        print()

if __name__ == "__main__":
    main()

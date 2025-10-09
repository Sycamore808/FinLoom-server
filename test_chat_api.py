#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试对话API修复
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from module_10_ai_interaction.fin_r1_integration import FINR1Integration


async def test_fin_r1_integration():
    """测试FIN-R1集成"""
    print("=" * 60)
    print("测试FIN-R1集成配置加载")
    print("=" * 60)
    
    try:
        # 测试1: 使用配置文件初始化
        print("\n[测试1] 使用配置文件初始化...")
        config_path = "module_10_ai_interaction/config/fin_r1_config.yaml"
        fin_r1 = FINR1Integration(config_path=config_path)
        print(f"✅ 配置加载成功")
        print(f"  - 模型路径: {fin_r1.model_path}")
        print(f"  - 设备: {fin_r1.device}")
        print(f"  - 温度: {fin_r1.temperature}")
        
        # 测试2: 处理简单请求
        print("\n[测试2] 处理简单请求...")
        user_input = "我想稳健投资，有10万资金"
        print(f"  用户输入: {user_input}")
        
        result = await fin_r1.process_request(user_input)
        print(f"✅ 请求处理成功")
        print(f"  - 解析的需求: {result.get('parsed_requirement', {}).get('investment_amount')}")
        print(f"  - 风险偏好: {result.get('parsed_requirement', {}).get('risk_tolerance')}")
        print(f"  - 策略推荐: {result.get('model_output', {}).get('strategy_recommendation')}")
        
        # 测试3: 测试返回值不是None
        print("\n[测试3] 验证返回值...")
        assert result is not None, "结果不应该是None"
        assert isinstance(result, dict), "结果应该是字典"
        assert 'parsed_requirement' in result, "结果应该包含parsed_requirement"
        assert 'model_output' in result, "结果应该包含model_output"
        print("✅ 返回值验证通过")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_api_call_simulation():
    """模拟API调用"""
    print("\n" + "=" * 60)
    print("模拟API调用")
    print("=" * 60)
    
    try:
        # 模拟 fin_r1_chat 函数的逻辑
        request = {
            "text": "请分析当前A股市场的整体走势和投资机会",
            "amount": None,
            "risk_tolerance": None
        }
        
        text = request.get("text", "")
        amount = request.get("amount")
        risk_tolerance = request.get("risk_tolerance")
        
        print(f"\n请求参数:")
        print(f"  - text: {text}")
        print(f"  - amount: {amount}")
        print(f"  - risk_tolerance: {risk_tolerance}")
        
        # 加载配置
        from pathlib import Path
        import yaml
        
        config_path = Path("module_10_ai_interaction") / "config" / "fin_r1_config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                fin_r1_config = yaml.safe_load(f)
                # 确保不是None
                if fin_r1_config is None:
                    fin_r1_config = {}
        else:
            fin_r1_config = {
                "model_path": "models/fin_r1",
                "device": "cpu",
                "temperature": 0.7,
            }
        
        print(f"\n配置加载:")
        print(f"  - 配置存在: {config_path.exists()}")
        print(f"  - 配置内容: {fin_r1_config is not None and len(fin_r1_config) > 0}")
        
        # 创建FIN-R1实例
        fin_r1 = FINR1Integration(fin_r1_config)
        
        # 处理请求
        full_request = text
        if amount:
            full_request += f"\n投资金额: {amount}元"
        if risk_tolerance:
            risk_map = {
                "conservative": "保守型",
                "moderate": "稳健型",
                "aggressive": "激进型",
                "very_aggressive": "非常激进型",
            }
            full_request += f"\n风险偏好: {risk_map.get(risk_tolerance, risk_tolerance)}"
        
        print(f"\n完整请求: {full_request[:100]}...")
        
        parsed_result = await fin_r1.process_request(full_request)
        
        print(f"\n✅ API调用模拟成功")
        print(f"  - parsed_result是None: {parsed_result is None}")
        print(f"  - parsed_result类型: {type(parsed_result)}")
        
        if parsed_result is not None:
            print(f"  - 包含parsed_requirement: {'parsed_requirement' in parsed_result}")
            print(f"  - 包含strategy_params: {'strategy_params' in parsed_result}")
            print(f"  - 包含risk_params: {'risk_params' in parsed_result}")
            
            # 测试.get()调用
            parsed_req = parsed_result.get("parsed_requirement", {})
            strategy_params = parsed_result.get("strategy_params", {})
            risk_params = parsed_result.get("risk_params", {})
            print(f"\n✅ .get()调用成功，不会抛出AttributeError")
        
        print("\n" + "=" * 60)
        print("🎉 API调用模拟测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ API调用模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("\n开始测试对话API修复...\n")
    
    # 运行测试
    test1 = await test_fin_r1_integration()
    test2 = await test_api_call_simulation()
    
    if test1 and test2:
        print("\n✅ 所有测试通过！对话API修复成功。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


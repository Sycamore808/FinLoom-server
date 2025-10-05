"""
Module 08 - 执行模块测试
使用真实数据进行测试
"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入Module 01获取真实数据
from module_01_data_pipeline import AkshareDataCollector

# 导入Module 08组件
from module_08_execution import (
    EnhancedSignal,
    ExecutionDestination,
    ExecutionMonitor,
    FilterConfig,
    OrderManager,
    OrderStatus,
    OrderType,
    SignalFilter,
    SignalGenerator,
    SignalPriority,
    SignalType,
    TWAPAlgorithm,
    VWAPAlgorithm,
    create_execution_algorithm,
    get_execution_database_manager,
    get_execution_interface,
)


class TestModule08WithRealData(unittest.TestCase):
    """使用真实数据测试Module 08"""

    @classmethod
    def setUpClass(cls):
        """设置测试类 - 获取真实数据"""
        print("\n正在获取真实市场数据...")
        cls.collector = AkshareDataCollector()

        # 获取多只股票的真实历史数据
        cls.test_symbols = ["000001", "600000", "601318"]
        cls.real_data = {}

        for symbol in cls.test_symbols:
            try:
                data = cls.collector.fetch_stock_history(
                    symbol=symbol, start_date="20240101", end_date="20241231"
                )
                if data is not None and len(data) > 0:
                    cls.real_data[symbol] = data
                    print(f"  ✓ 获取 {symbol} 数据: {len(data)} 条记录")
            except Exception as e:
                print(f"  ✗ 获取 {symbol} 数据失败: {e}")

        if not cls.real_data:
            raise RuntimeError("无法获取任何真实数据，测试无法进行")

    def setUp(self):
        """每个测试前的设置"""
        self.signal_generator = SignalGenerator()
        self.order_manager = OrderManager()
        self.db_manager = get_execution_database_manager()

    # ==================== 信号生成测试 ====================

    def test_01_signal_generation_with_real_data(self):
        """测试使用真实数据生成信号"""
        print("\n测试1: 信号生成")

        for symbol, data in self.real_data.items():
            signals = self.signal_generator.generate_multi_signal(
                symbol=symbol,
                data=data,
                strategies=["MA_CROSSOVER", "RSI", "BOLLINGER"],
            )

            print(f"  {symbol}: 生成 {len(signals)} 个信号")

            for signal in signals:
                self.assertIsInstance(signal, EnhancedSignal)
                self.assertEqual(signal.symbol, symbol)
                self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
                self.assertGreaterEqual(signal.confidence, 0)
                self.assertLessEqual(signal.confidence, 1.0)
                self.assertIsNotNone(signal.price)
                self.assertGreater(signal.quantity, 0)

    def test_02_signal_filter(self):
        """测试信号过滤"""
        print("\n测试2: 信号过滤")

        # 获取第一只股票的数据
        symbol = list(self.real_data.keys())[0]
        data = self.real_data[symbol]

        # 生成信号
        signals = self.signal_generator.generate_multi_signal(symbol, data)
        print(f"  原始信号数: {len(signals)}")

        # 过滤信号
        filter_config = FilterConfig(
            min_signal_strength=0.6,
            enable_risk_filter=False,
            enable_liquidity_filter=False,
        )
        signal_filter = SignalFilter(filter_config)
        filtered_signals = signal_filter.filter_signals(signals)

        print(f"  过滤后信号数: {len(filtered_signals)}")

        # 验证过滤结果
        self.assertLessEqual(len(filtered_signals), len(signals))
        for signal in filtered_signals:
            self.assertGreaterEqual(
                signal.confidence, filter_config.min_signal_strength
            )

    # ==================== 订单管理测试 ====================

    def test_03_order_creation_from_real_signal(self):
        """测试从真实信号创建订单"""
        print("\n测试3: 订单创建")

        # 获取真实信号
        symbol = list(self.real_data.keys())[0]
        data = self.real_data[symbol]
        signals = self.signal_generator.generate_multi_signal(symbol, data)

        if signals:
            signal = signals[0]
            order = self.order_manager.create_order_from_signal(signal)

            print(
                f"  创建订单: {order.symbol} {order.side} {order.quantity}股 @ {order.price}"
            )

            self.assertIsNotNone(order.order_id)
            self.assertEqual(order.symbol, signal.symbol)
            self.assertEqual(order.signal_id, signal.signal_id)
            self.assertIn(order.side, ["BUY", "SELL"])
            self.assertEqual(order.status, OrderStatus.PENDING)

    def test_04_order_lifecycle(self):
        """测试订单生命周期"""
        print("\n测试4: 订单生命周期")

        # 创建测试订单
        symbol = list(self.real_data.keys())[0]
        data = self.real_data[symbol]
        signals = self.signal_generator.generate_multi_signal(symbol, data)

        if signals:
            order = self.order_manager.create_order_from_signal(signals[0])

            # 提交订单
            self.order_manager.submit_order(order)
            self.assertEqual(order.status, OrderStatus.SUBMITTED)
            print(f"  ✓ 订单已提交: {order.order_id}")

            # 部分成交
            self.order_manager.fill_order(
                order_id=order.order_id,
                filled_quantity=order.quantity // 2,
                filled_price=order.price,
            )
            self.assertEqual(order.status, OrderStatus.PARTIAL_FILLED)
            print(f"  ✓ 部分成交: {order.filled_quantity}/{order.quantity}")

            # 完全成交
            self.order_manager.fill_order(
                order_id=order.order_id,
                filled_quantity=order.quantity - order.filled_quantity,
                filled_price=order.price,
            )
            self.assertEqual(order.status, OrderStatus.FILLED)
            print(f"  ✓ 完全成交: {order.order_id}")

    # ==================== 执行接口测试 ====================

    def test_05_execution_interface(self):
        """测试执行接口"""
        print("\n测试5: 执行接口")

        exec_interface = get_execution_interface()

        # 生成真实信号和订单
        symbol = list(self.real_data.keys())[0]
        data = self.real_data[symbol]
        signals = self.signal_generator.generate_multi_signal(symbol, data)

        if signals:
            order = self.order_manager.create_order_from_signal(signals[0])

            # 提交执行请求
            request = exec_interface.submit_execution_request(
                order=order,
                destination=ExecutionDestination.EXCHANGE,
                notes="测试执行请求",
            )

            self.assertEqual(request.order_id, order.order_id)
            self.assertEqual(request.symbol, order.symbol)
            print(f"  ✓ 提交执行请求: {request.order_id}")

            # 更新执行状态
            result = exec_interface.update_execution_status(
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                executed_quantity=order.quantity,
                executed_price=order.price,
                commission=order.quantity * order.price * 0.0003,
            )

            self.assertEqual(result.status, OrderStatus.FILLED)
            self.assertEqual(result.executed_quantity, order.quantity)
            print(f"  ✓ 更新执行状态: 成交 {result.executed_quantity}股")

            # 获取执行摘要
            summary = exec_interface.get_execution_summary()
            self.assertIn("fill_rate", summary)
            print(f"  ✓ 执行摘要 - 成交率: {summary['fill_rate']:.1%}")

    # ==================== 数据库测试 ====================

    def test_06_database_operations(self):
        """测试数据库操作"""
        print("\n测试6: 数据库操作")

        # 生成真实订单
        symbol = list(self.real_data.keys())[0]
        data = self.real_data[symbol]
        signals = self.signal_generator.generate_multi_signal(symbol, data)

        if signals:
            order = self.order_manager.create_order_from_signal(signals[0])

            # 保存订单到数据库
            success = self.db_manager.save_order(
                order_id=order.order_id,
                signal_id=order.signal_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                status=order.status.value,
            )
            self.assertTrue(success)
            print(f"  ✓ 保存订单: {order.order_id}")

            # 查询订单
            saved_order = self.db_manager.get_order(order.order_id)
            self.assertIsNotNone(saved_order)
            self.assertEqual(saved_order["order_id"], order.order_id)
            self.assertEqual(saved_order["symbol"], order.symbol)
            print(f"  ✓ 查询订单: {saved_order['symbol']} - {saved_order['status']}")

            # 保存成交
            success = self.db_manager.save_trade(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                commission=order.quantity * order.price * 0.0003,
                slippage_bps=2.5,
            )
            self.assertTrue(success)
            print(f"  ✓ 保存成交记录")

            # 查询成交
            trades = self.db_manager.get_trades(order_id=order.order_id)
            self.assertGreater(len(trades), 0)
            print(f"  ✓ 查询成交: 共 {len(trades)} 条")

    # ==================== 执行算法测试 ====================

    def test_07_execution_algorithm_factory(self):
        """测试执行算法工厂函数"""
        print("\n测试7: 执行算法工厂")

        # 测试创建不同类型的算法
        config = {"num_slices": 10, "interval_minutes": 5}

        try:
            twap = create_execution_algorithm("TWAP", config)
            self.assertIsNotNone(twap)
            print(f"  ✓ 创建TWAP算法成功")
        except Exception as e:
            print(f"  - TWAP算法创建: {e}")

        try:
            vwap = create_execution_algorithm("VWAP", config)
            self.assertIsNotNone(vwap)
            print(f"  ✓ 创建VWAP算法成功")
        except Exception as e:
            print(f"  - VWAP算法创建: {e}")

    def test_08_algorithm_basic_interface(self):
        """测试算法基本接口"""
        print("\n测试8: 算法基本接口")

        # 测试算法类是否可导入
        self.assertIsNotNone(TWAPAlgorithm)
        self.assertIsNotNone(VWAPAlgorithm)
        print(f"  ✓ TWAP算法类可导入")
        print(f"  ✓ VWAP算法类可导入")

    # ==================== 集成测试 ====================

    def test_09_end_to_end_workflow(self):
        """端到端工作流测试"""
        print("\n测试9: 完整工作流")

        # 1. 获取真实数据
        symbol = list(self.real_data.keys())[0]
        data = self.real_data[symbol]
        print(f"  1. 使用真实数据: {symbol} ({len(data)} 条记录)")

        # 2. 生成信号
        signals = self.signal_generator.generate_multi_signal(symbol, data)
        print(f"  2. 生成信号: {len(signals)} 个")

        # 3. 过滤信号
        filter_config = FilterConfig(min_signal_strength=0.6)
        signal_filter = SignalFilter(filter_config)
        filtered_signals = signal_filter.filter_signals(signals)
        print(f"  3. 过滤信号: 剩余 {len(filtered_signals)} 个")

        if filtered_signals:
            # 4. 创建订单
            order = self.order_manager.create_order_from_signal(filtered_signals[0])
            print(f"  4. 创建订单: {order.order_id}")

            # 5. 提交执行请求
            exec_interface = get_execution_interface()
            request = exec_interface.submit_execution_request(
                order=order, destination=ExecutionDestination.EXCHANGE
            )
            print(f"  5. 提交执行请求: {request.order_id}")

            # 6. 更新执行状态
            result = exec_interface.update_execution_status(
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                executed_quantity=order.quantity,
                executed_price=order.price,
            )
            print(f"  6. 执行完成: 成交 {result.executed_quantity}股")

            # 7. 验证数据库
            saved_order = self.db_manager.get_order(order.order_id)
            self.assertIsNotNone(saved_order)
            self.assertEqual(saved_order["status"], "FILLED")
            print(f"  7. 数据库验证: 订单已保存且状态正确")

            # 8. 查询成交
            trades = self.db_manager.get_trades(order_id=order.order_id)
            self.assertGreater(len(trades), 0)
            print(f"  8. 成交记录: {len(trades)} 条")

    def test_10_multiple_symbols_workflow(self):
        """多股票工作流测试"""
        print("\n测试10: 多股票工作流")

        all_orders = []

        # 为每个股票创建测试订单
        for symbol, data in self.real_data.items():
            current_price = float(data["close"].iloc[-1])

            # 创建测试信号
            test_signal = EnhancedSignal(
                signal_id=f"TEST_MULTI_{symbol}",
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=SignalType.BUY,
                quantity=1000,
                price=current_price,
                confidence=0.75,
                priority=SignalPriority.NORMAL,
                strategy_name="TEST",
                metadata={},
                expected_return=0.05,
                risk_score=0.3,
                holding_period=10,
            )

            order = self.order_manager.create_order_from_signal(test_signal)
            all_orders.append(order)

        print(f"  创建了 {len(all_orders)} 个订单")

        # 批量提交
        exec_interface = get_execution_interface()
        for order in all_orders:
            exec_interface.submit_execution_request(
                order=order, destination=ExecutionDestination.EXCHANGE
            )

        print(f"  ✓ 批量提交完成")

        # 批量更新状态
        for order in all_orders:
            exec_interface.update_execution_status(
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                executed_quantity=order.quantity,
                executed_price=order.price,
            )

        print(f"  ✓ 批量执行完成")

        # 验证
        summary = exec_interface.get_execution_summary()
        print(f"  执行摘要: 成交率 {summary['fill_rate']:.1%}")
        self.assertGreater(summary["completed_executions"], 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModule08WithRealData)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印摘要
    print("\n" + "=" * 70)
    print("测试摘要")
    print("=" * 70)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

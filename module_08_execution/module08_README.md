# Module 08 - 交易执行模块

## 概述

交易执行模块负责信号处理、订单管理、执行算法和交易监控。模块专注于交易执行逻辑，不涉及具体券商连接，所有数据持久化到SQLite数据库。

## 核心设计

### 设计原则
1. **专注执行逻辑**：生成待执行订单信息，不模拟券商连接
2. **数据持久化**：所有执行数据保存到SQLite数据库
3. **模块集成**：清晰的接口与其他模块协作
4. **易于扩展**：预留接口支持未来真实券商集成

### 核心功能
- **信号生成与过滤**：多策略信号生成、风险过滤
- **订单管理**：订单生命周期管理
- **执行算法**：TWAP、VWAP、IS等专业算法
- **执行监控**：实时质量监控、市场冲击估算
- **数据管理**：SQLite持久化存储

---

## 文件结构

```
module_08_execution/
├── __init__.py                      # 模块导出
├── signal_generator.py              # 信号生成
├── signal_filter.py                 # 信号过滤
├── order_manager.py                 # 订单管理
├── order_router.py                  # 订单路由
├── execution_algorithms.py          # 执行算法（TWAP/VWAP等）
├── execution_monitor.py             # 执行监控
├── market_impact_model.py          # 市场冲击模型
├── execution_interface.py          # 执行接口
├── transaction_logger.py           # 交易日志
├── database_manager.py             # 数据库管理
├── broker_connector.py             # 券商连接器（预留，未实现）
└── module08_README.md              # 本文档

tests/
└── module08_execution_test.py                # 完整测试（使用真实数据）
```

---

## API 文档

### 1. 信号生成 (SignalGenerator)

```python
from module_08_execution import SignalGenerator

# 初始化
signal_gen = SignalGenerator()

# 生成多策略信号
signals = signal_gen.generate_multi_signal(
    symbol="000001",
    data=price_data,  # pandas DataFrame with OHLCV
    strategies=["MA_CROSSOVER", "RSI", "BOLLINGER"]
)

# 生成单一策略信号
ma_signal = signal_gen.generate_ma_crossover_signal("000001", price_data)
rsi_signal = signal_gen.generate_rsi_signal("000001", price_data)
bb_signal = signal_gen.generate_bollinger_bands_signal("000001", price_data)
```

**返回值**: `List[EnhancedSignal]`
- `signal_id`: 信号ID
- `symbol`: 股票代码
- `signal_type`: SignalType.BUY / SignalType.SELL
- `confidence`: 置信度 (0-1)
- `price`: 建议价格
- `quantity`: 建议数量
- `strategy_name`: 策略名称

### 2. 信号过滤 (SignalFilter)

```python
from module_08_execution import SignalFilter, FilterConfig

# 配置过滤器
filter_config = FilterConfig(
    min_signal_strength=0.7,          # 最低信号强度
    enable_risk_filter=True,          # 启用风险过滤
    enable_liquidity_filter=True,     # 启用流动性过滤
    max_position_size_ratio=0.1       # 最大单一持仓比例
)

signal_filter = SignalFilter(filter_config)

# 过滤信号
filtered_signals = signal_filter.filter_signals(
    signals=raw_signals,
    risk_manager=risk_manager,        # 来自Module 05
    current_portfolio=portfolio_dict,
    market_data=market_data_dict
)
```

### 3. 订单管理 (OrderManager)

```python
from module_08_execution import OrderManager, OrderStatus

# 初始化
order_manager = OrderManager()

# 从信号创建订单
order = order_manager.create_order_from_signal(signal)

# 提交订单
order_manager.submit_order(order)

# 成交订单
order_manager.fill_order(
    order_id=order.order_id,
    filled_quantity=1000,
    filled_price=15.48
)

# 取消订单
order_manager.cancel_order(order.order_id)

# 获取订单状态
order = order_manager.get_order(order.order_id)
print(order.status)  # PENDING, SUBMITTED, FILLED, etc.
```

### 4. 执行接口 (ExecutionInterface)

**核心类 - 无需券商连接**

```python
from module_08_execution import get_execution_interface, ExecutionDestination, OrderStatus

# 获取全局实例
exec_interface = get_execution_interface()

# 提交执行请求（生成待执行订单信息）
request = exec_interface.submit_execution_request(
    order=order,
    destination=ExecutionDestination.EXCHANGE,
    notes="自动交易信号"
)

# 外部系统更新执行状态
result = exec_interface.update_execution_status(
    order_id=order.order_id,
    status=OrderStatus.FILLED,
    executed_quantity=1000,
    executed_price=15.48,
    commission=4.64
)

# 获取待执行请求
pending = exec_interface.get_all_pending_requests()
by_symbol = exec_interface.get_pending_requests_by_symbol("000001")

# 获取执行摘要
summary = exec_interface.get_execution_summary()
print(f"成交率: {summary['fill_rate']:.1%}")
print(f"平均滑点: {summary['avg_slippage_bps']:.2f} bps")

# 取消执行请求
exec_interface.cancel_execution_request(order_id, reason="市场波动过大")
```

### 5. 执行算法

#### TWAP (时间加权平均价格)

```python
from module_08_execution import TWAPAlgorithm
from module_08_execution.execution_algorithms import TWAPConfig

config = TWAPConfig(
    num_slices=10,           # 分10次执行
    interval_minutes=5       # 每5分钟执行一次
)

twap = TWAPAlgorithm(config)
plan = twap.create_execution_plan(order, market_data)

# 执行
for slice in plan.slices:
    print(f"执行切片: {slice.quantity}股 @ {slice.scheduled_time}")
```

#### VWAP (成交量加权平均价格)

```python
from module_08_execution import VWAPAlgorithm
from module_08_execution.execution_algorithms import VWAPConfig

config = VWAPConfig(
    volume_target_percentage=10,  # 目标成交量的10%
    num_slices=20
)

vwap = VWAPAlgorithm(config)
plan = vwap.create_execution_plan(order, market_data)
```

### 6. 执行监控 (ExecutionMonitor)

```python
from module_08_execution import create_execution_monitor

# 创建监控器
monitor = create_execution_monitor()

# 注册告警回调
def on_alert(alert):
    print(f"告警: {alert.alert_type} - {alert.message}")

monitor.register_alert_callback(on_alert)

# 计算执行指标
metrics = monitor.calculate_metrics(orders, trades)
print(f"成交率: {metrics.fill_rate:.1%}")
print(f"平均滑点: {metrics.avg_slippage_bps:.2f} bps")
```

### 7. 市场冲击模型

```python
from module_08_execution import create_impact_model

# 创建市场冲击模型
impact_model = create_impact_model(model_type="almgren_chriss")

# 估算冲击
estimate = impact_model.estimate_impact(
    symbol="000001",
    quantity=10000,
    side="BUY",
    market_data=market_data
)

print(f"临时冲击: {estimate.temporary_impact_bps:.2f} bps")
print(f"永久冲击: {estimate.permanent_impact_bps:.2f} bps")
```

### 8. 数据库管理 (ExecutionDatabaseManager)

```python
from module_08_execution import get_execution_database_manager

# 获取数据库管理器
db = get_execution_database_manager()

# 保存订单
db.save_order(
    order_id="ORD_001",
    signal_id="SIG_001",
    symbol="000001",
    side="BUY",
    order_type="LIMIT",
    quantity=1000,
    price=15.50,
    status="FILLED"
)

# 保存成交
db.save_trade(
    order_id="ORD_001",
    symbol="000001",
    side="BUY",
    quantity=1000,
    price=15.48,
    commission=4.64,
    slippage_bps=1.29
)

# 查询订单
order = db.get_order("ORD_001")
orders = db.get_orders(symbol="000001", status="FILLED")

# 查询成交
trades = db.get_trades(symbol="000001")

# 获取统计
stats = db.get_execution_statistics(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)
print(f"总订单数: {stats['total_orders']}")
print(f"平均成交率: {stats['avg_fill_rate']:.1%}")
```

---

## 完整使用流程

```python
from datetime import datetime
from module_01_data_pipeline import AkshareDataCollector
from module_08_execution import (
    SignalGenerator,
    SignalFilter,
    FilterConfig,
    OrderManager,
    get_execution_interface,
    ExecutionDestination,
    OrderStatus,
    get_execution_database_manager
)

# ===== 1. 获取市场数据 (Module 01) =====
collector = AkshareDataCollector()
data = collector.fetch_stock_history("000001", "20240101", "20241231")

# ===== 2. 生成交易信号 =====
signal_gen = SignalGenerator()
signals = signal_gen.generate_multi_signal("000001", data)

# ===== 3. 过滤信号 =====
filter_config = FilterConfig(min_signal_strength=0.7)
signal_filter = SignalFilter(filter_config)
filtered_signals = signal_filter.filter_signals(signals)

# ===== 4. 创建订单 =====
order_manager = OrderManager()
orders = [order_manager.create_order_from_signal(s) for s in filtered_signals]

# ===== 5. 提交执行请求 =====
exec_interface = get_execution_interface()
for order in orders:
    exec_interface.submit_execution_request(
        order=order,
        destination=ExecutionDestination.EXCHANGE
    )

# ===== 6. 外部系统更新执行状态 =====
# (由实际的交易系统或人工操作更新)
exec_interface.update_execution_status(
    order_id=orders[0].order_id,
    status=OrderStatus.FILLED,
    executed_quantity=1000,
    executed_price=15.48,
    commission=4.64
)

# ===== 7. 查询执行数据 =====
db = get_execution_database_manager()
order = db.get_order(orders[0].order_id)
trades = db.get_trades()

# ===== 8. 获取执行摘要 =====
summary = exec_interface.get_execution_summary()
print(f"成交率: {summary['fill_rate']:.1%}")
print(f"平均滑点: {summary['avg_slippage_bps']:.2f} bps")
```

---

## 模块集成

### Module 08 提供的接口

Module 08 是一个**完整独立**的模块，对外提供以下功能：

#### 1. 信号生成与过滤
- `SignalGenerator`: 生成交易信号
- `SignalFilter`: 过滤信号
- `EnhancedSignal`: 增强信号数据结构

#### 2. 订单管理
- `OrderManager`: 订单生命周期管理
- `ExecutionInterface`: 简化的执行接口
- `Order`: 订单数据结构

#### 3. 执行算法
- `TWAPAlgorithm`, `VWAPAlgorithm`: 执行算法
- `ExecutionPlan`: 执行计划

#### 4. 监控与分析
- `ExecutionMonitor`: 执行质量监控
- `MarketImpactModel`: 市场冲击估算

#### 5. 数据持久化
- `ExecutionDatabaseManager`: SQLite数据库管理

### Module 08 需要的外部接口

#### 从 Module 01 (数据管道)
```python
# Module 01 提供
from module_01_data_pipeline import AkshareDataCollector

collector = AkshareDataCollector()
data = collector.fetch_stock_history(symbol, start_date, end_date)
```

#### 从 Module 05 (风险管理) - 可选

如果要启用风险过滤，Module 05 需要提供：

```python
class RiskLimitManager:
    """Module 05 应该提供的风险管理接口"""
    
    def check_signal(self, signal: Signal) -> bool:
        """
        检查信号是否符合风险限制
        
        Args:
            signal: 交易信号
            
        Returns:
            bool: True=通过，False=拒绝
        """
        pass
    
    def validate_order(self, order: Order, current_portfolio: Dict, 
                      account_balance: float) -> Dict[str, Any]:
        """
        验证订单是否符合风险限制
        
        Args:
            order: 订单对象
            current_portfolio: 当前投资组合
            account_balance: 账户余额
            
        Returns:
            Dict: {
                'approved': bool,  # 是否批准
                'reason': str,     # 拒绝原因
                'adjusted_quantity': int  # 调整后的数量
            }
        """
        pass
```

**使用方式**:
```python
from module_05_risk_management import RiskLimitManager
from module_08_execution import SignalFilter, FilterConfig

risk_manager = RiskLimitManager()
filter_config = FilterConfig(enable_risk_filter=True)
signal_filter = SignalFilter(filter_config)

filtered = signal_filter.filter_signals(
    signals=signals,
    risk_manager=risk_manager,
    current_portfolio=portfolio
)
```

#### 从 Module 06 (监控告警) - 可选

如果要发送执行告警，Module 06 需要提供：

```python
class AlertManager:
    """Module 06 应该提供的告警管理接口"""
    
    def send_alert(self, level: str, title: str, message: str, 
                  channels: List[str]) -> bool:
        """
        发送告警消息
        
        Args:
            level: 告警级别 ('info', 'warning', 'error', 'critical')
            title: 告警标题
            message: 告警内容
            channels: 通知渠道 ['email', 'wechat', 'sms']
            
        Returns:
            bool: 是否发送成功
        """
        pass
```

**使用方式**:
```python
from module_06_monitoring_alerting import AlertManager
from module_08_execution import ExecutionMonitor

alert_manager = AlertManager()
execution_monitor = ExecutionMonitor()

def on_execution_alert(alert):
    alert_manager.send_alert(
        level=alert.severity,
        title=f"执行告警: {alert.alert_type}",
        message=alert.message,
        channels=['email']
    )

execution_monitor.register_alert_callback(on_execution_alert)
```

---

## 前端API集成

### RESTful API 端点

#### 1. 信号管理

**生成信号**
```http
POST /api/v1/execution/signals/generate
Content-Type: application/json

{
  "symbol": "000001",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "strategies": ["MA_CROSS", "RSI"]
}

Response:
{
  "success": true,
  "data": {
    "signals": [...],
    "total": 10
  }
}
```

**获取信号列表**
```http
GET /api/v1/execution/signals?symbol=000001&page=1&page_size=20

Response:
{
  "success": true,
  "data": {
    "signals": [...],
    "total": 100,
    "page": 1,
    "total_pages": 5
  }
}
```

#### 2. 订单管理

**创建订单**
```http
POST /api/v1/execution/orders
Content-Type: application/json

{
  "signal_id": "SIG_001",
  "order_type": "LIMIT",
  "price": 15.50,
  "quantity": 1000
}

Response:
{
  "success": true,
  "data": {
    "order": {
      "order_id": "ORD_001",
      "status": "PENDING",
      ...
    }
  }
}
```

**提交订单**
```http
POST /api/v1/execution/orders/{order_id}/submit

Response:
{
  "success": true,
  "data": {
    "order_id": "ORD_001",
    "status": "SUBMITTED"
  }
}
```

**获取订单详情**
```http
GET /api/v1/execution/orders/{order_id}

Response:
{
  "success": true,
  "data": {
    "order": {
      "order_id": "ORD_001",
      "symbol": "000001",
      "status": "FILLED",
      ...
    }
  }
}
```

#### 3. 执行监控

**获取执行指标**
```http
GET /api/v1/execution/metrics?start_date=2024-01-01&end_date=2024-12-31

Response:
{
  "success": true,
  "data": {
    "metrics": {
      "total_orders": 100,
      "filled_orders": 95,
      "fill_rate": 0.95,
      "avg_slippage_bps": 3.5,
      "total_commission": 1250.50
    }
  }
}
```

**获取成交记录**
```http
GET /api/v1/execution/trades?symbol=000001&page=1&page_size=20

Response:
{
  "success": true,
  "data": {
    "trades": [...],
    "total": 50,
    "summary": {
      "total_quantity": 5000,
      "avg_price": 15.47
    }
  }
}
```

### WebSocket 实时推送

```javascript
const ws = new WebSocket('ws://api.example.com/api/v1/execution/ws');

// 订阅实时更新
ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    channels: ['orders', 'trades']
  }));
};

// 订单状态更新
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.channel === 'orders' && data.event === 'order_update') {
    console.log('订单更新:', data.data);
    // { order_id, status, filled_quantity, ... }
  }
  
  if (data.channel === 'trades' && data.event === 'new_trade') {
    console.log('新成交:', data.data);
    // { trade_id, symbol, quantity, price, ... }
  }
};
```

---

## 数据库设计

### SQLite 表结构

#### 订单表 (orders)
```sql
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    signal_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL,
    status TEXT NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    filled_price REAL,
    created_at TIMESTAMP,
    submitted_time TIMESTAMP,
    filled_time TIMESTAMP,
    metadata TEXT
)
```

#### 成交表 (trades)
```sql
CREATE TABLE trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    commission REAL DEFAULT 0.0,
    slippage REAL DEFAULT 0.0,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
)
```

#### 执行指标表 (execution_metrics)
```sql
CREATE TABLE execution_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE NOT NULL,
    total_orders INTEGER,
    filled_orders INTEGER,
    fill_rate REAL,
    avg_slippage REAL,
    total_commission REAL,
    market_impact REAL
)
```

---

## 测试

运行完整测试（使用真实数据）：
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
conda activate study
python -m pytest tests/test_module08.py -v
```

或直接运行：
```bash
python tests/test_module08.py
```

**测试说明**：
- 所有测试使用 Module 01 获取的真实市场数据
- 不使用任何模拟数据
- 测试涵盖信号生成、订单管理、执行接口、数据库等完整流程

---

## 配置说明

### SignalGenerator 配置
- 无需配置，使用默认参数

### SignalFilter 配置
```python
FilterConfig(
    min_signal_strength=0.6,          # 最低信号强度
    max_position_size_ratio=0.05,     # 最大单一持仓比例
    enable_risk_filter=True,          # 启用风险过滤
    enable_liquidity_filter=True,     # 启用流动性过滤
    min_volume_ratio=0.5,             # 最小成交量比率
    blacklist=[],                     # 黑名单股票
    max_daily_trades_per_symbol=5     # 单日单股票最大交易次数
)
```

### TWAP 算法配置
```python
TWAPConfig(
    num_slices=10,           # 切片数量
    interval_minutes=5,      # 执行间隔（分钟）
    start_time=None,         # 开始时间
    end_time=None,           # 结束时间
    urgency_factor=0.5       # 紧急程度 (0-1)
)
```

### VWAP 算法配置
```python
VWAPConfig(
    num_slices=20,                    # 切片数量
    volume_target_percentage=10,      # 目标成交量百分比
    participation_rate=0.1,           # 市场参与率
    min_slice_size=100               # 最小切片大小
)
```

---

## 注意事项

1. **数据来源**: 使用Module 01获取真实市场数据，不使用模拟数据
2. **数据持久化**: 所有执行数据自动保存到SQLite数据库
3. **风险管理**: 如需风险过滤，需要Module 05提供相应接口
4. **券商集成**: 当前不涉及真实券商连接，预留扩展接口
5. **测试环境**: 测试需在conda的`study`环境中运行


## 常见问题

**Q: 为什么不直接连接券商？**
A: 模块专注于执行逻辑和数据管理，实际交易由外部系统或人工操作完成，降低复杂度和风险。

**Q: 如何查看所有待执行订单？**
A: 使用 `exec_interface.get_all_pending_requests()` 或查询SQLite数据库。

**Q: 数据库文件在哪里？**
A: 默认位置是 `data/execution_data.db` 或 `data/module08_execution.db`。

**Q: 如何集成风险管理？**
A: 在 `SignalFilter` 中传入 `risk_manager` 参数，需要Module 05提供相应接口。

**Q: 是否支持回测？**
A: 支持，使用历史数据调用信号生成和执行算法即可。
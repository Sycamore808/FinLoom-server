"""
回测引擎模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from common.data_structures import MarketData, Signal, Position
from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("backtest_engine")

@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    benchmark_symbol: Optional[str] = None
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

@dataclass
class BacktestResult:
    """回测结果"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Dict[str, Any]] = field(default_factory=list)

class BacktestEngine:
    """回测引擎类"""
    
    def __init__(self, config: BacktestConfig):
        """初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_curve = []
        self.trades = []
        self.strategy_func: Optional[Callable] = None
        self.market_data: Dict[str, pd.DataFrame] = {}
        
    def load_market_data(self, symbols: List[str], data: Dict[str, pd.DataFrame]):
        """加载市场数据
        
        Args:
            symbols: 股票代码列表
            data: 市场数据字典
        """
        try:
            for symbol in symbols:
                if symbol in data:
                    df = data[symbol].copy()
                    # 确保数据按时间排序
                    df = df.sort_index()
                    # 过滤日期范围
                    df = df[(df.index >= self.config.start_date) & 
                           (df.index <= self.config.end_date)]
                    self.market_data[symbol] = df
                    logger.info(f"Loaded {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            raise QuantSystemError(f"Market data loading failed: {e}")
    
    def set_strategy(self, strategy_func: Callable):
        """设置策略函数
        
        Args:
            strategy_func: 策略函数
        """
        self.strategy_func = strategy_func
        logger.info("Strategy function set")
    
    def run(self) -> BacktestResult:
        """运行回测
        
        Returns:
            回测结果
        """
        try:
            if not self.strategy_func:
                raise QuantSystemError("Strategy function not set")
            
            if not self.market_data:
                raise QuantSystemError("Market data not loaded")
            
            logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
            
            # 生成交易日期
            trading_dates = self._generate_trading_dates()
            
            # 逐日回测
            for date in trading_dates:
                self._process_trading_day(date)
            
            # 计算回测结果
            result = self._calculate_results()
            
            logger.info(f"Backtest completed. Final capital: {result.final_capital:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise QuantSystemError(f"Backtest execution failed: {e}")
    
    def _generate_trading_dates(self) -> List[datetime]:
        """生成交易日期
        
        Returns:
            交易日期列表
        """
        dates = []
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            # 检查是否有市场数据
            has_data = False
            for symbol, data in self.market_data.items():
                if current_date in data.index:
                    has_data = True
                    break
            
            if has_data:
                dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        return dates
    
    def _process_trading_day(self, date: datetime):
        """处理单个交易日
        
        Args:
            date: 交易日期
        """
        try:
            # 获取当日市场数据
            current_data = {}
            for symbol, data in self.market_data.items():
                if date in data.index:
                    current_data[symbol] = data.loc[date]
            
            if not current_data:
                return
            
            # 更新持仓市值
            self._update_positions_value(current_data)
            
            # 计算当前总资产
            total_equity = self._calculate_total_equity()
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': self.current_capital
            })
            
            # 生成交易信号
            if self.strategy_func:
                signals = self.strategy_func(current_data, self.positions, self.current_capital)
                if signals:
                    self._execute_signals(signals, current_data)
            
        except Exception as e:
            logger.error(f"Error processing trading day {date}: {e}")
    
    def _update_positions_value(self, current_data: Dict[str, pd.Series]):
        """更新持仓市值
        
        Args:
            current_data: 当前市场数据
        """
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol]['close']
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)
                position.last_update = datetime.now()
    
    def _calculate_total_equity(self) -> float:
        """计算总资产
        
        Returns:
            总资产
        """
        total_equity = self.current_capital
        for position in self.positions.values():
            total_equity += position.market_value
        return total_equity
    
    def _execute_signals(self, signals: List[Signal], current_data: Dict[str, pd.Series]):
        """执行交易信号
        
        Args:
            signals: 交易信号列表
            current_data: 当前市场数据
        """
        for signal in signals:
            try:
                if signal.symbol not in current_data:
                    logger.warning(f"No data for {signal.symbol}, skipping signal")
                    continue
                
                current_price = current_data[signal.symbol]['close']
                
                if signal.action == "BUY":
                    self._execute_buy(signal, current_price)
                elif signal.action == "SELL":
                    self._execute_sell(signal, current_price)
                    
            except Exception as e:
                logger.error(f"Error executing signal {signal.signal_id}: {e}")
    
    def _execute_buy(self, signal: Signal, current_price: float):
        """执行买入操作
        
        Args:
            signal: 买入信号
            current_price: 当前价格
        """
        # 计算实际价格（考虑滑点）
        slippage = current_price * (self.config.slippage_bps / 10000)
        execution_price = current_price + slippage
        
        # 计算交易成本
        trade_value = signal.quantity * execution_price
        commission = trade_value * self.config.commission_rate
        total_cost = trade_value + commission
        
        # 检查资金是否足够
        if total_cost > self.current_capital:
            logger.warning(f"Insufficient capital for buy order: {signal.signal_id}")
            return
        
        # 更新资金
        self.current_capital -= total_cost
        
        # 更新持仓
        if signal.symbol in self.positions:
            position = self.positions[signal.symbol]
            # 计算新的平均成本
            total_quantity = position.quantity + signal.quantity
            total_cost_basis = (position.quantity * position.avg_cost) + trade_value
            position.avg_cost = total_cost_basis / total_quantity
            position.quantity = total_quantity
        else:
            # 新建持仓
            self.positions[signal.symbol] = Position(
                position_id=f"pos_{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=signal.symbol,
                quantity=signal.quantity,
                avg_cost=execution_price,
                current_price=execution_price,
                market_value=trade_value,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                open_time=datetime.now(),
                last_update=datetime.now()
            )
        
        # 记录交易
        self.trades.append({
            'date': datetime.now(),
            'symbol': signal.symbol,
            'action': 'BUY',
            'quantity': signal.quantity,
            'price': execution_price,
            'value': trade_value,
            'commission': commission,
            'signal_id': signal.signal_id
        })
        
        logger.info(f"Executed buy: {signal.quantity} {signal.symbol} at {execution_price:.2f}")
    
    def _execute_sell(self, signal: Signal, current_price: float):
        """执行卖出操作
        
        Args:
            signal: 卖出信号
            current_price: 当前价格
        """
        if signal.symbol not in self.positions:
            logger.warning(f"No position for {signal.symbol}, skipping sell signal")
            return
        
        position = self.positions[signal.symbol]
        
        # 检查持仓数量
        sell_quantity = min(signal.quantity, position.quantity)
        if sell_quantity <= 0:
            logger.warning(f"No quantity to sell for {signal.symbol}")
            return
        
        # 计算实际价格（考虑滑点）
        slippage = current_price * (self.config.slippage_bps / 10000)
        execution_price = current_price - slippage
        
        # 计算交易金额
        trade_value = sell_quantity * execution_price
        commission = trade_value * self.config.commission_rate
        net_proceeds = trade_value - commission
        
        # 更新资金
        self.current_capital += net_proceeds
        
        # 计算已实现盈亏
        realized_pnl = (execution_price - position.avg_cost) * sell_quantity
        position.realized_pnl += realized_pnl
        
        # 更新持仓
        position.quantity -= sell_quantity
        if position.quantity <= 0:
            del self.positions[signal.symbol]
        
        # 记录交易
        self.trades.append({
            'date': datetime.now(),
            'symbol': signal.symbol,
            'action': 'SELL',
            'quantity': sell_quantity,
            'price': execution_price,
            'value': trade_value,
            'commission': commission,
            'realized_pnl': realized_pnl,
            'signal_id': signal.signal_id
        })
        
        logger.info(f"Executed sell: {sell_quantity} {signal.symbol} at {execution_price:.2f}")
    
    def _calculate_results(self) -> BacktestResult:
        """计算回测结果
        
        Returns:
            回测结果
        """
        try:
            # 计算基本指标
            final_capital = self._calculate_total_equity()
            total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital
            
            # 计算年化收益率
            days = (self.config.end_date - self.config.start_date).days
            annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # 计算权益曲线
            equity_df = pd.DataFrame(self.equity_curve)
            if not equity_df.empty:
                equity_df.set_index('date', inplace=True)
                equity_returns = equity_df['equity'].pct_change().dropna()
                
                # 计算波动率
                volatility = equity_returns.std() * np.sqrt(252)
                
                # 计算夏普比率
                sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if equity_returns.std() > 0 else 0
                
                # 计算最大回撤
                cumulative_returns = (1 + equity_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
            
            # 计算交易统计
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('realized_pnl', 0) > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算盈亏比
            total_profit = sum([t.get('realized_pnl', 0) for t in self.trades if t.get('realized_pnl', 0) > 0])
            total_loss = abs(sum([t.get('realized_pnl', 0) for t in self.trades if t.get('realized_pnl', 0) < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # 性能指标
            performance_metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'final_capital': final_capital
            }
            
            return BacktestResult(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_capital=self.config.initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                performance_metrics=performance_metrics,
                equity_curve=equity_df,
                trades=self.trades
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate results: {e}")
            raise QuantSystemError(f"Results calculation failed: {e}")

# 便捷函数
def create_backtest_engine(config_dict: Dict[str, Any]) -> BacktestEngine:
    """创建回测引擎的便捷函数
    
    Args:
        config_dict: 配置字典
        
    Returns:
        回测引擎实例
    """
    config = BacktestConfig(**config_dict)
    return BacktestEngine(config)
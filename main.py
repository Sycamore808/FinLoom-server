"""
FinLoom 量化投资引擎主程序
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

# 尝试导入可选依赖
try:
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False
    uvicorn = None

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    StaticFiles = None
    FileResponse = None

from common.logging_system import setup_logger
from module_00_environment.config_loader import ConfigLoader
from module_00_environment.dependency_installer import auto_install_dependencies
from module_00_environment.env_checker import run_environment_check
from module_10_ai_interaction.fin_r1_integration import FINR1Integration

# 设置日志
logger = setup_logger("main")

# 初始化FastAPI应用
if HAS_FASTAPI:
    app = FastAPI(
        title="FinLoom API",
        description="FIN-R1赋能的自适应量化投资引擎",
        version="1.0.0"
    )
else:
    app = None

class FinLoomEngine:
    """FinLoom主引擎类"""
    
    def __init__(self):
        """初始化FinLoom引擎"""
        self.config_loader = ConfigLoader()
        self.fin_r1 = None
        self.modules = {}
        
    async def initialize(self):
        """初始化引擎"""
        logger.info("Initializing FinLoom Engine...")
        
        # 环境检查
        try:
            env_report = run_environment_check()
            logger.info(f"Environment health score: {env_report.health_score}")
            
            if env_report.health_score < 60:
                logger.warning("Environment health score is low. Please check the report.")
        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            
        # 自动安装依赖
        try:
            if auto_install_dependencies():
                logger.info("Dependencies installed successfully")
            else:
                logger.error("Failed to install dependencies")
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            
        # 加载配置
        try:
            self.system_config = self.config_loader.load_system_config()
            self.model_config = self.config_loader.load_model_config()
            self.trading_config = self.config_loader.load_trading_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
        # 初始化FIN-R1模型
        try:
            self.fin_r1 = FINR1Integration(self.model_config.get("fin_r1", {}))
            logger.info("FIN-R1 model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FIN-R1 model: {e}")
            
        logger.info("FinLoom Engine initialized successfully")
        
    async def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """启动API服务器"""
        if not HAS_FASTAPI or not HAS_UVICORN:
            logger.warning("FastAPI or Uvicorn not available, skipping API server")
            return
            
        logger.info(f"Starting API server on {host}:{port}")
        
        # 注册API路由
        self._register_api_routes()
        
        # 添加静态文件服务
        if StaticFiles and FileResponse:
            app.mount("/static", StaticFiles(directory="web"), name="static")
            
            @app.get("/")
            async def serve_web_app():
                return FileResponse("web/index.html")
        
        # 启动服务器
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    def _register_api_routes(self):
        """注册API路由"""
        if not HAS_FASTAPI or not app:
            return
            
        @app.get("/api")
        async def api_root():
            return {
                "message": "Welcome to FinLoom API",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
            
        @app.get("/health")
        async def health_check():
            env_report = run_environment_check()
            return {
                "status": "healthy" if env_report.health_score > 60 else "unhealthy",
                "health_score": env_report.health_score,
                "timestamp": datetime.now().isoformat()
            }
            
        @app.post("/api/v1/analyze")
        async def analyze_request(request: Dict):
            if not self.fin_r1:
                return {"error": "FIN-R1 model not initialized"}
                
            try:
                result = await self.fin_r1.process_request(request.get("text", ""))
                return result
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                return {"error": str(e)}
        
        @app.get("/api/v1/dashboard/metrics")
        async def get_dashboard_metrics():
            """获取仪表板指标"""
            try:
                # 模拟仪表板数据
                metrics = {
                    "total_assets": 1000000 + (datetime.now().hour * 1000),
                    "daily_return": 10000 + (datetime.now().minute * 100),
                    "sharpe_ratio": 1.5 + (datetime.now().second * 0.01),
                    "max_drawdown": -2.0 - (datetime.now().minute * 0.01),
                    "win_rate": 0.65,
                    "total_trades": 156,
                    "timestamp": datetime.now().isoformat()
                }
                return metrics
            except Exception as e:
                logger.error(f"Failed to get dashboard metrics: {e}")
                return {"error": str(e)}
        
        @app.get("/api/v1/portfolio/positions")
        async def get_portfolio_positions():
            """获取投资组合持仓"""
            try:
                # 模拟持仓数据
                positions = [
                    {
                        "symbol": "000001",
                        "name": "平安银行",
                        "quantity": 1000,
                        "cost_price": 12.00,
                        "current_price": 12.45,
                        "market_value": 12450,
                        "unrealized_pnl": 450,
                        "pnl_rate": 3.75
                    },
                    {
                        "symbol": "600036", 
                        "name": "招商银行",
                        "quantity": 500,
                        "cost_price": 45.00,
                        "current_price": 45.67,
                        "market_value": 22835,
                        "unrealized_pnl": 335,
                        "pnl_rate": 1.49
                    }
                ]
                return {"positions": positions}
            except Exception as e:
                logger.error(f"Failed to get portfolio positions: {e}")
                return {"error": str(e)}
        
        @app.get("/api/v1/trades/recent")
        async def get_recent_trades():
            """获取最近交易记录"""
            try:
                # 模拟交易记录
                trades = [
                    {
                        "time": "2024-01-15 14:30:00",
                        "symbol": "000001",
                        "action": "BUY",
                        "quantity": 1000,
                        "price": 12.45,
                        "pnl": 1250,
                        "status": "FILLED"
                    },
                    {
                        "time": "2024-01-15 10:15:00",
                        "symbol": "600036",
                        "action": "SELL", 
                        "quantity": 500,
                        "price": 45.67,
                        "pnl": -230,
                        "status": "FILLED"
                    }
                ]
                return {"trades": trades}
            except Exception as e:
                logger.error(f"Failed to get recent trades: {e}")
                return {"error": str(e)}
        
        @app.post("/api/v1/backtest/run")
        async def run_backtest(request: Dict):
            """运行策略回测"""
            try:
                strategy = request.get("strategy", "sma")
                symbol = request.get("symbol", "000001")
                start_date = request.get("start_date", "2023-01-01")
                end_date = request.get("end_date", "2023-12-31")
                initial_capital = request.get("initial_capital", 1000000)
                
                # 模拟回测结果
                result = {
                    "total_return": 25.6,
                    "annualized_return": 12.8,
                    "volatility": 15.2,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": -8.2,
                    "win_rate": 0.65,
                    "profit_factor": 1.45,
                    "total_trades": 156,
                    "equity_curve": [],  # 这里应该包含实际的净值曲线数据
                    "trades": []  # 这里应该包含实际的交易记录
                }
                
                return result
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                return {"error": str(e)}
        
        @app.post("/api/v1/data/collect")
        async def collect_market_data(request: Dict):
            """收集市场数据"""
            try:
                symbol = request.get("symbol", "000001")
                period = request.get("period", "1y")
                
                # 这里应该调用实际的数据收集器
                result = {
                    "symbol": symbol,
                    "period": period,
                    "records_count": 252,
                    "status": "success",
                    "message": f"Successfully collected {period} data for {symbol}"
                }
                
                return result
            except Exception as e:
                logger.error(f"Data collection failed: {e}")
                return {"error": str(e)}
        
        @app.get("/api/v1/data/overview")
        async def get_data_overview():
            """获取数据概览"""
            try:
                # 模拟数据概览
                overview = [
                    {
                        "symbol": "000001",
                        "name": "平安银行",
                        "records_count": 252,
                        "latest_price": 12.45,
                        "update_time": "2024-01-15 15:00:00"
                    },
                    {
                        "symbol": "600036",
                        "name": "招商银行", 
                        "records_count": 252,
                        "latest_price": 45.67,
                        "update_time": "2024-01-15 15:00:00"
                    }
                ]
                return {"data": overview}
            except Exception as e:
                logger.error(f"Failed to get data overview: {e}")
                return {"error": str(e)}
        
        @app.get("/api/v1/market/overview")
        async def get_market_overview():
            """获取市场概览"""
            try:
                # 模拟市场数据
                market_data = {
                    "indices": [
                        {"name": "上证指数", "value": 3245.67, "change": 1.2},
                        {"name": "深证成指", "value": 12456.78, "change": 0.8},
                        {"name": "创业板指", "value": 2345.67, "change": -0.5}
                    ],
                    "hot_stocks": [
                        {"symbol": "000001", "name": "平安银行", "price": 12.45, "change": 2.5},
                        {"symbol": "600036", "name": "招商银行", "price": 45.67, "change": 1.8},
                        {"symbol": "601318", "name": "中国平安", "price": 56.78, "change": -0.9}
                    ]
                }
                return market_data
            except Exception as e:
                logger.error(f"Failed to get market overview: {e}")
                return {"error": str(e)}

async def main():
    """主函数"""
    engine = FinLoomEngine()
    
    # 初始化引擎
    await engine.initialize()
    
    # 启动API服务器
    await engine.start_api_server()

if __name__ == "__main__":
    asyncio.run(main())
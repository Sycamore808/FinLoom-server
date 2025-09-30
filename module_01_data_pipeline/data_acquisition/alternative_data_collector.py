# -*- coding: utf-8 -*-
"""
中国市场另类数据采集器模块
专门针对中国股票市场，采集宏观经济数据
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# 尝试导入akshare，用于获取中国市场另类数据
try:
    import akshare as ak

    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("cn_alternative_data_collector")


class ChineseAlternativeDataCollector:
    """
    中国市场另类数据采集器
    专注于中国股票市场相关的另类数据源：
    - 宏观经济数据（GDP、CPI、PMI等）
    - 新闻联播数据
    - 板块行情数据
    """

    def __init__(self, rate_limit: float = 0.5):
        """
        初始化中国另类数据采集器

        Args:
            rate_limit: 请求间隔（秒）
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for real data collection. Install with: pip install akshare"
            )

    def _rate_limit_check(self):
        """检查并执行速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def fetch_macro_economic_data(
        self, indicator: str = "all"
    ) -> Dict[str, pd.DataFrame]:
        """
        获取中国宏观经济数据

        Args:
            indicator: 指标类型 ("GDP", "CPI", "PMI", "all")

        Returns:
            Dict[str, pd.DataFrame]: 宏观经济数据字典
        """
        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for macro data collection. Install with: pip install akshare"
            )

        self._rate_limit_check()
        macro_data = {}

        try:
            if indicator == "all" or indicator == "GDP":
                gdp_data = ak.macro_china_gdp_yearly()
                macro_data["GDP"] = gdp_data
                logger.info(f"Fetched GDP data: {len(gdp_data)} records")

            if indicator == "all" or indicator == "CPI":
                cpi_data = ak.macro_china_cpi_monthly()
                macro_data["CPI"] = cpi_data
                logger.info(f"Fetched CPI data: {len(cpi_data)} records")

            if indicator == "all" or indicator == "PMI":
                pmi_data = ak.macro_china_pmi_yearly()
                macro_data["PMI"] = pmi_data
                logger.info(f"Fetched PMI data: {len(pmi_data)} records")

            return macro_data

        except Exception as e:
            logger.error(f"Failed to fetch macro economic data: {e}")
            raise DataError(f"Macro data fetch failed: {e}")

    def fetch_stock_news(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """
        获取个股新闻数据

        Args:
            symbol: 股票代码
            limit: 返回条数限制

        Returns:
            pd.DataFrame: 个股新闻数据，包含关键词、新闻标题、内容、来源等
        """
        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for stock news data collection. Install with: pip install akshare"
            )

        self._rate_limit_check()

        try:
            # 获取个股新闻数据
            news_df = ak.stock_news_em(symbol=symbol)

            if news_df.empty:
                logger.warning(f"No stock news data found for symbol {symbol}")
                return pd.DataFrame()

            # 限制返回条数
            if len(news_df) > limit:
                news_df = news_df.head(limit)

            logger.info(f"Fetched {len(news_df)} stock news items for {symbol}")
            return news_df

        except Exception as e:
            logger.error(f"Failed to fetch stock news data for {symbol}: {e}")
            raise DataError(f"Stock news data fetch failed: {e}")

    def fetch_daily_market_overview(self, date: str = None) -> pd.DataFrame:
        """
        获取上海证券交易所每日概况数据
        默认获取近一年的数据

        Args:
            date: 日期，格式为YYYYMMDD，默认为近一年数据

        Returns:
            pd.DataFrame: 每日市场概况数据
        """
        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for daily market overview collection. Install with: pip install akshare"
            )

        # 如果指定了具体日期，只获取那一天的数据
        if date is not None:
            return self._fetch_single_day_market_overview(date)

        # 默认获取近一年的数据
        logger.info("默认获取近一年的市场概况数据")
        return self.fetch_one_year_market_overview()

    def _fetch_single_day_market_overview(self, date: str) -> pd.DataFrame:
        """
        获取单日市场概况数据

        Args:
            date: 日期，格式为YYYYMMDD

        Returns:
            pd.DataFrame: 单日市场概况数据
        """
        self._rate_limit_check()

        try:
            # 获取上海证券交易所每日概况
            overview_df = ak.stock_sse_deal_daily(date=date)

            if overview_df.empty:
                logger.warning(f"No daily market overview data found for date {date}")
                # 如果没有数据，尝试获取前一天的数据
                yesterday = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime(
                    "%Y%m%d"
                )
                overview_df = ak.stock_sse_deal_daily(date=yesterday)
                if overview_df.empty:
                    raise DataError(
                        f"No daily market overview data available for {date} or {yesterday}"
                    )
                else:
                    date = yesterday

            # 添加日期列
            overview_df["date"] = date

            logger.info(f"Fetched daily market overview for date {date}")
            return overview_df

        except Exception as e:
            logger.error(f"Failed to fetch daily market overview: {e}")
            raise DataError(f"Daily market overview fetch failed: {e}")

    def fetch_detail(self, symbol: str) -> Dict[str, Any]:
        """
        获取个股详细信息（完整版，统一接口）
        结合东财和雪球两个API获取全面信息

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 个股详细信息
        """
        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for individual stock info collection. Install with: pip install akshare"
            )

        self._rate_limit_check()

        try:
            # 先获取东财的基本信息
            basic_info = {}
            try:
                info_df = ak.stock_individual_info_em(symbol=symbol)
                if not info_df.empty:
                    for _, row in info_df.iterrows():
                        item = row.get("item", "")
                        value = row.get("value", "")
                        if item and value:
                            basic_info[item] = value
            except Exception as e:
                logger.warning(f"Failed to fetch basic info from EM for {symbol}: {e}")

            # 在获取雪球的详细信息
            detailed_info = {}
            try:
                # 根据股票代码格式化为雪球格式
                if symbol.startswith("0") or symbol.startswith("3"):
                    xq_symbol = f"SZ{symbol}"
                elif symbol.startswith("6"):
                    xq_symbol = f"SH{symbol}"
                else:
                    xq_symbol = symbol

                detail_df = ak.stock_individual_basic_info_xq(symbol=xq_symbol)
                if not detail_df.empty:
                    for _, row in detail_df.iterrows():
                        item = row.get("item", "")
                        value = row.get("value", "")
                        if item and value:
                            detailed_info[item] = value
            except Exception as e:
                logger.warning(
                    f"Failed to fetch detailed info from XQ for {symbol}: {e}"
                )

            # 如果两个源都没有数据，返回空
            if not basic_info and not detailed_info:
                logger.warning(f"No individual stock info found for symbol {symbol}")
                return {}

            # 整合两个数据源，使用实际返回的字段名
            comprehensive_info = {
                "symbol": symbol,
                # 基本信息（东财）
                "stock_code": basic_info.get("股票代码", symbol),
                "name": basic_info.get(
                    "股票简称", detailed_info.get("org_short_name_cn", "")
                ),
                "latest_price": self._safe_float(basic_info.get("最新", 0)),
                "total_shares": self._safe_float(basic_info.get("总股本", 0)),
                "circulating_shares": self._safe_float(basic_info.get("流通股", 0)),
                "total_market_value": self._safe_float(basic_info.get("总市值", 0)),
                "circulating_market_value": self._safe_float(
                    basic_info.get("流通市值", 0)
                ),
                "industry": basic_info.get("行业", ""),
                "listing_date": str(basic_info.get("上市时间", "")),
                # 公司基本信息（雪球）
                "org_name_cn": detailed_info.get("org_name_cn", ""),
                "org_short_name_cn": detailed_info.get("org_short_name_cn", ""),
                "org_name_en": detailed_info.get("org_name_en", ""),
                "org_short_name_en": detailed_info.get("org_short_name_en", ""),
                "main_operation_business": detailed_info.get(
                    "main_operation_business", ""
                ),
                "operating_scope": detailed_info.get("operating_scope", ""),
                "org_cn_introduction": detailed_info.get("org_cn_introduction", ""),
                # 管理层信息
                "legal_representative": detailed_info.get("legal_representative", ""),
                "general_manager": detailed_info.get("general_manager", ""),
                "secretary": detailed_info.get("secretary", ""),
                "chairman": detailed_info.get("chairman", ""),
                "executives_nums": self._safe_int(
                    detailed_info.get("executives_nums", 0)
                ),
                # 财务信息
                "established_date": str(detailed_info.get("established_date", "")),
                "reg_asset": self._safe_float(detailed_info.get("reg_asset", 0)),
                "staff_num": self._safe_int(detailed_info.get("staff_num", 0)),
                "currency": detailed_info.get("currency", ""),
                "listed_date_timestamp": str(detailed_info.get("listed_date", "")),
                # 联系信息
                "telephone": detailed_info.get("telephone", ""),
                "postcode": detailed_info.get("postcode", ""),
                "fax": detailed_info.get("fax", ""),
                "email": detailed_info.get("email", ""),
                "org_website": detailed_info.get("org_website", ""),
                "reg_address_cn": detailed_info.get("reg_address_cn", ""),
                "reg_address_en": detailed_info.get("reg_address_en", ""),
                "office_address_cn": detailed_info.get("office_address_cn", ""),
                "office_address_en": detailed_info.get("office_address_en", ""),
                # 控制权信息
                "provincial_name": detailed_info.get("provincial_name", ""),
                "actual_controller": detailed_info.get("actual_controller", ""),
                "classi_name": detailed_info.get("classi_name", ""),
                "pre_name_cn": detailed_info.get("pre_name_cn", ""),
                # 发行信息
                "actual_issue_vol": self._safe_float(
                    detailed_info.get("actual_issue_vol", 0)
                ),
                "issue_price": self._safe_float(detailed_info.get("issue_price", 0)),
                "actual_rc_net_amt": self._safe_float(
                    detailed_info.get("actual_rc_net_amt", 0)
                ),
                "pe_after_issuing": self._safe_float(
                    detailed_info.get("pe_after_issuing", 0)
                ),
                "online_success_rate_of_issue": self._safe_float(
                    detailed_info.get("online_success_rate_of_issue", 0)
                ),
                # 行业信息
                "affiliate_industry_code": "",
                "affiliate_industry_name": "",
                "update_time": datetime.now().isoformat(),
            }

            # 处理行业信息（如果是字典格式）
            affiliate_industry = detailed_info.get("affiliate_industry")
            if affiliate_industry and isinstance(affiliate_industry, dict):
                comprehensive_info["affiliate_industry_code"] = affiliate_industry.get(
                    "ind_code", ""
                )
                comprehensive_info["affiliate_industry_name"] = affiliate_industry.get(
                    "ind_name", ""
                )
            elif affiliate_industry and isinstance(affiliate_industry, str):
                # 如果是字符串，尝试解析
                try:
                    import ast

                    parsed = ast.literal_eval(affiliate_industry)
                    if isinstance(parsed, dict):
                        comprehensive_info["affiliate_industry_code"] = parsed.get(
                            "ind_code", ""
                        )
                        comprehensive_info["affiliate_industry_name"] = parsed.get(
                            "ind_name", ""
                        )
                except:
                    pass

            logger.info(f"Fetched comprehensive individual stock info for {symbol}")
            return comprehensive_info

        except Exception as e:
            logger.error(f"Failed to fetch individual stock info for {symbol}: {e}")
            raise DataError(f"Individual stock info fetch failed: {e}")

    def _safe_int(self, value, default=0) -> int:
        """
        安全转换为整数类型

        Args:
            value: 要转换的值
            default: 默认值

        Returns:
            int: 转换后的整数，失败返回默认值
        """
        try:
            if pd.isna(value) or value is None or value == "":
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value) -> float:
        """
        安全转换为float类型

        Args:
            value: 要转换的值

        Returns:
            float: 转换后的数值，失败返回0.0
        """
        try:
            if pd.isna(value) or value is None or value == "":
                return 0.0
            # 移除可能的单位名称或特殊字符
            if isinstance(value, str):
                value = value.replace(",", "").replace("万", "").replace("亿", "")
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def fetch_historical_news_data(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        获取历史新闻联播数据（按日期范围）

        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD

        Returns:
            pd.DataFrame: 历史新闻数据
        """
        all_news = []
        current_date = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        while current_date <= end:
            date_str = current_date.strftime("%Y%m%d")
            try:
                daily_news = self.fetch_news_data(date=date_str, limit=20)
                if not daily_news.empty:
                    all_news.append(daily_news)
                    logger.info(f"Fetched news for {date_str}: {len(daily_news)} items")
                else:
                    logger.debug(f"No news found for {date_str}")
            except Exception as e:
                logger.warning(f"Failed to fetch news for {date_str}: {e}")

            # 移动到下一天
            current_date += pd.Timedelta(days=1)
            # 率限制控制
            self._rate_limit_check()

        if all_news:
            combined_news = pd.concat(all_news, ignore_index=True)
            logger.info(
                f"Fetched total {len(combined_news)} historical news items from {start_date} to {end_date}"
            )
            return combined_news
        else:
            logger.warning(
                f"No historical news data found from {start_date} to {end_date}"
            )
            return pd.DataFrame()

    def fetch_historical_stock_news(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        获取个股历史新闻数据（按天数）
        注：个股新闻接口通常返回最近的新闻，不是按日期查询

        Args:
            symbol: 股票代码
            days: 天数（用作参考，实际数据量由接口决定）

        Returns:
            pd.DataFrame: 个股历史新闻数据
        """
        try:
            # 个股新闻接口通常返回最近100条新闻，不支持日期范围查询
            stock_news = self.fetch_stock_news(symbol, limit=100)
            if not stock_news.empty:
                logger.info(
                    f"Fetched {len(stock_news)} historical stock news for {symbol}"
                )
            return stock_news
        except Exception as e:
            logger.error(f"Failed to fetch historical stock news for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_daily_market_overview(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        获取历史每日市场概况数据（默认1年数据）

        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD

        Returns:
            pd.DataFrame: 历史每日市场概况数据
        """
        all_overviews = []
        current_date = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 计算总天数和预估工作日
        total_days = (end - current_date).days + 1
        estimated_work_days = total_days * 5 // 7  # 预估工作日
        processed_days = 0

        logger.info(
            f"Starting to collect market overview data from {start_date} to {end_date}"
        )
        logger.info(f"Estimated {estimated_work_days} work days to process")

        while current_date <= end:
            date_str = current_date.strftime("%Y%m%d")
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4代表周一到周五
                try:
                    daily_overview = self._fetch_single_day_market_overview(
                        date=date_str
                    )
                    if not daily_overview.empty:
                        all_overviews.append(daily_overview)
                        processed_days += 1
                        if processed_days % 50 == 0:  # 每50天显示一次进度
                            logger.info(
                                f"Processed {processed_days} work days, collected {len(all_overviews)} records"
                            )
                    else:
                        logger.debug(f"No market overview found for {date_str}")
                except Exception as e:
                    # 早期数据可能不存在，记录但不中断
                    if "20211227" in str(e) or "不支持获取" in str(e):
                        logger.debug(
                            f"Data not available for {date_str} (before 2021-12-27)"
                        )
                    else:
                        logger.warning(
                            f"Failed to fetch market overview for {date_str}: {e}"
                        )

            # 移动到下一天
            current_date += pd.Timedelta(days=1)
            # 率限制控制
            self._rate_limit_check()

        if all_overviews:
            # 保存每一天的数据，需要传递日期信息
            logger.info(
                f"Successfully collected {len(all_overviews)} historical market overview records from {start_date} to {end_date}"
            )
            # 为了方便批量处理，返回所有数据
            # 每个DataFrame都已经包含了date列
            combined_overviews = pd.concat(all_overviews, ignore_index=True)
            return combined_overviews
        else:
            logger.warning(
                f"No historical market overview data found from {start_date} to {end_date}"
            )
            return pd.DataFrame()

    def fetch_one_year_market_overview(self) -> pd.DataFrame:
        """
        获取近一月的市场概况数据

        Returns:
            pd.DataFrame: 近一月市场概况数据
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        return self.fetch_historical_daily_market_overview(
            start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        )

    def fetch_news_data(self, date: str = None, limit: int = 50) -> pd.DataFrame:
        """
        获取新闻联播文字稿数据

        Args:
            date: 新闻日期，格式为YYYYMMDD，如"20240424"，默认为今天
            limit: 返回条数限制（实际返回条数由akshare接口决定）

        Returns:
            pd.DataFrame: 新闻数据，包含date、title、content列
        """
        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for news data collection. Install with: pip install akshare"
            )

        self._rate_limit_check()

        try:
            # 如果没有指定日期，使用今天的日期
            if date is None:
                date = pd.Timestamp.now().strftime("%Y%m%d")

            # 获取新闻联播文字稿
            news_df = ak.news_cctv(date=date)

            if news_df.empty:
                logger.warning(f"No news data found for date {date}")
                # 如果没有数据，尝试获取前一天的数据
                yesterday = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime(
                    "%Y%m%d"
                )
                news_df = ak.news_cctv(date=yesterday)
                if news_df.empty:
                    raise DataError(f"No news data available for {date} or {yesterday}")

            # 限制返回条数
            if len(news_df) > limit:
                news_df = news_df.head(limit)

            # 添加情绪分析（简单的关键词匹配）
            if "content" in news_df.columns:
                news_df["sentiment"] = news_df["content"].apply(self._analyze_sentiment)
            elif "内容" in news_df.columns:
                news_df["sentiment"] = news_df["内容"].apply(self._analyze_sentiment)
            else:
                # 如果没有内容列，默认为中性
                news_df["sentiment"] = "neutral"

            logger.info(f"Fetched {len(news_df)} real news items for date {date}")
            return news_df

        except Exception as e:
            logger.error(f"Failed to fetch news data: {e}")
            raise DataError(f"News data fetch failed: {e}")

    def _analyze_sentiment(self, content: str) -> str:
        """
        简单的情绪分析（基于关键词）

        Args:
            content: 新闻内容

        Returns:
            str: 情绪标签（positive/neutral/negative）
        """
        if pd.isna(content) or not isinstance(content, str):
            return "neutral"

        positive_keywords = [
            "增长",
            "上涨",
            "利好",
            "突破",
            "创新",
            "发展",
            "合作",
            "成功",
            "繁荣",
            "优化",
        ]
        negative_keywords = [
            "下跌",
            "风险",
            "危机",
            "问题",
            "困难",
            "挑战",
            "担忧",
            "衰退",
            "下降",
            "损失",
        ]

        positive_count = sum(1 for keyword in positive_keywords if keyword in content)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def fetch_sector_performance(self, indicator: str = "新浪行业") -> pd.DataFrame:
        """
        获取板块行情数据

        Args:
            indicator: 板块类型 ("新浪行业", "启明星行业", "概念", "地域", "行业")

        Returns:
            pd.DataFrame: 板块行情数据
        """
        if not HAS_AKSHARE:
            raise ImportError(
                "Akshare is required for sector data collection. Install with: pip install akshare"
            )

        self._rate_limit_check()

        try:
            # 获取板块行情数据
            sector_df = ak.stock_sector_spot(indicator=indicator)

            if sector_df.empty:
                logger.warning(f"No sector data found for indicator {indicator}")
                raise DataError(f"No sector data available for indicator: {indicator}")

            # 添加日期列
            sector_df["date"] = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetched {len(sector_df)} real sector records for {indicator}")
            return sector_df

        except Exception as e:
            logger.error(f"Failed to fetch sector data: {e}")
            raise DataError(f"Sector data fetch failed: {e}")


# For backward compatibility, create alias
AlternativeDataCollector = ChineseAlternativeDataCollector

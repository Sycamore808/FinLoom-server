# -*- coding: utf-8 -*-
"""
替代数据采集器模块
支持多种第三方API、爬虫等方式采集金融相关替代数据
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("alternative_data_collector")


class AlternativeDataCollector:
    """
    替代数据采集器，支持多种数据源（如宏观经济、社交媒体、新闻、天气等）
    """

    def __init__(self, proxies: Optional[Dict[str, str]] = None):
        self.proxies = proxies

    def fetch_macro_data(
        self, indicator: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        获取宏观经济数据（示例：FRED API）
        Args:
                indicator: 指标代码（如GDP、CPI等）
                start_date: 开始日期（YYYY-MM-DD）
                end_date: 结束日期（YYYY-MM-DD）
        Returns:
                pd.DataFrame: 包含日期和数值的DataFrame
        Raises:
                DataError: 数据获取失败
        """
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={indicator}&cosd={start_date}&coed={end_date}"
            resp = requests.get(url, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            df = pd.read_csv(pd.compat.StringIO(resp.text))
            logger.info(
                f"Fetched macro data: {indicator} {start_date}~{end_date}, {len(df)} rows"
            )
            return df
        except Exception as e:
            logger.error(f"Failed to fetch macro data: {indicator}, error: {e}")
            raise DataError(f"Macro data fetch failed: {e}")

    def fetch_social_sentiment(
        self, keyword: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取社交媒体情绪数据（示例：Reddit API，伪代码）
        Args:
                keyword: 关键词
                limit: 最大返回条数
        Returns:
                List[Dict[str, Any]]: 帖子列表
        Raises:
                DataError: 数据获取失败
        """
        try:
            # 这里只做伪实现，实际可用praw等库
            url = f"https://www.reddit.com/search.json?q={keyword}&limit={limit}"
            headers = {"User-Agent": "FinLoomBot/1.0"}
            resp = requests.get(url, headers=headers, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            posts = [
                {
                    "title": item["data"]["title"],
                    "created_utc": datetime.utcfromtimestamp(
                        item["data"]["created_utc"]
                    ),
                    "score": item["data"]["score"],
                    "num_comments": item["data"]["num_comments"],
                }
                for item in data["data"]["children"]
            ]
            logger.info(f"Fetched {len(posts)} social sentiment posts for '{keyword}'")
            return posts
        except Exception as e:
            logger.error(f"Failed to fetch social sentiment: {keyword}, error: {e}")
            raise DataError(f"Social sentiment fetch failed: {e}")

    def fetch_news(
        self, keyword: str, from_date: str, to_date: str, api_key: str
    ) -> List[Dict[str, Any]]:
        """
        获取新闻数据（示例：NewsAPI）
        Args:
                keyword: 关键词
                from_date: 起始日期（YYYY-MM-DD）
                to_date: 截止日期（YYYY-MM-DD）
                api_key: NewsAPI密钥
        Returns:
                List[Dict[str, Any]]: 新闻列表
        Raises:
                DataError: 数据获取失败
        """
        try:
            url = f"https://newsapi.org/v2/everything?q={keyword}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}"
            resp = requests.get(url, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            logger.info(f"Fetched {len(articles)} news articles for '{keyword}'")
            return articles
        except Exception as e:
            logger.error(f"Failed to fetch news: {keyword}, error: {e}")
            raise DataError(f"News fetch failed: {e}")

    def fetch_weather(self, location: str, api_key: str) -> Dict[str, Any]:
        """
        获取天气数据（示例：OpenWeatherMap API）
        Args:
                location: 地点（如城市名）
                api_key: OpenWeatherMap密钥
        Returns:
                Dict[str, Any]: 天气信息
        Raises:
                DataError: 数据获取失败
        """
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
            resp = requests.get(url, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            weather = resp.json()
            logger.info(
                f"Fetched weather for {location}: {weather.get('weather', [{}])[0].get('description', '')}"
            )
            return weather
        except Exception as e:
            logger.error(f"Failed to fetch weather: {location}, error: {e}")
            raise DataError(f"Weather fetch failed: {e}")

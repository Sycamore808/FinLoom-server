"""
反爬虫工具模块
提供User-Agent轮换、代理IP、重试机制等功能
"""

import random
import time
from functools import wraps
from typing import Callable, Dict, List, Optional

from common.logging_system import setup_logger

logger = setup_logger("anti_spider_utils")


# User-Agent池
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/118.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


def get_random_user_agent() -> str:
    """获取随机User-Agent"""
    return random.choice(USER_AGENTS)


def random_delay(min_delay: float = 0.5, max_delay: float = 2.0):
    """随机延迟，避免请求过于规律

    Args:
        min_delay: 最小延迟时间（秒）
        max_delay: 最大延迟时间（秒）
    """
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
):
    """重试装饰器，使用指数退避策略

    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟时间（秒）
        backoff_factor: 退避因子
        max_delay: 最大延迟时间（秒）
        exceptions: 需要捕获的异常类型
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(
                            f"重试 {func.__name__}，第 {attempt}/{max_retries} 次"
                        )

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(f"{func.__name__} 在第 {attempt} 次重试后成功")

                    return result

                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # 添加随机性，避免多个请求同时重试
                        actual_delay = delay * (0.5 + random.random())
                        logger.warning(
                            f"{func.__name__} 失败: {str(e)}，"
                            f"{actual_delay:.2f}秒后重试（{attempt + 1}/{max_retries}）"
                        )
                        time.sleep(actual_delay)

                        # 指数退避
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__} 在 {max_retries} 次重试后仍然失败: {str(e)}"
                        )

            raise last_exception

        return wrapper

    return decorator


class AntiSpiderSession:
    """反爬虫会话管理器"""

    def __init__(
        self,
        min_delay: float = 0.5,
        max_delay: float = 2.0,
        rotate_user_agent: bool = True,
    ):
        """初始化反爬虫会话

        Args:
            min_delay: 最小请求间隔（秒）
            max_delay: 最大请求间隔（秒）
            rotate_user_agent: 是否轮换User-Agent
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.rotate_user_agent = rotate_user_agent
        self.last_request_time = 0
        self.request_count = 0
        self.current_user_agent = get_random_user_agent()

    def get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        if self.rotate_user_agent and self.request_count % 5 == 0:
            # 每5次请求更换一次User-Agent
            self.current_user_agent = get_random_user_agent()

        headers = {
            "User-Agent": self.current_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        return headers

    def throttle(self):
        """请求节流"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            # 添加随机性
            sleep_time += random.uniform(0, self.max_delay - self.min_delay)
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1


class ProxyPool:
    """代理IP池（可扩展）"""

    def __init__(self, proxies: Optional[List[str]] = None):
        """初始化代理池

        Args:
            proxies: 代理列表，格式如 ["http://ip:port", ...]
        """
        self.proxies = proxies or []
        self.current_index = 0
        self.failed_proxies = set()

    def get_proxy(self) -> Optional[Dict[str, str]]:
        """获取下一个可用代理"""
        if not self.proxies:
            return None

        available_proxies = [p for p in self.proxies if p not in self.failed_proxies]

        if not available_proxies:
            # 所有代理都失败了，重置失败列表
            logger.warning("所有代理都已失败，重置代理池")
            self.failed_proxies.clear()
            available_proxies = self.proxies

        proxy = available_proxies[self.current_index % len(available_proxies)]
        self.current_index += 1

        return {"http": proxy, "https": proxy}

    def mark_failed(self, proxy: str):
        """标记代理失败"""
        self.failed_proxies.add(proxy)
        logger.warning(f"代理 {proxy} 已标记为失败")

    def add_proxy(self, proxy: str):
        """添加新代理"""
        if proxy not in self.proxies:
            self.proxies.append(proxy)
            logger.info(f"添加新代理: {proxy}")

    def remove_proxy(self, proxy: str):
        """移除代理"""
        if proxy in self.proxies:
            self.proxies.remove(proxy)
            logger.info(f"移除代理: {proxy}")


def patch_akshare_headers():
    """为akshare打补丁，使其使用随机User-Agent

    注意：这是一个实验性功能，可能随akshare版本变化而失效
    """
    try:
        import requests

        # 保存原始的get方法
        original_get = requests.Session.get
        original_post = requests.Session.post

        session = AntiSpiderSession()

        def patched_get(self, url, **kwargs):
            """打补丁的get方法"""
            # 添加随机延迟
            session.throttle()

            # 使用随机User-Agent
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            kwargs["headers"].update(session.get_headers())

            return original_get(self, url, **kwargs)

        def patched_post(self, url, **kwargs):
            """打补丁的post方法"""
            # 添加随机延迟
            session.throttle()

            # 使用随机User-Agent
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            kwargs["headers"].update(session.get_headers())

            return original_post(self, url, **kwargs)

        # 应用补丁
        requests.Session.get = patched_get
        requests.Session.post = patched_post

        logger.info("✅ 已为akshare应用反爬虫补丁")
        return True

    except Exception as e:
        logger.warning(f"无法为akshare打补丁: {e}")
        return False


# 全局代理池实例
_proxy_pool: Optional[ProxyPool] = None


def init_proxy_pool(proxies: List[str]):
    """初始化全局代理池

    Args:
        proxies: 代理列表
    """
    global _proxy_pool
    _proxy_pool = ProxyPool(proxies)
    logger.info(f"初始化代理池，共 {len(proxies)} 个代理")


def get_proxy_pool() -> Optional[ProxyPool]:
    """获取全局代理池"""
    return _proxy_pool

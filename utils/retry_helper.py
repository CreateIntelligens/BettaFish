"""
重試機制工具模塊
提供通用的網絡請求重試功能，增強系統健壯性
"""

import time
from functools import wraps
from typing import Callable, Any
import requests
from loguru import logger

# 配置日誌
class RetryConfig:
    """重試配置類"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retry_on_exceptions: tuple = None
    ):
        """
        初始化重試配置
        
        Args:
            max_retries: 最大重試次數
            initial_delay: 初始延遲秒數
            backoff_factor: 退避因子（每次重試延遲翻倍）
            max_delay: 最大延遲秒數
            retry_on_exceptions: 需要重試的異常類型元組
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        
        # 默認需要重試的異常類型
        if retry_on_exceptions is None:
            self.retry_on_exceptions = (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.TooManyRedirects,
                ConnectionError,
                TimeoutError,
                Exception  # OpenAI和其他API可能拋出的一般異常
            )
        else:
            self.retry_on_exceptions = retry_on_exceptions

# 默認配置
DEFAULT_RETRY_CONFIG = RetryConfig()

def with_retry(config: RetryConfig = None):
    """
    重試裝飾器
    
    Args:
        config: 重試配置，如果不提供則使用默認配置
    
    Returns:
        裝飾器函數
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 因爲第一次不算重試
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"函數 {func.__name__} 在第 {attempt + 1} 次嘗試後成功")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # 最後一次嘗試也失敗了
                        logger.error(f"函數 {func.__name__} 在 {config.max_retries + 1} 次嘗試後仍然失敗")
                        logger.error(f"最終錯誤: {str(e)}")
                        raise e
                    
                    # 計算延遲時間
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"函數 {func.__name__} 第 {attempt + 1} 次嘗試失敗: {str(e)}")
                    logger.info(f"將在 {delay:.1f} 秒後進行第 {attempt + 2} 次嘗試...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # 不在重試列表中的異常，直接拋出
                    logger.error(f"函數 {func.__name__} 遇到不可重試的異常: {str(e)}")
                    raise e
            
            # 這裏不應該到達，但作爲安全網
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator

def retry_on_network_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    專門用於網絡錯誤的重試裝飾器（簡化版）
    
    Args:
        max_retries: 最大重試次數
        initial_delay: 初始延遲秒數
        backoff_factor: 退避因子
    
    Returns:
        裝飾器函數
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    return with_retry(config)

class RetryableError(Exception):
    """自定義的可重試異常"""
    pass

def with_graceful_retry(config: RetryConfig = None, default_return=None):
    """
    優雅重試裝飾器 - 用於非關鍵API調用
    失敗後不會拋出異常，而是返回默認值，保證系統繼續運行
    
    Args:
        config: 重試配置，如果不提供則使用默認配置
        default_return: 所有重試失敗後返回的默認值
    
    Returns:
        裝飾器函數
    """
    if config is None:
        config = SEARCH_API_RETRY_CONFIG
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 因爲第一次不算重試
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"非關鍵API {func.__name__} 在第 {attempt + 1} 次嘗試後成功")
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        # 最後一次嘗試也失敗了，返回默認值而不拋出異常
                        logger.warning(f"非關鍵API {func.__name__} 在 {config.max_retries + 1} 次嘗試後仍然失敗")
                        logger.warning(f"最終錯誤: {str(e)}")
                        logger.info(f"返回默認值以保證系統繼續運行: {default_return}")
                        return default_return
                    
                    # 計算延遲時間
                    delay = min(
                        config.initial_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )
                    
                    logger.warning(f"非關鍵API {func.__name__} 第 {attempt + 1} 次嘗試失敗: {str(e)}")
                    logger.info(f"將在 {delay:.1f} 秒後進行第 {attempt + 2} 次嘗試...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # 不在重試列表中的異常，返回默認值
                    logger.warning(f"非關鍵API {func.__name__} 遇到不可重試的異常: {str(e)}")
                    logger.info(f"返回默認值以保證系統繼續運行: {default_return}")
                    return default_return
            
            # 這裏不應該到達，但作爲安全網
            return default_return
            
        return wrapper
    return decorator

def make_retryable_request(
    request_func: Callable,
    *args,
    max_retries: int = 5,
    **kwargs
) -> Any:
    """
    直接執行可重試的請求（不使用裝飾器）
    
    Args:
        request_func: 要執行的請求函數
        *args: 傳遞給請求函數的位置參數
        max_retries: 最大重試次數
        **kwargs: 傳遞給請求函數的關鍵字參數
    
    Returns:
        請求函數的返回值
    """
    config = RetryConfig(max_retries=max_retries)
    
    @with_retry(config)
    def _execute():
        return request_func(*args, **kwargs)
    
    return _execute()

# 預定義一些常用的重試配置
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=6,        # 保持額外重試次數
    initial_delay=60.0,   # 首次等待至少 1 分鐘
    backoff_factor=2.0,   # 繼續使用指數退避
    max_delay=600.0       # 單次等待最長 10 分鐘
)

SEARCH_API_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # 增加到5次重試
    initial_delay=2.0,    # 增加初始延遲
    backoff_factor=1.6,   # 調整退避因子
    max_delay=25.0        # 增加最大延遲
)

DB_RETRY_CONFIG = RetryConfig(
    max_retries=5,        # 增加到5次重試
    initial_delay=1.0,    # 保持較短的數據庫重試延遲
    backoff_factor=1.5,
    max_delay=10.0
)

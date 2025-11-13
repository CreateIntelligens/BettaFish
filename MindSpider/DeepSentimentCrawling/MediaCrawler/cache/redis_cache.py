# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
# 1. 不得用於任何商業用途。  
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。  
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。  
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。   
# 5. 不得用於任何非法或不當的用途。
#   
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。  
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。  


# -*- coding: utf-8 -*-
# @Author  : relakkes@gmail.com
# @Name    : 程序員阿江-Relakkes
# @Time    : 2024/5/29 22:57
# @Desc    : RedisCache實現
import pickle
import time
from typing import Any, List

from redis import Redis

from cache.abs_cache import AbstractCache
from config import db_config


class RedisCache(AbstractCache):

    def __init__(self) -> None:
        # 連接redis, 返回redis客戶端
        self._redis_client = self._connet_redis()

    @staticmethod
    def _connet_redis() -> Redis:
        """
        連接redis, 返回redis客戶端, 這裏按需配置redis連接信息
        :return:
        """
        return Redis(
            host=db_config.REDIS_DB_HOST,
            port=db_config.REDIS_DB_PORT,
            db=db_config.REDIS_DB_NUM,
            password=db_config.REDIS_DB_PWD,
        )

    def get(self, key: str) -> Any:
        """
        從緩存中獲取鍵的值, 並且反序列化
        :param key:
        :return:
        """
        value = self._redis_client.get(key)
        if value is None:
            return None
        return pickle.loads(value)

    def set(self, key: str, value: Any, expire_time: int) -> None:
        """
        將鍵的值設置到緩存中, 並且序列化
        :param key:
        :param value:
        :param expire_time:
        :return:
        """
        self._redis_client.set(key, pickle.dumps(value), ex=expire_time)

    def keys(self, pattern: str) -> List[str]:
        """
        獲取所有符合pattern的key
        """
        return [key.decode() for key in self._redis_client.keys(pattern)]


if __name__ == '__main__':
    redis_cache = RedisCache()
    # basic usage
    redis_cache.set("name", "程序員阿江-Relakkes", 1)
    print(redis_cache.get("name"))  # Relakkes
    print(redis_cache.keys("*"))  # ['name']
    time.sleep(2)
    print(redis_cache.get("name"))  # None

    # special python type usage
    # list
    redis_cache.set("list", [1, 2, 3], 10)
    _value = redis_cache.get("list")
    print(_value, f"value type:{type(_value)}")  # [1, 2, 3]

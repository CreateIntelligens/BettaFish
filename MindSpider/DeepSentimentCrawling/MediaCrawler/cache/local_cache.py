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
# @Time    : 2024/6/2 11:05
# @Desc    : 本地緩存

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from cache.abs_cache import AbstractCache


class ExpiringLocalCache(AbstractCache):

    def __init__(self, cron_interval: int = 10):
        """
        初始化本地緩存
        :param cron_interval: 定時清楚cache的時間間隔
        :return:
        """
        self._cron_interval = cron_interval
        self._cache_container: Dict[str, Tuple[Any, float]] = {}
        self._cron_task: Optional[asyncio.Task] = None
        # 開啓定時清理任務
        self._schedule_clear()

    def __del__(self):
        """
        析構函數，清理定時任務
        :return:
        """
        if self._cron_task is not None:
            self._cron_task.cancel()

    def get(self, key: str) -> Optional[Any]:
        """
        從緩存中獲取鍵的值
        :param key:
        :return:
        """
        value, expire_time = self._cache_container.get(key, (None, 0))
        if value is None:
            return None

        # 如果鍵已過期，則刪除鍵並返回None
        if expire_time < time.time():
            del self._cache_container[key]
            return None

        return value

    def set(self, key: str, value: Any, expire_time: int) -> None:
        """
        將鍵的值設置到緩存中
        :param key:
        :param value:
        :param expire_time:
        :return:
        """
        self._cache_container[key] = (value, time.time() + expire_time)

    def keys(self, pattern: str) -> List[str]:
        """
        獲取所有符合pattern的key
        :param pattern: 匹配模式
        :return:
        """
        if pattern == '*':
            return list(self._cache_container.keys())

        # 本地緩存通配符暫時將*替換爲空
        if '*' in pattern:
            pattern = pattern.replace('*', '')

        return [key for key in self._cache_container.keys() if pattern in key]

    def _schedule_clear(self):
        """
        開啓定時清理任務,
        :return:
        """

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._cron_task = loop.create_task(self._start_clear_cron())

    def _clear(self):
        """
        根據過期時間清理緩存
        :return:
        """
        for key, (value, expire_time) in self._cache_container.items():
            if expire_time < time.time():
                del self._cache_container[key]

    async def _start_clear_cron(self):
        """
        開啓定時清理任務
        :return:
        """
        while True:
            self._clear()
            await asyncio.sleep(self._cron_interval)


if __name__ == '__main__':
    cache = ExpiringLocalCache(cron_interval=2)
    cache.set('name', '程序員阿江-Relakkes', 3)
    print(cache.get('key'))
    print(cache.keys("*"))
    time.sleep(4)
    print(cache.get('key'))
    del cache
    time.sleep(1)
    print("done")

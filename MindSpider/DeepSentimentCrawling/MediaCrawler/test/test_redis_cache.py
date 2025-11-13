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
# @Time    : 2024/6/2 19:54
# @Desc    :

import time
import unittest

from cache.redis_cache import RedisCache


class TestRedisCache(unittest.TestCase):

    def setUp(self):
        self.redis_cache = RedisCache()

    def test_set_and_get(self):
        self.redis_cache.set('key', 'value', 10)
        self.assertEqual(self.redis_cache.get('key'), 'value')

    def test_expired_key(self):
        self.redis_cache.set('key', 'value', 1)
        time.sleep(2)  # wait for the key to expire
        self.assertIsNone(self.redis_cache.get('key'))

    def test_keys(self):
        self.redis_cache.set('key1', 'value1', 10)
        self.redis_cache.set('key2', 'value2', 10)
        keys = self.redis_cache.keys('*')
        self.assertIn('key1', keys)
        self.assertIn('key2', keys)

    def tearDown(self):
        # self.redis_cache._redis_client.flushdb()  # 清空redis數據庫
        pass


if __name__ == '__main__':
    unittest.main()

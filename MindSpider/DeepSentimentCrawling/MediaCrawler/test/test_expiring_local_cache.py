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
# @Time    : 2024/6/2 10:35
# @Desc    :

import time
import unittest

from cache.local_cache import ExpiringLocalCache


class TestExpiringLocalCache(unittest.TestCase):

    def setUp(self):
        self.cache = ExpiringLocalCache(cron_interval=10)

    def test_set_and_get(self):
        self.cache.set('key', 'value', 10)
        self.assertEqual(self.cache.get('key'), 'value')

    def test_expired_key(self):
        self.cache.set('key', 'value', 1)
        time.sleep(2)  # wait for the key to expire
        self.assertIsNone(self.cache.get('key'))

    def test_clear(self):
        # 設置兩個鍵值對，過期時間爲11秒
        self.cache.set('key', 'value', 11)
        # 睡眠12秒，讓cache類的定時任務執行一次
        time.sleep(12)
        self.assertIsNone(self.cache.get('key'))

    def tearDown(self):
        del self.cache


if __name__ == '__main__':
    unittest.main()

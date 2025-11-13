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
# @Time    : 2024/6/2 11:23
# @Desc    :


class CacheFactory:
    """
    緩存工廠類
    """

    @staticmethod
    def create_cache(cache_type: str, *args, **kwargs):
        """
        創建緩存對象
        :param cache_type: 緩存類型
        :param args: 參數
        :param kwargs: 關鍵字參數
        :return:
        """
        if cache_type == 'memory':
            from .local_cache import ExpiringLocalCache
            return ExpiringLocalCache(*args, **kwargs)
        elif cache_type == 'redis':
            from .redis_cache import RedisCache
            return RedisCache()
        else:
            raise ValueError(f'Unknown cache type: {cache_type}')

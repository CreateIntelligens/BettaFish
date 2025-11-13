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
# @Time    : 2024/6/2 11:06
# @Desc    : 抽象類

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class AbstractCache(ABC):

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        從緩存中獲取鍵的值。
        這是一個抽象方法。子類必須實現這個方法。
        :param key: 鍵
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, expire_time: int) -> None:
        """
        將鍵的值設置到緩存中。
        這是一個抽象方法。子類必須實現這個方法。
        :param key: 鍵
        :param value: 值
        :param expire_time: 過期時間
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def keys(self, pattern: str) -> List[str]:
        """
        獲取所有符合pattern的key
        :param pattern: 匹配模式
        :return:
        """
        raise NotImplementedError

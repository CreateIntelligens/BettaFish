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
# @Time    : 2023/12/3 16:20
# @Desc    :

from enum import Enum


class SearchOrderType(Enum):
    # 綜合排序
    DEFAULT = ""

    # 最多點擊
    MOST_CLICK = "click"

    # 最新發布
    LAST_PUBLISH = "pubdate"

    # 最多彈幕
    MOST_DANMU = "dm"

    # 最多收藏
    MOST_MARK = "stow"


class CommentOrderType(Enum):
    # 僅按熱度
    DEFAULT = 0

    # 按熱度+按時間
    MIXED = 1

    # 按時間
    TIME = 2

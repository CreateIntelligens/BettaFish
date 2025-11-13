# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
# 1. 不得用於任何商業用途。  
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。  
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。  
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。   
# 5. 不得用於任何非法或不當的用途。
#   
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。  
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。  


from enum import Enum


class SearchSortType(Enum):
    """search sort type"""
    # 按時間倒序
    TIME_DESC = "1"
    # 按時間順序
    TIME_ASC = "0"
    # 按相關性順序
    RELEVANCE_ORDER = "2"


class SearchNoteType(Enum):
    # 只看主題貼
    MAIN_THREAD = "1"
    # 混合模式（帖子+回覆）
    FIXED_THREAD = "0"

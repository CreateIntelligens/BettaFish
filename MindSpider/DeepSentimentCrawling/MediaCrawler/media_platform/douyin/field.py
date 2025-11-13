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


class SearchChannelType(Enum):
    """search channel type"""
    GENERAL = "aweme_general"  # 綜合
    VIDEO = "aweme_video_web"  # 視頻
    USER = "aweme_user_web"  # 用戶
    LIVE = "aweme_live"  # 直播


class SearchSortType(Enum):
    """search sort type"""
    GENERAL = 0  # 綜合排序
    MOST_LIKE = 1  # 最多點贊
    LATEST = 2  # 最新發布

class PublishTimeType(Enum):
    """publish time type"""
    UNLIMITED = 0  # 不限
    ONE_DAY = 1  # 一天內
    ONE_WEEK = 7  # 一週內
    SIX_MONTH = 180  # 半年內

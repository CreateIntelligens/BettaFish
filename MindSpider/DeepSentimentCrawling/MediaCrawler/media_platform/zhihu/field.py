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
from typing import NamedTuple

from constant import zhihu as zhihu_constant


class SearchTime(Enum):
    """
    搜索時間範圍
    """
    DEFAULT = ""  # 不限時間
    ONE_DAY = "a_day"  # 一天內
    ONE_WEEK = "a_week"  # 一週內
    ONE_MONTH = "a_month"  # 一個月內
    THREE_MONTH = "three_months"  # 三個月內
    HALF_YEAR = "half_a_year"  # 半年內
    ONE_YEAR = "a_year"  # 一年內


class SearchType(Enum):
    """
    搜索結果類型
    """
    DEFAULT = ""  # 不限類型
    ANSWER = zhihu_constant.ANSWER_NAME  # 只看回答
    ARTICLE = zhihu_constant.ARTICLE_NAME  # 只看文章
    VIDEO = zhihu_constant.VIDEO_NAME  # 只看視頻


class SearchSort(Enum):
    """
    搜索結果排序
    """
    DEFAULT = ""  # 綜合排序
    UPVOTED_COUNT = "upvoted_count"  # 最多贊同
    CREATE_TIME = "created_time"  # 最新發布

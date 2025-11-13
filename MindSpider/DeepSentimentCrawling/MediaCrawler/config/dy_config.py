# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。

# 抖音平臺配置
PUBLISH_TIME_TYPE = 0

# 指定DY視頻URL列表 (支持多種格式)
# 支持格式:
# 1. 完整視頻URL: "https://www.douyin.com/video/7525538910311632128"
# 2. 帶modal_id的URL: "https://www.douyin.com/user/xxx?modal_id=7525538910311632128"
# 3. 搜索頁帶modal_id: "https://www.douyin.com/root/search/python?modal_id=7525538910311632128"
# 4. 短鏈接: "https://v.douyin.com/drIPtQ_WPWY/"
# 5. 純視頻ID: "7280854932641664319"
DY_SPECIFIED_ID_LIST = [
    "https://www.douyin.com/video/7525538910311632128",
    "https://v.douyin.com/drIPtQ_WPWY/",
    "https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?from_tab_name=main&modal_id=7525538910311632128",
    "7202432992642387233",    
    # ........................
]

# 指定DY創作者URL列表 (支持完整URL或sec_user_id)
# 支持格式:
# 1. 完整創作者主頁URL: "https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?from_tab_name=main"
# 2. sec_user_id: "MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE"
DY_CREATOR_ID_LIST = [
    "https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?from_tab_name=main",
    "MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE"
    # ........................
]

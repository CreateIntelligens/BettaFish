# -*- coding: utf-8 -*-
# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。
# bilili 平臺配置

# 每天爬取視頻/帖子的數量控制
MAX_NOTES_PER_DAY = 1

# 指定B站視頻URL列表 (支持完整URL或BV號)
# 示例:
# - 完整URL: "https://www.bilibili.com/video/BV1dwuKzmE26/?spm_id_from=333.1387.homepage.video_card.click"
# - BV號: "BV1d54y1g7db"
BILI_SPECIFIED_ID_LIST = [
    "https://www.bilibili.com/video/BV1dwuKzmE26/?spm_id_from=333.1387.homepage.video_card.click",
    "BV1Sz4y1U77N",
    "BV14Q4y1n7jz",
    # ........................
]

# 指定B站創作者URL列表 (支持完整URL或UID)
# 示例:
# - 完整URL: "https://space.bilibili.com/434377496?spm_id_from=333.1007.0.0"
# - UID: "20813884"
BILI_CREATOR_ID_LIST = [
    "https://space.bilibili.com/434377496?spm_id_from=333.1007.0.0",
    "20813884",
    # ........................
]

# 指定時間範圍
START_DAY = "2024-01-01"
END_DAY = "2024-01-01"

# 搜索模式
BILI_SEARCH_MODE = "normal"

# 視頻清晰度（qn）配置，常見取值：
# 16=360p, 32=480p, 64=720p, 80=1080p, 112=1080p高碼率, 116=1080p60, 120=4K
# 注意：更高清晰度需要賬號/視頻本身支持
BILI_QN = 80

# 是否爬取用戶信息
CREATOR_MODE = True

# 開始爬取用戶信息頁碼
START_CONTACTS_PAGE = 1

# 單個視頻/帖子最大爬取評論數
CRAWLER_MAX_CONTACTS_COUNT_SINGLENOTES = 100

# 單個視頻/帖子最大爬取動態數
CRAWLER_MAX_DYNAMICS_COUNT_SINGLENOTES = 50

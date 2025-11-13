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


# 小紅書平臺配置

# 排序方式，具體的枚舉值在media_platform/xhs/field.py中
SORT_TYPE = "popularity_descending"

# 指定筆記URL列表, 必須要攜帶xsec_token參數
XHS_SPECIFIED_NOTE_URL_LIST = [
    "https://www.xiaohongshu.com/explore/68f99f6d0000000007033fcf?xsec_token=ABZEzjuN2fPjKF9EcMsCCxfbt3IBRsFZldGFoCJbdDmXI=&xsec_source=pc_feed"
    # ........................
]

# 指定創作者URL列表 (支持完整URL或純ID)
# 支持格式:
# 1. 完整創作者主頁URL (帶xsec_token和xsec_source參數): "https://www.xiaohongshu.com/user/profile/5eb8e1d400000000010075ae?xsec_token=AB1nWBKCo1vE2HEkfoJUOi5B6BE5n7wVrbdpHoWIj5xHw=&xsec_source=pc_feed"
# 2. 純user_id: "63e36c9a000000002703502b"
XHS_CREATOR_ID_LIST = [
    "https://www.xiaohongshu.com/user/profile/5eb8e1d400000000010075ae?xsec_token=AB1nWBKCo1vE2HEkfoJUOi5B6BE5n7wVrbdpHoWIj5xHw=&xsec_source=pc_feed",
    "63e36c9a000000002703502b",    
    # ........................
]

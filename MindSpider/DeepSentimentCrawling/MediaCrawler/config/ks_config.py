# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。

# 快手平臺配置

# 指定快手視頻URL列表 (支持完整URL或純ID)
# 支持格式:
# 1. 完整視頻URL: "https://www.kuaishou.com/short-video/3x3zxz4mjrsc8ke?authorId=3x84qugg4ch9zhs&streamSource=search"
# 2. 純視頻ID: "3xf8enb8dbj6uig"
KS_SPECIFIED_ID_LIST = [
    "https://www.kuaishou.com/short-video/3x3zxz4mjrsc8ke?authorId=3x84qugg4ch9zhs&streamSource=search&area=searchxxnull&searchKey=python",
    "3xf8enb8dbj6uig",
    # ........................
]

# 指定快手創作者URL列表 (支持完整URL或純ID)
# 支持格式:
# 1. 創作者主頁URL: "https://www.kuaishou.com/profile/3x84qugg4ch9zhs"
# 2. 純user_id: "3x4sm73aye7jq7i"
KS_CREATOR_ID_LIST = [
    "https://www.kuaishou.com/profile/3x84qugg4ch9zhs",
    "3x4sm73aye7jq7i",
    # ........................
]

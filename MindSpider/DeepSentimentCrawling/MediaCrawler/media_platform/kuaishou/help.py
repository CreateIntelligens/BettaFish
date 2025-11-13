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

import re
from model.m_kuaishou import VideoUrlInfo, CreatorUrlInfo


def parse_video_info_from_url(url: str) -> VideoUrlInfo:
    """
    從快手視頻URL中解析出視頻ID
    支持以下格式:
    1. 完整視頻URL: "https://www.kuaishou.com/short-video/3x3zxz4mjrsc8ke?authorId=3x84qugg4ch9zhs&streamSource=search"
    2. 純視頻ID: "3x3zxz4mjrsc8ke"

    Args:
        url: 快手視頻鏈接或視頻ID
    Returns:
        VideoUrlInfo: 包含視頻ID的對象
    """
    # 如果不包含http且不包含kuaishou.com，認爲是純ID
    if not url.startswith("http") and "kuaishou.com" not in url:
        return VideoUrlInfo(video_id=url, url_type="normal")

    # 從標準視頻URL中提取ID: /short-video/視頻ID
    video_pattern = r'/short-video/([a-zA-Z0-9_-]+)'
    match = re.search(video_pattern, url)
    if match:
        video_id = match.group(1)
        return VideoUrlInfo(video_id=video_id, url_type="normal")

    raise ValueError(f"無法從URL中解析出視頻ID: {url}")


def parse_creator_info_from_url(url: str) -> CreatorUrlInfo:
    """
    從快手創作者主頁URL中解析出創作者ID
    支持以下格式:
    1. 創作者主頁: "https://www.kuaishou.com/profile/3x84qugg4ch9zhs"
    2. 純ID: "3x4sm73aye7jq7i"

    Args:
        url: 快手創作者主頁鏈接或user_id
    Returns:
        CreatorUrlInfo: 包含創作者ID的對象
    """
    # 如果不包含http且不包含kuaishou.com，認爲是純ID
    if not url.startswith("http") and "kuaishou.com" not in url:
        return CreatorUrlInfo(user_id=url)

    # 從創作者主頁URL中提取user_id: /profile/xxx
    user_pattern = r'/profile/([a-zA-Z0-9_-]+)'
    match = re.search(user_pattern, url)
    if match:
        user_id = match.group(1)
        return CreatorUrlInfo(user_id=user_id)

    raise ValueError(f"無法從URL中解析出創作者ID: {url}")


if __name__ == '__main__':
    # 測試視頻URL解析
    print("=== 視頻URL解析測試 ===")
    test_video_urls = [
        "https://www.kuaishou.com/short-video/3x3zxz4mjrsc8ke?authorId=3x84qugg4ch9zhs&streamSource=search&area=searchxxnull&searchKey=python",
        "3xf8enb8dbj6uig",
    ]
    for url in test_video_urls:
        try:
            result = parse_video_info_from_url(url)
            print(f"✓ URL: {url[:80]}...")
            print(f"  結果: {result}\n")
        except Exception as e:
            print(f"✗ URL: {url}")
            print(f"  錯誤: {e}\n")

    # 測試創作者URL解析
    print("=== 創作者URL解析測試 ===")
    test_creator_urls = [
        "https://www.kuaishou.com/profile/3x84qugg4ch9zhs",
        "3x4sm73aye7jq7i",
    ]
    for url in test_creator_urls:
        try:
            result = parse_creator_info_from_url(url)
            print(f"✓ URL: {url[:80]}...")
            print(f"  結果: {result}\n")
        except Exception as e:
            print(f"✗ URL: {url}")
            print(f"  錯誤: {e}\n")

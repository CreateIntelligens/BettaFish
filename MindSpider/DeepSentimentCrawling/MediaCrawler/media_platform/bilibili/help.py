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
# @Time    : 2023/12/2 23:26
# @Desc    : bilibili 請求參數簽名
# 逆向實現參考：https://socialsisteryi.github.io/bilibili-API-collect/docs/misc/sign/wbi.html#wbi%E7%AD%BE%E5%90%8D%E7%AE%97%E6%B3%95
import re
import urllib.parse
from hashlib import md5
from typing import Dict

from model.m_bilibili import VideoUrlInfo, CreatorUrlInfo
from tools import utils


class BilibiliSign:
    def __init__(self, img_key: str, sub_key: str):
        self.img_key = img_key
        self.sub_key = sub_key
        self.map_table = [
            46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
            33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
            61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
            36, 20, 34, 44, 52
        ]

    def get_salt(self) -> str:
        """
        獲取加鹽的 key
        :return:
        """
        salt = ""
        mixin_key = self.img_key + self.sub_key
        for mt in self.map_table:
            salt += mixin_key[mt]
        return salt[:32]

    def sign(self, req_data: Dict) -> Dict:
        """
        請求參數中加上當前時間戳對請求參數中的key進行字典序排序
        再將請求參數進行 url 編碼集合 salt 進行 md5 就可以生成w_rid參數了
        :param req_data:
        :return:
        """
        current_ts = utils.get_unix_timestamp()
        req_data.update({"wts": current_ts})
        req_data = dict(sorted(req_data.items()))
        req_data = {
            # 過濾 value 中的 "!'()*" 字符
            k: ''.join(filter(lambda ch: ch not in "!'()*", str(v)))
            for k, v
            in req_data.items()
        }
        query = urllib.parse.urlencode(req_data)
        salt = self.get_salt()
        wbi_sign = md5((query + salt).encode()).hexdigest()  # 計算 w_rid
        req_data['w_rid'] = wbi_sign
        return req_data


def parse_video_info_from_url(url: str) -> VideoUrlInfo:
    """
    從B站視頻URL中解析出視頻ID
    Args:
        url: B站視頻鏈接
            - https://www.bilibili.com/video/BV1dwuKzmE26/?spm_id_from=333.1387.homepage.video_card.click
            - https://www.bilibili.com/video/BV1d54y1g7db
            - BV1d54y1g7db (直接傳入BV號)
    Returns:
        VideoUrlInfo: 包含視頻ID的對象
    """
    # 如果傳入的已經是BV號,直接返回
    if url.startswith("BV"):
        return VideoUrlInfo(video_id=url)

    # 使用正則表達式提取BV號
    # 匹配 /video/BV... 或 /video/av... 格式
    bv_pattern = r'/video/(BV[a-zA-Z0-9]+)'
    match = re.search(bv_pattern, url)

    if match:
        video_id = match.group(1)
        return VideoUrlInfo(video_id=video_id)

    raise ValueError(f"無法從URL中解析出視頻ID: {url}")


def parse_creator_info_from_url(url: str) -> CreatorUrlInfo:
    """
    從B站創作者空間URL中解析出創作者ID
    Args:
        url: B站創作者空間鏈接
            - https://space.bilibili.com/434377496?spm_id_from=333.1007.0.0
            - https://space.bilibili.com/20813884
            - 434377496 (直接傳入UID)
    Returns:
        CreatorUrlInfo: 包含創作者ID的對象
    """
    # 如果傳入的已經是純數字ID,直接返回
    if url.isdigit():
        return CreatorUrlInfo(creator_id=url)

    # 使用正則表達式提取UID
    # 匹配 /space.bilibili.com/數字 格式
    uid_pattern = r'space\.bilibili\.com/(\d+)'
    match = re.search(uid_pattern, url)

    if match:
        creator_id = match.group(1)
        return CreatorUrlInfo(creator_id=creator_id)

    raise ValueError(f"無法從URL中解析出創作者ID: {url}")


if __name__ == '__main__':
    # 測試視頻URL解析
    video_url1 = "https://www.bilibili.com/video/BV1dwuKzmE26/?spm_id_from=333.1387.homepage.video_card.click"
    video_url2 = "BV1d54y1g7db"
    print("視頻URL解析測試:")
    print(f"URL1: {video_url1} -> {parse_video_info_from_url(video_url1)}")
    print(f"URL2: {video_url2} -> {parse_video_info_from_url(video_url2)}")

    # 測試創作者URL解析
    creator_url1 = "https://space.bilibili.com/434377496?spm_id_from=333.1007.0.0"
    creator_url2 = "20813884"
    print("\n創作者URL解析測試:")
    print(f"URL1: {creator_url1} -> {parse_creator_info_from_url(creator_url1)}")
    print(f"URL2: {creator_url2} -> {parse_creator_info_from_url(creator_url2)}")

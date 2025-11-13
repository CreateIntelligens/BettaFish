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
# @Name    : 程序員阿江-Relakkes
# @Time    : 2024/6/10 02:24
# @Desc    : 獲取 a_bogus 參數, 學習交流使用，請勿用作商業用途，侵權聯繫作者刪除

import random
import re
from typing import Optional

import execjs
from playwright.async_api import Page

from model.m_douyin import VideoUrlInfo, CreatorUrlInfo
from tools.crawler_util import extract_url_params_to_dict

douyin_sign_obj = execjs.compile(open('libs/douyin.js', encoding='utf-8-sig').read())

def get_web_id():
    """
    生成隨機的webid
    Returns:

    """

    def e(t):
        if t is not None:
            return str(t ^ (int(16 * random.random()) >> (t // 4)))
        else:
            return ''.join(
                [str(int(1e7)), '-', str(int(1e3)), '-', str(int(4e3)), '-', str(int(8e3)), '-', str(int(1e11))]
            )

    web_id = ''.join(
        e(int(x)) if x in '018' else x for x in e(None)
    )
    return web_id.replace('-', '')[:19]



async def get_a_bogus(url: str, params: str, post_data: dict, user_agent: str, page: Page = None):
    """
    獲取 a_bogus 參數, 目前不支持post請求類型的簽名
    """
    return get_a_bogus_from_js(url, params, user_agent)

def get_a_bogus_from_js(url: str, params: str, user_agent: str):
    """
    通過js獲取 a_bogus 參數
    Args:
        url:
        params:
        user_agent:

    Returns:

    """
    sign_js_name = "sign_datail"
    if "/reply" in url:
        sign_js_name = "sign_reply"
    return douyin_sign_obj.call(sign_js_name, params, user_agent)



async def get_a_bogus_from_playright(params: str, post_data: dict, user_agent: str, page: Page):
    """
    通過playright獲取 a_bogus 參數
    playwright版本已失效
    Returns:

    """
    if not post_data:
        post_data = ""
    a_bogus = await page.evaluate(
        "([params, post_data, ua]) => window.bdms.init._v[2].p[42].apply(null, [0, 1, 8, params, post_data, ua])",
        [params, post_data, user_agent])

    return a_bogus


def parse_video_info_from_url(url: str) -> VideoUrlInfo:
    """
    從抖音視頻URL中解析出視頻ID
    支持以下格式:
    1. 普通視頻鏈接: https://www.douyin.com/video/7525082444551310602
    2. 帶modal_id參數的鏈接:
       - https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?modal_id=7525082444551310602
       - https://www.douyin.com/root/search/python?modal_id=7471165520058862848
    3. 短鏈接: https://v.douyin.com/iF12345ABC/ (需要client解析)
    4. 純ID: 7525082444551310602

    Args:
        url: 抖音視頻鏈接或ID
    Returns:
        VideoUrlInfo: 包含視頻ID的對象
    """
    # 如果是純數字ID,直接返回
    if url.isdigit():
        return VideoUrlInfo(aweme_id=url, url_type="normal")

    # 檢查是否是短鏈接 (v.douyin.com)
    if "v.douyin.com" in url or url.startswith("http") and len(url) < 50 and "video" not in url:
        return VideoUrlInfo(aweme_id="", url_type="short")  # 需要通過client解析

    # 嘗試從URL參數中提取modal_id
    params = extract_url_params_to_dict(url)
    modal_id = params.get("modal_id")
    if modal_id:
        return VideoUrlInfo(aweme_id=modal_id, url_type="modal")

    # 從標準視頻URL中提取ID: /video/數字
    video_pattern = r'/video/(\d+)'
    match = re.search(video_pattern, url)
    if match:
        aweme_id = match.group(1)
        return VideoUrlInfo(aweme_id=aweme_id, url_type="normal")

    raise ValueError(f"無法從URL中解析出視頻ID: {url}")


def parse_creator_info_from_url(url: str) -> CreatorUrlInfo:
    """
    從抖音創作者主頁URL中解析出創作者ID (sec_user_id)
    支持以下格式:
    1. 創作者主頁: https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?from_tab_name=main
    2. 純ID: MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE

    Args:
        url: 抖音創作者主頁鏈接或sec_user_id
    Returns:
        CreatorUrlInfo: 包含創作者ID的對象
    """
    # 如果是純ID格式(通常以MS4wLjABAAAA開頭),直接返回
    if url.startswith("MS4wLjABAAAA") or (not url.startswith("http") and "douyin.com" not in url):
        return CreatorUrlInfo(sec_user_id=url)

    # 從創作者主頁URL中提取sec_user_id: /user/xxx
    user_pattern = r'/user/([^/?]+)'
    match = re.search(user_pattern, url)
    if match:
        sec_user_id = match.group(1)
        return CreatorUrlInfo(sec_user_id=sec_user_id)

    raise ValueError(f"無法從URL中解析出創作者ID: {url}")


if __name__ == '__main__':
    # 測試視頻URL解析
    print("=== 視頻URL解析測試 ===")
    test_urls = [
        "https://www.douyin.com/video/7525082444551310602",
        "https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?from_tab_name=main&modal_id=7525082444551310602",
        "https://www.douyin.com/root/search/python?aid=b733a3b0-4662-4639-9a72-c2318fba9f3f&modal_id=7471165520058862848&type=general",
        "7525082444551310602",
    ]
    for url in test_urls:
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
        "https://www.douyin.com/user/MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE?from_tab_name=main",
        "MS4wLjABAAAATJPY7LAlaa5X-c8uNdWkvz0jUGgpw4eeXIwu_8BhvqE",
    ]
    for url in test_creator_urls:
        try:
            result = parse_creator_info_from_url(url)
            print(f"✓ URL: {url[:80]}...")
            print(f"  結果: {result}\n")
        except Exception as e:
            print(f"✗ URL: {url}")
            print(f"  錯誤: {e}\n")


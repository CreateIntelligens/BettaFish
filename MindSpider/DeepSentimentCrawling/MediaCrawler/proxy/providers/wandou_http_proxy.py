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
# @Time    : 2025/7/31
# @Desc    : 豌豆HTTP 代理IP實現
import os
from typing import Dict, List
from urllib.parse import urlencode

import httpx

from proxy import IpCache, IpGetError, ProxyProvider
from proxy.types import IpInfoModel
from tools import utils


class WanDouHttpProxy(ProxyProvider):

    def __init__(self, app_key: str, num: int = 100):
        """
        豌豆HTTP 代理IP實現
        :param app_key: 開放的app_key,可以通過用戶中心獲取
        :param num: 單次提取IP數量,最大100
        """
        self.proxy_brand_name = "WANDOUHTTP"
        self.api_path = "https://api.wandouapp.com/"
        self.params = {
            "app_key": app_key,
            "num": num,
        }
        self.ip_cache = IpCache()

    async def get_proxy(self, num: int) -> List[IpInfoModel]:
        """
        :param num:
        :return:
        """

        # 優先從緩存中拿 IP
        ip_cache_list = self.ip_cache.load_all_ip(
            proxy_brand_name=self.proxy_brand_name
        )
        if len(ip_cache_list) >= num:
            return ip_cache_list[:num]

        # 如果緩存中的數量不夠，從IP代理商獲取補上，再存入緩存中
        need_get_count = num - len(ip_cache_list)
        self.params.update({"num": min(need_get_count, 100)})  # 最大100
        ip_infos = []
        async with httpx.AsyncClient() as client:
            url = self.api_path + "?" + urlencode(self.params)
            utils.logger.info(f"[WanDouHttpProxy.get_proxy] get ip proxy url:{url}")
            response = await client.get(
                url,
                headers={
                    "User-Agent": "MediaCrawler https://github.com/NanmiCoder/MediaCrawler",
                },
            )
            res_dict: Dict = response.json()
            if res_dict.get("code") == 200:
                data: List[Dict] = res_dict.get("data", [])
                current_ts = utils.get_unix_timestamp()
                for ip_item in data:
                    ip_info_model = IpInfoModel(
                        ip=ip_item.get("ip"),
                        port=ip_item.get("port"),
                        user="",  # 豌豆HTTP不需要用戶名密碼認證
                        password="",
                        expired_time_ts=utils.get_unix_time_from_time_str(
                            ip_item.get("expire_time")
                        ),
                    )
                    ip_key = f"WANDOUHTTP_{ip_info_model.ip}_{ip_info_model.port}"
                    ip_value = ip_info_model.model_dump_json()
                    ip_infos.append(ip_info_model)
                    self.ip_cache.set_ip(
                        ip_key, ip_value, ex=ip_info_model.expired_time_ts - current_ts
                    )
            else:
                error_msg = res_dict.get("msg", "unknown error")
                # 處理具體錯誤碼
                error_code = res_dict.get("code")
                if error_code == 10001:
                    error_msg = "通用錯誤，具體錯誤信息查看msg內容"
                elif error_code == 10048:
                    error_msg = "沒有可用套餐"
                raise IpGetError(f"{error_msg} (code: {error_code})")
        return ip_cache_list + ip_infos


def new_wandou_http_proxy() -> WanDouHttpProxy:
    """
    構造豌豆HTTP實例
    Returns:

    """
    return WanDouHttpProxy(
        app_key=os.getenv(
            "wandou_app_key", "你的豌豆HTTP app_key"
        ),  # 通過環境變量的方式獲取豌豆HTTP app_key
    )

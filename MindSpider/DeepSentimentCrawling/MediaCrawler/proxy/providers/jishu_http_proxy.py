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
# @Time    : 2024/4/5 09:32
# @Desc    : 已廢棄！！！！！倒閉了！！！極速HTTP 代理IP實現. 請使用快代理實現（proxy/providers/kuaidl_proxy.py）
import os
from typing import Dict, List
from urllib.parse import urlencode

import httpx

from proxy import IpCache, IpGetError, ProxyProvider
from proxy.types import IpInfoModel
from tools import utils


class JiSuHttpProxy(ProxyProvider):

    def __init__(self, key: str, crypto: str, time_validity_period: int):
        """
        極速HTTP 代理IP實現
        :param key: 提取key值 (去官網註冊後獲取)
        :param crypto: 加密簽名 (去官網註冊後獲取)
        """
        self.proxy_brand_name = "JISUHTTP"
        self.api_path = "https://api.jisuhttp.com"
        self.params = {
            "key": key,
            "crypto": crypto,
            "time": time_validity_period,  # IP使用時長，支持3、5、10、15、30分鐘時效
            "type": "json",  # 數據結果爲json
            "port": "2",  # IP協議：1:HTTP、2:HTTPS、3:SOCKS5
            "pw": "1",  # 是否使用賬密驗證， 1：是，0：否，否表示白名單驗證；默認爲0
            "se": "1",  # 返回JSON格式時是否顯示IP過期時間， 1：顯示，0：不顯示；默認爲0
        }
        self.ip_cache = IpCache()

    async def get_proxy(self, num: int) -> List[IpInfoModel]:
        """
        :param num:
        :return:
        """

        # 優先從緩存中拿 IP
        ip_cache_list = self.ip_cache.load_all_ip(proxy_brand_name=self.proxy_brand_name)
        if len(ip_cache_list) >= num:
            return ip_cache_list[:num]

        # 如果緩存中的數量不夠，從IP代理商獲取補上，再存入緩存中
        need_get_count = num - len(ip_cache_list)
        self.params.update({"num": need_get_count})
        ip_infos = []
        async with httpx.AsyncClient() as client:
            url = self.api_path + "/fetchips" + '?' + urlencode(self.params)
            utils.logger.info(f"[JiSuHttpProxy.get_proxy] get ip proxy url:{url}")
            response = await client.get(url, headers={
                "User-Agent": "MediaCrawler https://github.com/NanmiCoder/MediaCrawler",
            })
            res_dict: Dict = response.json()
            if res_dict.get("code") == 0:
                data: List[Dict] = res_dict.get("data")
                current_ts = utils.get_unix_timestamp()
                for ip_item in data:
                    ip_info_model = IpInfoModel(
                        ip=ip_item.get("ip"),
                        port=ip_item.get("port"),
                        user=ip_item.get("user"),
                        password=ip_item.get("pass"),
                        expired_time_ts=utils.get_unix_time_from_time_str(ip_item.get("expire")),
                    )
                    ip_key = f"JISUHTTP_{ip_info_model.ip}_{ip_info_model.port}_{ip_info_model.user}_{ip_info_model.password}"
                    ip_value = ip_info_model.json()
                    ip_infos.append(ip_info_model)
                    self.ip_cache.set_ip(ip_key, ip_value, ex=ip_info_model.expired_time_ts - current_ts)
            else:
                raise IpGetError(res_dict.get("msg", "unkown err"))
        return ip_cache_list + ip_infos


def new_jisu_http_proxy() -> JiSuHttpProxy:
    """
    構造極速HTTP實例
    Returns:

    """
    return JiSuHttpProxy(
        key=os.getenv("jisu_key", ""),  # 通過環境變量的方式獲取極速HTTPIP提取key值
        crypto=os.getenv("jisu_crypto", ""),  # 通過環境變量的方式獲取極速HTTPIP提取加密簽名
        time_validity_period=30  # 30分鐘（最長時效）
    )

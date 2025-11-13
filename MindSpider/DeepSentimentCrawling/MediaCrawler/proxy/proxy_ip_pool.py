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
# @Time    : 2023/12/2 13:45
# @Desc    : ip代理池實現
import random
from typing import Dict, List

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

import config
from proxy.providers import (
    new_kuai_daili_proxy,
    new_wandou_http_proxy,
)
from tools import utils

from .base_proxy import ProxyProvider
from .types import IpInfoModel, ProviderNameEnum


class ProxyIpPool:

    def __init__(
        self, ip_pool_count: int, enable_validate_ip: bool, ip_provider: ProxyProvider
    ) -> None:
        """

        Args:
            ip_pool_count:
            enable_validate_ip:
            ip_provider:
        """
        self.valid_ip_url = "https://echo.apifox.cn/"  # 驗證 IP 是否有效的地址
        self.ip_pool_count = ip_pool_count
        self.enable_validate_ip = enable_validate_ip
        self.proxy_list: List[IpInfoModel] = []
        self.ip_provider: ProxyProvider = ip_provider

    async def load_proxies(self) -> None:
        """
        加載IP代理
        Returns:

        """
        self.proxy_list = await self.ip_provider.get_proxy(self.ip_pool_count)

    async def _is_valid_proxy(self, proxy: IpInfoModel) -> bool:
        """
        驗證代理IP是否有效
        :param proxy:
        :return:
        """
        utils.logger.info(
            f"[ProxyIpPool._is_valid_proxy] testing {proxy.ip} is it valid "
        )
        try:
            # httpx 0.28.1 需要直接傳入代理URL字符串，而不是字典
            if proxy.user and proxy.password:
                proxy_url = f"http://{proxy.user}:{proxy.password}@{proxy.ip}:{proxy.port}"
            else:
                proxy_url = f"http://{proxy.ip}:{proxy.port}"
            
            async with httpx.AsyncClient(proxy=proxy_url) as client:
                response = await client.get(self.valid_ip_url)
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            utils.logger.info(
                f"[ProxyIpPool._is_valid_proxy] testing {proxy.ip} err: {e}"
            )
            raise e

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def get_proxy(self) -> IpInfoModel:
        """
        從代理池中隨機提取一個代理IP
        :return:
        """
        if len(self.proxy_list) == 0:
            await self._reload_proxies()

        proxy = random.choice(self.proxy_list)
        self.proxy_list.remove(proxy)  # 取出來一個IP就應該移出掉
        if self.enable_validate_ip:
            if not await self._is_valid_proxy(proxy):
                raise Exception(
                    "[ProxyIpPool.get_proxy] current ip invalid and again get it"
                )
        return proxy

    async def _reload_proxies(self):
        """
        # 重新加載代理池
        :return:
        """
        self.proxy_list = []
        await self.load_proxies()


IpProxyProvider: Dict[str, ProxyProvider] = {
    ProviderNameEnum.KUAI_DAILI_PROVIDER.value: new_kuai_daili_proxy(),
    ProviderNameEnum.WANDOU_HTTP_PROVIDER.value: new_wandou_http_proxy(),
}


async def create_ip_pool(ip_pool_count: int, enable_validate_ip: bool) -> ProxyIpPool:
    """
     創建 IP 代理池
    :param ip_pool_count: ip池子的數量
    :param enable_validate_ip: 是否開啓驗證IP代理
    :return:
    """
    pool = ProxyIpPool(
        ip_pool_count=ip_pool_count,
        enable_validate_ip=enable_validate_ip,
        ip_provider=IpProxyProvider.get(config.IP_PROXY_PROVIDER_NAME),
    )
    await pool.load_proxies()
    return pool


if __name__ == "__main__":
    pass

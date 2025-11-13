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
# @Time    : 2023/12/2 14:42
# @Desc    :
from unittest import IsolatedAsyncioTestCase

from proxy.proxy_ip_pool import create_ip_pool
from proxy.types import IpInfoModel


class TestIpPool(IsolatedAsyncioTestCase):
    async def test_ip_pool(self):
        pool = await create_ip_pool(ip_pool_count=1, enable_validate_ip=True)
        print("\n")
        for i in range(3):
            ip_proxy_info: IpInfoModel = await pool.get_proxy()
            print(ip_proxy_info)
            self.assertIsNotNone(ip_proxy_info.ip, msg="驗證 ip 是否獲取成功")


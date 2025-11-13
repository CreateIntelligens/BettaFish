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
# @Time    : 2024/4/5 10:18
# @Desc    : 基礎類型
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProviderNameEnum(Enum):
    KUAI_DAILI_PROVIDER: str = "kuaidaili"
    WANDOU_HTTP_PROVIDER: str = "wandouhttp"


class IpInfoModel(BaseModel):
    """Unified IP model"""

    ip: str = Field(title="ip")
    port: int = Field(title="端口")
    user: str = Field(title="IP代理認證的用戶名")
    protocol: str = Field(default="https://", title="代理IP的協議")
    password: str = Field(title="IP代理認證用戶的密碼")
    expired_time_ts: Optional[int] = Field(title="IP 過期時間")

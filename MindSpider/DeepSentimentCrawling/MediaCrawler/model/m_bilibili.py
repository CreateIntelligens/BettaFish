# 聲明:本代碼僅供學習和研究目的使用。使用者應遵守以下原則:
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率,避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。


# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field


class VideoUrlInfo(BaseModel):
    """B站視頻URL信息"""
    video_id: str = Field(title="video id (BV id)")
    video_type: str = Field(default="video", title="video type")


class CreatorUrlInfo(BaseModel):
    """B站創作者URL信息"""
    creator_id: str = Field(title="creator id (UID)")

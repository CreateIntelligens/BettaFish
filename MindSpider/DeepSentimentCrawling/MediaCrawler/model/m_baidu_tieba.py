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
from typing import Optional

from pydantic import BaseModel, Field


class TiebaNote(BaseModel):
    """
    百度貼吧帖子
    """
    note_id: str = Field(..., description="帖子ID")
    title: str = Field(..., description="帖子標題")
    desc: str = Field(default="", description="帖子描述")
    note_url: str = Field(..., description="帖子鏈接")
    publish_time: str = Field(default="", description="發佈時間")
    user_link: str = Field(default="", description="用戶主頁鏈接")
    user_nickname: str = Field(default="", description="用戶暱稱")
    user_avatar: str = Field(default="", description="用戶頭像地址")
    tieba_name: str = Field(..., description="貼吧名稱")
    tieba_link: str = Field(..., description="貼吧鏈接")
    total_replay_num: int = Field(default=0, description="回覆總數")
    total_replay_page: int = Field(default=0, description="回覆總頁數")
    ip_location: Optional[str] = Field(default="", description="IP地理位置")
    source_keyword: str = Field(default="", description="來源關鍵詞")


class TiebaComment(BaseModel):
    """
    百度貼吧評論
    """

    comment_id: str = Field(..., description="評論ID")
    parent_comment_id: str = Field(default="", description="父評論ID")
    content: str = Field(..., description="評論內容")
    user_link: str = Field(default="", description="用戶主頁鏈接")
    user_nickname: str = Field(default="", description="用戶暱稱")
    user_avatar: str = Field(default="", description="用戶頭像地址")
    publish_time: str = Field(default="", description="發佈時間")
    ip_location: Optional[str] = Field(default="", description="IP地理位置")
    sub_comment_count: int = Field(default=0, description="子評論數")
    note_id: str = Field(..., description="帖子ID")
    note_url: str = Field(..., description="帖子鏈接")
    tieba_id: str = Field(..., description="所屬的貼吧ID")
    tieba_name: str = Field(..., description="所屬的貼吧名稱")
    tieba_link: str = Field(..., description="貼吧鏈接")


class TiebaCreator(BaseModel):
    """
    百度貼吧創作者
    """
    user_id: str = Field(..., description="用戶ID")
    user_name: str = Field(..., description="用戶名")
    nickname: str = Field(..., description="用戶暱稱")
    gender: str = Field(default="", description="用戶性別")
    avatar: str = Field(..., description="用戶頭像地址")
    ip_location: Optional[str] = Field(default="", description="IP地理位置")
    follows: int = Field(default=0, description="關注數")
    fans: int = Field(default=0, description="粉絲數")
    registration_duration: str = Field(default="", description="註冊時長")

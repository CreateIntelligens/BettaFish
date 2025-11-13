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


class ZhihuContent(BaseModel):
    """
    知乎內容（回答、文章、視頻）
    """
    content_id: str = Field(default="", description="內容ID")
    content_type: str = Field(default="", description="內容類型(article | answer | zvideo)")
    content_text: str = Field(default="", description="內容文本, 如果是視頻類型這裏爲空")
    content_url: str = Field(default="", description="內容落地鏈接")
    question_id: str = Field(default="", description="問題ID, type爲answer時有值")
    title: str = Field(default="", description="內容標題")
    desc: str = Field(default="", description="內容描述")
    created_time: int = Field(default=0, description="創建時間")
    updated_time: int = Field(default=0, description="更新時間")
    voteup_count: int = Field(default=0, description="贊同人數")
    comment_count: int = Field(default=0, description="評論數量")
    source_keyword: str = Field(default="", description="來源關鍵詞")

    user_id: str = Field(default="", description="用戶ID")
    user_link: str = Field(default="", description="用戶主頁鏈接")
    user_nickname: str = Field(default="", description="用戶暱稱")
    user_avatar: str = Field(default="", description="用戶頭像地址")
    user_url_token: str = Field(default="", description="用戶url_token")


class ZhihuComment(BaseModel):
    """
    知乎評論
    """

    comment_id: str = Field(default="", description="評論ID")
    parent_comment_id: str = Field(default="", description="父評論ID")
    content: str = Field(default="", description="評論內容")
    publish_time: int = Field(default=0, description="發佈時間")
    ip_location: Optional[str] = Field(default="", description="IP地理位置")
    sub_comment_count: int = Field(default=0, description="子評論數")
    like_count: int = Field(default=0, description="點贊數")
    dislike_count: int = Field(default=0, description="踩數")
    content_id: str = Field(default="", description="內容ID")
    content_type: str = Field(default="", description="內容類型(article | answer | zvideo)")

    user_id: str = Field(default="", description="用戶ID")
    user_link: str = Field(default="", description="用戶主頁鏈接")
    user_nickname: str = Field(default="", description="用戶暱稱")
    user_avatar: str = Field(default="", description="用戶頭像地址")


class ZhihuCreator(BaseModel):
    """
    知乎創作者
    """
    user_id: str = Field(default="", description="用戶ID")
    user_link: str = Field(default="", description="用戶主頁鏈接")
    user_nickname: str = Field(default="", description="用戶暱稱")
    user_avatar: str = Field(default="", description="用戶頭像地址")
    url_token: str = Field(default="", description="用戶url_token")
    gender: str = Field(default="", description="用戶性別")
    ip_location: Optional[str] = Field(default="", description="IP地理位置")
    follows: int = Field(default=0, description="關注數")
    fans: int = Field(default=0, description="粉絲數")
    anwser_count: int = Field(default=0, description="回答數")
    video_count: int = Field(default=0, description="視頻數")
    question_count: int = Field(default=0, description="提問數")
    article_count: int = Field(default=0, description="文章數")
    column_count: int = Field(default=0, description="專欄數")
    get_voteup_count: int = Field(default=0, description="獲得的贊同數")


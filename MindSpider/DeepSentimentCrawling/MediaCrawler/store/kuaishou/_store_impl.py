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
# @Author  : persist1@126.com
# @Time    : 2025/9/5 19:34
# @Desc    : 快手存儲實現類
import asyncio
import csv
import json
import os
import pathlib
from typing import Dict
from tools.async_file_writer import AsyncFileWriter

import aiofiles
from sqlalchemy import select

import config
from base.base_crawler import AbstractStore
from database.db_session import get_session
from database.models import KuaishouVideo, KuaishouVideoComment
from tools import utils, words
from var import crawler_type_var


def calculate_number_of_files(file_store_path: str) -> int:
    """計算數據保存文件的前部分排序數字，支持每次運行代碼不寫到同一個文件中
    Args:
        file_store_path;
    Returns:
        file nums
    """
    if not os.path.exists(file_store_path):
        return 1
    try:
        return max([int(file_name.split("_")[0]) for file_name in os.listdir(file_store_path)]) + 1
    except ValueError:
        return 1


class KuaishouCsvStoreImplement(AbstractStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = AsyncFileWriter(platform="kuaishou", crawler_type=crawler_type_var.get())

    async def store_content(self, content_item: Dict):
        """
        Kuaishou content CSV storage implementation
        Args:
            content_item: note item dict

        Returns:

        """
        await self.writer.write_to_csv(item_type="contents", item=content_item)

    async def store_comment(self, comment_item: Dict):
        """
        Kuaishou comment CSV storage implementation
        Args:
            comment_item: comment item dict

        Returns:

        """
        await self.writer.write_to_csv(item_type="comments", item=comment_item)

    async def store_creator(self, creator: Dict):
        pass


class KuaishouDbStoreImplement(AbstractStore):
    async def store_creator(self, creator: Dict):
        pass

    async def store_content(self, content_item: Dict):
        """
        Kuaishou content DB storage implementation
        Args:
            content_item: content item dict
        """
        video_id = content_item.get("video_id")
        async with get_session() as session:
            result = await session.execute(select(KuaishouVideo).where(KuaishouVideo.video_id == video_id))
            video_detail = result.scalar_one_or_none()

            if not video_detail:
                content_item["add_ts"] = utils.get_current_timestamp()
                new_content = KuaishouVideo(**content_item)
                session.add(new_content)
            else:
                for key, value in content_item.items():
                    setattr(video_detail, key, value)
            await session.commit()

    async def store_comment(self, comment_item: Dict):
        """
        Kuaishou comment DB storage implementation
        Args:
            comment_item: comment item dict
        """
        comment_id = comment_item.get("comment_id")
        async with get_session() as session:
            result = await session.execute(
                select(KuaishouVideoComment).where(KuaishouVideoComment.comment_id == comment_id))
            comment_detail = result.scalar_one_or_none()

            if not comment_detail:
                comment_item["add_ts"] = utils.get_current_timestamp()
                new_comment = KuaishouVideoComment(**comment_item)
                session.add(new_comment)
            else:
                for key, value in comment_item.items():
                    setattr(comment_detail, key, value)
            await session.commit()


class KuaishouJsonStoreImplement(AbstractStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = AsyncFileWriter(platform="kuaishou", crawler_type=crawler_type_var.get())

    async def store_content(self, content_item: Dict):
        """
        content JSON storage implementation
        Args:
            content_item:

        Returns:

        """
        await self.writer.write_single_item_to_json(item_type="contents", item=content_item)

    async def store_comment(self, comment_item: Dict):
        """
        comment JSON storage implementation
        Args:
            comment_item:

        Returns:

        """
        await self.writer.write_single_item_to_json(item_type="comments", item=comment_item)

    async def store_creator(self, creator: Dict):
        pass


class KuaishouSqliteStoreImplement(KuaishouDbStoreImplement):
    async def store_creator(self, creator: Dict):
        pass
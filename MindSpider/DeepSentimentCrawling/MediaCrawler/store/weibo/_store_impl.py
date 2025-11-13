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
# @Desc    : 微博存儲實現類
import asyncio
import csv
import json
import os
import pathlib
from typing import Dict

import aiofiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import config
from base.base_crawler import AbstractStore
from database.models import WeiboCreator, WeiboNote, WeiboNoteComment
from tools import utils, words
from tools.async_file_writer import AsyncFileWriter
from database.db_session import get_session
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


class WeiboCsvStoreImplement(AbstractStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = AsyncFileWriter(platform="weibo", crawler_type=crawler_type_var.get())

    async def store_content(self, content_item: Dict):
        """
        Weibo content CSV storage implementation
        Args:
            content_item: note item dict

        Returns:

        """
        await self.writer.write_to_csv(item_type="contents", item=content_item)

    async def store_comment(self, comment_item: Dict):
        """
        Weibo comment CSV storage implementation
        Args:
            comment_item: comment item dict

        Returns:

        """
        await self.writer.write_to_csv(item_type="comments", item=comment_item)

    async def store_creator(self, creator: Dict):
        """
        Weibo creator CSV storage implementation
        Args:
            creator:

        Returns:

        """
        await self.writer.write_to_csv(item_type="creators", item=creator)


class WeiboDbStoreImplement(AbstractStore):

    async def store_content(self, content_item: Dict):
        """
        Weibo content DB storage implementation
        Args:
            content_item: content item dict

        Returns:

        """
        note_id = content_item.get("note_id")
        async with get_session() as session:
            stmt = select(WeiboNote).where(WeiboNote.note_id == note_id)
            res = await session.execute(stmt)
            db_note = res.scalar_one_or_none()
            if db_note:
                db_note.last_modify_ts = utils.get_current_timestamp()
                for key, value in content_item.items():
                    if hasattr(db_note, key):
                        setattr(db_note, key, value)
            else:
                content_item["add_ts"] = utils.get_current_timestamp()
                content_item["last_modify_ts"] = utils.get_current_timestamp()
                db_note = WeiboNote(**content_item)
                session.add(db_note)
            await session.commit()

    async def store_comment(self, comment_item: Dict):
        """
        Weibo content DB storage implementation
        Args:
            comment_item: comment item dict

        Returns:

        """
        comment_id = comment_item.get("comment_id")
        async with get_session() as session:
            stmt = select(WeiboNoteComment).where(WeiboNoteComment.comment_id == comment_id)
            res = await session.execute(stmt)
            db_comment = res.scalar_one_or_none()
            if db_comment:
                db_comment.last_modify_ts = utils.get_current_timestamp()
                for key, value in comment_item.items():
                    if hasattr(db_comment, key):
                        setattr(db_comment, key, value)
            else:
                comment_item["add_ts"] = utils.get_current_timestamp()
                comment_item["last_modify_ts"] = utils.get_current_timestamp()
                db_comment = WeiboNoteComment(**comment_item)
                session.add(db_comment)
            await session.commit()

    async def store_creator(self, creator: Dict):
        """
        Weibo creator DB storage implementation
        Args:
            creator:

        Returns:

        """
        user_id = creator.get("user_id")
        async with get_session() as session:
            stmt = select(WeiboCreator).where(WeiboCreator.user_id == user_id)
            res = await session.execute(stmt)
            db_creator = res.scalar_one_or_none()
            if db_creator:
                db_creator.last_modify_ts = utils.get_current_timestamp()
                for key, value in creator.items():
                    if hasattr(db_creator, key):
                        setattr(db_creator, key, value)
            else:
                creator["add_ts"] = utils.get_current_timestamp()
                creator["last_modify_ts"] = utils.get_current_timestamp()
                db_creator = WeiboCreator(**creator)
                session.add(db_creator)
            await session.commit()


class WeiboJsonStoreImplement(AbstractStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = AsyncFileWriter(platform="weibo", crawler_type=crawler_type_var.get())

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
        """
        creator JSON storage implementation
        Args:
            creator:

        Returns:

        """
        await self.writer.write_single_item_to_json(item_type="creators", item=creator)


class WeiboSqliteStoreImplement(WeiboDbStoreImplement):
    """
    Weibo content SQLite storage implementation
    """
    pass

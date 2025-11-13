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
# @Author  : helloteemo
# @Time    : 2024/7/12 20:01
# @Desc    : bilibili 媒體保存
import pathlib
from typing import Dict

import aiofiles

from base.base_crawler import AbstractStoreImage, AbstractStoreVideo
from tools import utils


class BilibiliVideo(AbstractStoreVideo):
    video_store_path: str = "data/bili/videos"

    async def store_video(self, video_content_item: Dict):
        """
        store content
        
        Args:
            video_content_item:

        Returns:

        """
        await self.save_video(video_content_item.get("aid"), video_content_item.get("video_content"), video_content_item.get("extension_file_name"))

    def make_save_file_name(self, aid: str, extension_file_name: str) -> str:
        """
        make save file name by store type
        
        Args:
            aid: aid
            extension_file_name: video filename with extension
            
        Returns:

        """
        return f"{self.video_store_path}/{aid}/{extension_file_name}"

    async def save_video(self, aid: int, video_content: str, extension_file_name="mp4"):
        """
        save video to local
        
        Args:
            aid: aid
            video_content: video content
            extension_file_name: video filename with extension

        Returns:

        """
        pathlib.Path(self.video_store_path + "/" + str(aid)).mkdir(parents=True, exist_ok=True)
        save_file_name = self.make_save_file_name(str(aid), extension_file_name)
        async with aiofiles.open(save_file_name, 'wb') as f:
            await f.write(video_content)
            utils.logger.info(f"[BilibiliVideoImplement.save_video] save save_video {save_file_name} success ...")

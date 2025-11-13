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
# @Author  : Erm
# @Time    : 2024/4/9 17:35
# @Desc    : 微博媒體保存
import pathlib
from typing import Dict

import aiofiles

from base.base_crawler import AbstractStoreImage, AbstractStoreVideo
from tools import utils


class WeiboStoreImage(AbstractStoreImage):
    image_store_path: str = "data/weibo/images"

    async def store_image(self, image_content_item: Dict):
        """
        store content
        
        Args:
            image_content_item:

        Returns:

        """
        await self.save_image(image_content_item.get("pic_id"), image_content_item.get("pic_content"), image_content_item.get("extension_file_name"))

    def make_save_file_name(self, picid: str, extension_file_name: str) -> str:
        """
        make save file name by store type
        
        Args:
            picid: image id
            extension_file_name: video filename with extension

        Returns:

        """
        return f"{self.image_store_path}/{picid}.{extension_file_name}"

    async def save_image(self, picid: str, pic_content: str, extension_file_name="jpg"):
        """
        save image to local
        
        Args:
            picid: image id
            pic_content: image content
            extension_file_name: image filename with extension

        Returns:

        """
        pathlib.Path(self.image_store_path).mkdir(parents=True, exist_ok=True)
        save_file_name = self.make_save_file_name(picid, extension_file_name)
        async with aiofiles.open(save_file_name, 'wb') as f:
            await f.write(pic_content)
            utils.logger.info(f"[WeiboImageStoreImplement.save_image] save image {save_file_name} success ...")

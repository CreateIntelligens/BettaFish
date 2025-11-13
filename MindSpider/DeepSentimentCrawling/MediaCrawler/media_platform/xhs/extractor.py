# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。

import json
import re
from typing import Dict, Optional

import humps


class XiaoHongShuExtractor:
    def __init__(self):
        pass

    def extract_note_detail_from_html(self, note_id: str, html: str) -> Optional[Dict]:
        """從html中提取筆記詳情

        Args:
            html (str): html字符串

        Returns:
            Dict: 筆記詳情字典
        """
        if "noteDetailMap" not in html:
            # 這種情況要麼是出了驗證碼了，要麼是筆記不存在
            return None

        state = re.findall(r"window.__INITIAL_STATE__=({.*})</script>", html)[
            0
        ].replace("undefined", '""')
        if state != "{}":
            note_dict = humps.decamelize(json.loads(state))
            return note_dict["note"]["note_detail_map"][note_id]["note"]
        return None

    def extract_creator_info_from_html(self, html: str) -> Optional[Dict]:
        """從html中提取用戶信息

        Args:
            html (str): html字符串

        Returns:
            Dict: 用戶信息字典
        """
        match = re.search(
            r"<script>window.__INITIAL_STATE__=(.+)<\/script>", html, re.M
        )
        if match is None:
            return None
        info = json.loads(match.group(1).replace(":undefined", ":null"), strict=False)
        if info is None:
            return None
        return info.get("user").get("userPageData")

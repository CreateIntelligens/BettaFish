# -*- coding: utf-8 -*-
# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。

import hashlib
import base64
import json
from typing import Any

def _build_c(e: Any, a: Any) -> str:
    c = str(e)
    if isinstance(a, (dict, list)):
        c += json.dumps(a, separators=(",", ":"), ensure_ascii=False)
    elif isinstance(a, str):
        c += a
    # 其它類型不拼
    return c


# ---------------------------
# p.Pu = MD5(c) => hex 小寫
# ---------------------------
def _md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()



# ============================================================
# Playwright 版本（異步）：傳入 page（Page 對象）
#    內部用 page.evaluate('window.mnsv2(...)')
# ============================================================
async def seccore_signv2_playwright(
    page,  # Playwright Page
    e: Any,
    a: Any,
) -> str:
    """
    使用 Playwright 的 page.evaluate 調用 window.mnsv2(c, d) 來生成簽名。
    需確保 page 上下文中已存在 window.mnsv2（比如已注入目標站點腳本）。

    用法：
      s = await page.evaluate("(c, d) => window.mnsv2(c, d)", c, d)
    """
    c = _build_c(e, a)
    d = _md5_hex(c)

    # 調用瀏覽器上下文裏的 window.mnsv2
    s = await page.evaluate("(c, d) => window.mnsv2(c, d)", [c, d])
    f = {
        "x0": "4.2.6",
        "x1": "xhs-pc-web",
        "x2": "Mac OS",
        "x3": s,
        "x4": a,
    }
    payload = json.dumps(f, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    token = "XYS_" + base64.b64encode(payload).decode("ascii")
    print(token)
    return token
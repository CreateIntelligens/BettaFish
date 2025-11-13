# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
# 1. 不得用於任何商業用途。  
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。  
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。  
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。   
# 5. 不得用於任何非法或不當的用途。
#   
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。  
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。  


import re
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

import config
from cache.abs_cache import AbstractCache
from cache.cache_factory import CacheFactory
from tools import utils

app = FastAPI()

cache_client : AbstractCache = CacheFactory.create_cache(cache_type=config.CACHE_TYPE_MEMORY)


class SmsNotification(BaseModel):
    platform: str
    current_number: str
    from_number: str
    sms_content: str
    timestamp: str


def extract_verification_code(message: str) -> str:
    """
    Extract verification code of 6 digits from the SMS.
    """
    pattern = re.compile(r'\b[0-9]{6}\b')
    codes: List[str] = pattern.findall(message)
    return codes[0] if codes else ""


@app.post("/")
def receive_sms_notification(sms: SmsNotification):
    """
    Receive SMS notification and send it to Redis.
    Args:
        sms:
            {
                "platform": "xhs",
                "from_number": "1069421xxx134",
                "sms_content": "【小紅書】您的驗證碼是: 171959， 3分鐘內有效。請勿向他人泄漏。如非本人操作，可忽略本消息。",
                "timestamp": "1686720601614",
                "current_number": "13152442222"
            }

    Returns:

    """
    utils.logger.info(f"Received SMS notification: {sms.platform}, {sms.current_number}")
    sms_code = extract_verification_code(sms.sms_content)
    if sms_code:
        # Save the verification code in Redis and set the expiration time to 3 minutes.
        key = f"{sms.platform}_{sms.current_number}"
        cache_client.set(key, sms_code, expire_time=60 * 3)

    return {"status": "ok"}


@app.get("/", status_code=status.HTTP_404_NOT_FOUND)
async def not_found():
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
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
# @Author  : relakkes@gmail.com
# @Time    : 2023/12/2 12:52
# @Desc    : 時間相關的工具函數

import time
from datetime import datetime, timedelta, timezone


def get_current_timestamp() -> int:
    """
    獲取當前的時間戳(13 位)：1701493264496
    :return:
    """
    return int(time.time() * 1000)


def get_current_time() -> str:
    """
    獲取當前的時間：'2023-12-02 13:01:23'
    :return:
    """
    return time.strftime('%Y-%m-%d %X', time.localtime())

def get_current_time_hour() -> str:
    """
    獲取當前的時間：'2023-12-02-13'
    :return:
    """
    return time.strftime('%Y-%m-%d-%H', time.localtime())

def get_current_date() -> str:
    """
    獲取當前的日期：'2023-12-02'
    :return:
    """
    return time.strftime('%Y-%m-%d', time.localtime())


def get_time_str_from_unix_time(unixtime):
    """
    unix 整數類型時間戳  ==> 字符串日期時間
    :param unixtime:
    :return:
    """
    if int(unixtime) > 1000000000000:
        unixtime = int(unixtime) / 1000
    return time.strftime('%Y-%m-%d %X', time.localtime(unixtime))


def get_date_str_from_unix_time(unixtime):
    """
    unix 整數類型時間戳  ==> 字符串日期
    :param unixtime:
    :return:
    """
    if int(unixtime) > 1000000000000:
        unixtime = int(unixtime) / 1000
    return time.strftime('%Y-%m-%d', time.localtime(unixtime))


def get_unix_time_from_time_str(time_str):
    """
    字符串時間 ==> unix 整數類型時間戳，精確到秒
    :param time_str:
    :return:
    """
    try:
        format_str = "%Y-%m-%d %H:%M:%S"
        tm_object = time.strptime(str(time_str), format_str)
        return int(time.mktime(tm_object))
    except Exception as e:
        return 0
    pass


def get_unix_timestamp():
    return int(time.time())


def rfc2822_to_china_datetime(rfc2822_time):
    # 定義RFC 2822格式
    rfc2822_format = "%a %b %d %H:%M:%S %z %Y"

    # 將RFC 2822時間字符串轉換爲datetime對象
    dt_object = datetime.strptime(rfc2822_time, rfc2822_format)

    # 將datetime對象的時區轉換爲中國時區
    dt_object_china = dt_object.astimezone(timezone(timedelta(hours=8)))
    return dt_object_china


def rfc2822_to_timestamp(rfc2822_time):
    # 定義RFC 2822格式
    rfc2822_format = "%a %b %d %H:%M:%S %z %Y"

    # 將RFC 2822時間字符串轉換爲datetime對象
    dt_object = datetime.strptime(rfc2822_time, rfc2822_format)

    # 將datetime對象轉換爲UTC時間
    dt_utc = dt_object.replace(tzinfo=timezone.utc)

    # 計算UTC時間對應的Unix時間戳
    timestamp = int(dt_utc.timestamp())

    return timestamp


if __name__ == '__main__':
    # 示例用法
    _rfc2822_time = "Sat Dec 23 17:12:54 +0800 2023"
    print(rfc2822_to_china_datetime(_rfc2822_time))

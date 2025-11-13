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
# @Time    : 2024/1/14 17:34
# @Desc    :
from typing import List

import config
from var import source_keyword_var

from .xhs_store_media import *
from ._store_impl import *


class XhsStoreFactory:
    STORES = {
        "csv": XhsCsvStoreImplement,
        "db": XhsDbStoreImplement,
        "json": XhsJsonStoreImplement,
        "sqlite": XhsSqliteStoreImplement,
        "postgresql": XhsDbStoreImplement,
    }

    @staticmethod
    def create_store() -> AbstractStore:
        store_class = XhsStoreFactory.STORES.get(config.SAVE_DATA_OPTION)
        if not store_class:
            raise ValueError("[XhsStoreFactory.create_store] Invalid save option only supported csv or db or json or sqlite or postgresql ...")
        return store_class()


def get_video_url_arr(note_item: Dict) -> List:
    """
    獲取視頻url數組
    Args:
        note_item:

    Returns:

    """
    if note_item.get('type') != 'video':
        return []

    videoArr = []
    originVideoKey = note_item.get('video').get('consumer').get('origin_video_key')
    if originVideoKey == '':
        originVideoKey = note_item.get('video').get('consumer').get('originVideoKey')
    # 降級有水印
    if originVideoKey == '':
        videos = note_item.get('video').get('media').get('stream').get('h264')
        if type(videos).__name__ == 'list':
            videoArr = [v.get('master_url') for v in videos]
    else:
        videoArr = [f"http://sns-video-bd.xhscdn.com/{originVideoKey}"]

    return videoArr


async def update_xhs_note(note_item: Dict):
    """
    更新小紅書筆記
    Args:
        note_item:

    Returns:

    """
    note_id = note_item.get("note_id")
    user_info = note_item.get("user", {})
    interact_info = note_item.get("interact_info", {})
    image_list: List[Dict] = note_item.get("image_list", [])
    tag_list: List[Dict] = note_item.get("tag_list", [])

    for img in image_list:
        if img.get('url_default') != '':
            img.update({'url': img.get('url_default')})

    video_url = ','.join(get_video_url_arr(note_item))

    local_db_item = {
        "note_id": note_item.get("note_id"),  # 帖子id
        "type": note_item.get("type"),  # 帖子類型
        "title": note_item.get("title") or note_item.get("desc", "")[:255],  # 帖子標題
        "desc": note_item.get("desc", ""),  # 帖子描述
        "video_url": video_url,  # 帖子視頻url
        "time": note_item.get("time"),  # 帖子發佈時間
        "last_update_time": note_item.get("last_update_time", 0),  # 帖子最後更新時間
        "user_id": user_info.get("user_id"),  # 用戶id
        "nickname": user_info.get("nickname"),  # 用戶暱稱
        "avatar": user_info.get("avatar"),  # 用戶頭像
        "liked_count": interact_info.get("liked_count"),  # 點贊數
        "collected_count": interact_info.get("collected_count"),  # 收藏數
        "comment_count": interact_info.get("comment_count"),  # 評論數
        "share_count": interact_info.get("share_count"),  # 分享數
        "ip_location": note_item.get("ip_location", ""),  # ip地址
        "image_list": ','.join([img.get('url', '') for img in image_list]),  # 圖片url
        "tag_list": ','.join([tag.get('name', '') for tag in tag_list if tag.get('type') == 'topic']),  # 標籤
        "last_modify_ts": utils.get_current_timestamp(),  # 最後更新時間戳（MediaCrawler程序生成的，主要用途在db存儲的時候記錄一條記錄最新更新時間）
        "note_url": f"https://www.xiaohongshu.com/explore/{note_id}?xsec_token={note_item.get('xsec_token')}&xsec_source=pc_search",  # 帖子url
        "source_keyword": source_keyword_var.get(),  # 搜索關鍵詞
        "xsec_token": note_item.get("xsec_token"),  # xsec_token
    }
    utils.logger.info(f"[store.xhs.update_xhs_note] xhs note: {local_db_item}")
    await XhsStoreFactory.create_store().store_content(local_db_item)


async def batch_update_xhs_note_comments(note_id: str, comments: List[Dict]):
    """
    批量更新小紅書筆記評論
    Args:
        note_id:
        comments:

    Returns:

    """
    if not comments:
        return
    for comment_item in comments:
        await update_xhs_note_comment(note_id, comment_item)


async def update_xhs_note_comment(note_id: str, comment_item: Dict):
    """
    更新小紅書筆記評論
    Args:
        note_id:
        comment_item:

    Returns:

    """
    user_info = comment_item.get("user_info", {})
    comment_id = comment_item.get("id")
    comment_pictures = [item.get("url_default", "") for item in comment_item.get("pictures", [])]
    target_comment = comment_item.get("target_comment", {})
    local_db_item = {
        "comment_id": comment_id,  # 評論id
        "create_time": comment_item.get("create_time"),  # 評論時間
        "ip_location": comment_item.get("ip_location"),  # ip地址
        "note_id": note_id,  # 帖子id
        "content": comment_item.get("content"),  # 評論內容
        "user_id": user_info.get("user_id"),  # 用戶id
        "nickname": user_info.get("nickname"),  # 用戶暱稱
        "avatar": user_info.get("image"),  # 用戶頭像
        "sub_comment_count": comment_item.get("sub_comment_count", 0),  # 子評論數
        "pictures": ",".join(comment_pictures),  # 評論圖片
        "parent_comment_id": target_comment.get("id", 0),  # 父評論id
        "last_modify_ts": utils.get_current_timestamp(),  # 最後更新時間戳（MediaCrawler程序生成的，主要用途在db存儲的時候記錄一條記錄最新更新時間）
        "like_count": comment_item.get("like_count", 0),
    }
    utils.logger.info(f"[store.xhs.update_xhs_note_comment] xhs note comment:{local_db_item}")
    await XhsStoreFactory.create_store().store_comment(local_db_item)


async def save_creator(user_id: str, creator: Dict):
    """
    保存小紅書創作者
    Args:
        user_id:
        creator:

    Returns:

    """
    user_info = creator.get('basicInfo', {})

    follows = 0
    fans = 0
    interaction = 0
    for i in creator.get('interactions'):
        if i.get('type') == 'follows':
            follows = i.get('count')
        elif i.get('type') == 'fans':
            fans = i.get('count')
        elif i.get('type') == 'interaction':
            interaction = i.get('count')

    def get_gender(gender):
        if gender == 1:
            return '女'
        elif gender == 0:
            return '男'
        else:
            return None

    local_db_item = {
        'user_id': user_id,  # 用戶id
        'nickname': user_info.get('nickname'),  # 暱稱
        'gender': get_gender(user_info.get('gender')),  # 性別
        'avatar': user_info.get('images'),  # 頭像
        'desc': user_info.get('desc'),  # 個人描述
        'ip_location': user_info.get('ipLocation'),  # ip地址
        'follows': follows,  # 關注數
        'fans': fans,  # 粉絲數
        'interaction': interaction,  # 互動數
        'tag_list': json.dumps({tag.get('tagType'): tag.get('name')
                                for tag in creator.get('tags')}, ensure_ascii=False),  # 標籤
        "last_modify_ts": utils.get_current_timestamp(),  # 最後更新時間戳（MediaCrawler程序生成的，主要用途在db存儲的時候記錄一條記錄最新更新時間）
    }
    utils.logger.info(f"[store.xhs.save_creator] creator:{local_db_item}")
    await XhsStoreFactory.create_store().store_creator(local_db_item)


async def update_xhs_note_image(note_id, pic_content, extension_file_name):
    """
    更新小紅書筆記圖片
    Args:
        note_id:
        pic_content:
        extension_file_name:

    Returns:

    """

    await XiaoHongShuImage().store_image({"notice_id": note_id, "pic_content": pic_content, "extension_file_name": extension_file_name})


async def update_xhs_note_video(note_id, video_content, extension_file_name):
    """
    更新小紅書筆記視頻
    Args:
        note_id:
        video_content:
        extension_file_name:

    Returns:

    """

    await XiaoHongShuVideo().store_video({"notice_id": note_id, "video_content": video_content, "extension_file_name": extension_file_name})

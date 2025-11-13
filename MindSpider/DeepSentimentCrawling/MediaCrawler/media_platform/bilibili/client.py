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
# @Time    : 2023/12/2 18:44
# @Desc    : bilibili 請求客戶端
import asyncio
import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

import httpx
from playwright.async_api import BrowserContext, Page

import config
from base.base_crawler import AbstractApiClient
from tools import utils

from .exception import DataFetchError
from .field import CommentOrderType, SearchOrderType
from .help import BilibiliSign


class BilibiliClient(AbstractApiClient):

    def __init__(
        self,
        timeout=60,  # 若開啓爬取媒體選項，b 站的長視頻需要更久的超時時間
        proxy=None,
        *,
        headers: Dict[str, str],
        playwright_page: Page,
        cookie_dict: Dict[str, str],
    ):
        self.proxy = proxy
        self.timeout = timeout
        self.headers = headers
        self._host = "https://api.bilibili.com"
        self.playwright_page = playwright_page
        self.cookie_dict = cookie_dict

    async def request(self, method, url, **kwargs) -> Any:
        async with httpx.AsyncClient(proxy=self.proxy) as client:
            response = await client.request(method, url, timeout=self.timeout, **kwargs)
        try:
            data: Dict = response.json()
        except json.JSONDecodeError:
            utils.logger.error(f"[BilibiliClient.request] Failed to decode JSON from response. status_code: {response.status_code}, response_text: {response.text}")
            raise DataFetchError(f"Failed to decode JSON, content: {response.text}")
        if data.get("code") != 0:
            raise DataFetchError(data.get("message", "unkonw error"))
        else:
            return data.get("data", {})

    async def pre_request_data(self, req_data: Dict) -> Dict:
        """
        發送請求進行請求參數簽名
        需要從 localStorage 拿 wbi_img_urls 這參數，值如下：
        https://i0.hdslb.com/bfs/wbi/7cd084941338484aae1ad9425b84077c.png-https://i0.hdslb.com/bfs/wbi/4932caff0ff746eab6f01bf08b70ac45.png
        :param req_data:
        :return:
        """
        if not req_data:
            return {}
        img_key, sub_key = await self.get_wbi_keys()
        return BilibiliSign(img_key, sub_key).sign(req_data)

    async def get_wbi_keys(self) -> Tuple[str, str]:
        """
        獲取最新的 img_key 和 sub_key
        :return:
        """
        local_storage = await self.playwright_page.evaluate("() => window.localStorage")
        wbi_img_urls = local_storage.get("wbi_img_urls", "")
        if not wbi_img_urls:
            img_url_from_storage = local_storage.get("wbi_img_url")
            sub_url_from_storage = local_storage.get("wbi_sub_url")
            if img_url_from_storage and sub_url_from_storage:
                wbi_img_urls = f"{img_url_from_storage}-{sub_url_from_storage}"
        if wbi_img_urls and "-" in wbi_img_urls:
            img_url, sub_url = wbi_img_urls.split("-")
        else:
            resp = await self.request(method="GET", url=self._host + "/x/web-interface/nav")
            img_url: str = resp['wbi_img']['img_url']
            sub_url: str = resp['wbi_img']['sub_url']
        img_key = img_url.rsplit('/', 1)[1].split('.')[0]
        sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
        return img_key, sub_key

    async def get(self, uri: str, params=None, enable_params_sign: bool = True) -> Dict:
        final_uri = uri
        if enable_params_sign:
            params = await self.pre_request_data(params)
        if isinstance(params, dict):
            final_uri = (f"{uri}?"
                         f"{urlencode(params)}")
        return await self.request(method="GET", url=f"{self._host}{final_uri}", headers=self.headers)

    async def post(self, uri: str, data: dict) -> Dict:
        data = await self.pre_request_data(data)
        json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        return await self.request(method="POST", url=f"{self._host}{uri}", data=json_str, headers=self.headers)

    async def pong(self) -> bool:
        """get a note to check if login state is ok"""
        utils.logger.info("[BilibiliClient.pong] Begin pong bilibili...")
        ping_flag = False
        try:
            check_login_uri = "/x/web-interface/nav"
            response = await self.get(check_login_uri)
            if response.get("isLogin"):
                utils.logger.info("[BilibiliClient.pong] Use cache login state get web interface successfull!")
                ping_flag = True
        except Exception as e:
            utils.logger.error(f"[BilibiliClient.pong] Pong bilibili failed: {e}, and try to login again...")
            ping_flag = False
        return ping_flag

    async def update_cookies(self, browser_context: BrowserContext):
        cookie_str, cookie_dict = utils.convert_cookies(await browser_context.cookies())
        self.headers["Cookie"] = cookie_str
        self.cookie_dict = cookie_dict

    async def search_video_by_keyword(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        order: SearchOrderType = SearchOrderType.DEFAULT,
        pubtime_begin_s: int = 0,
        pubtime_end_s: int = 0,
    ) -> Dict:
        """
        KuaiShou web search api
        :param keyword: 搜索關鍵詞
        :param page: 分頁參數具體第幾頁
        :param page_size: 每一頁參數的數量
        :param order: 搜索結果排序，默認位綜合排序
        :param pubtime_begin_s: 發佈時間開始時間戳
        :param pubtime_end_s: 發佈時間結束時間戳
        :return:
        """
        uri = "/x/web-interface/wbi/search/type"
        post_data = {
            "search_type": "video",
            "keyword": keyword,
            "page": page,
            "page_size": page_size,
            "order": order.value,
            "pubtime_begin_s": pubtime_begin_s,
            "pubtime_end_s": pubtime_end_s
        }
        return await self.get(uri, post_data)

    async def get_video_info(self, aid: Union[int, None] = None, bvid: Union[str, None] = None) -> Dict:
        """
        Bilibli web video detail api, aid 和 bvid任選一個參數
        :param aid: 稿件avid
        :param bvid: 稿件bvid
        :return:
        """
        if not aid and not bvid:
            raise ValueError("請提供 aid 或 bvid 中的至少一個參數")

        uri = "/x/web-interface/view/detail"
        params = dict()
        if aid:
            params.update({"aid": aid})
        else:
            params.update({"bvid": bvid})
        return await self.get(uri, params, enable_params_sign=False)

    async def get_video_play_url(self, aid: int, cid: int) -> Dict:
        """
        Bilibli web video play url api
        :param aid: 稿件avid
        :param cid: cid
        :return:
        """
        if not aid or not cid or aid <= 0 or cid <= 0:
            raise ValueError("aid 和 cid 必須存在")
        uri = "/x/player/wbi/playurl"
        qn_value = getattr(config, "BILI_QN", 80)
        params = {
            "avid": aid,
            "cid": cid,
            "qn": qn_value,
            "fourk": 1,
            "fnval": 1,
            "platform": "pc",
        }

        return await self.get(uri, params, enable_params_sign=True)

    async def get_video_media(self, url: str) -> Union[bytes, None]:
        # Follow CDN 302 redirects and treat any 2xx as success (some endpoints return 206)
        async with httpx.AsyncClient(proxy=self.proxy, follow_redirects=True) as client:
            try:
                response = await client.request("GET", url, timeout=self.timeout, headers=self.headers)
                response.raise_for_status()
                if 200 <= response.status_code < 300:
                    return response.content
                utils.logger.error(
                    f"[BilibiliClient.get_video_media] Unexpected status {response.status_code} for {url}"
                )
                return None
            except httpx.HTTPError as exc:  # some wrong when call httpx.request method, such as connection error, client error, server error or response status code is not 2xx
                utils.logger.error(f"[BilibiliClient.get_video_media] {exc.__class__.__name__} for {exc.request.url} - {exc}")  # 保留原始異常類型名稱，以便開發者調試
                return None

    async def get_video_comments(
        self,
        video_id: str,
        order_mode: CommentOrderType = CommentOrderType.DEFAULT,
        next: int = 0,
    ) -> Dict:
        """get video comments
        :param video_id: 視頻 ID
        :param order_mode: 排序方式
        :param next: 評論頁選擇
        :return:
        """
        uri = "/x/v2/reply/wbi/main"
        post_data = {"oid": video_id, "mode": order_mode.value, "type": 1, "ps": 20, "next": next}
        return await self.get(uri, post_data)

    async def get_video_all_comments(
        self,
        video_id: str,
        crawl_interval: float = 1.0,
        is_fetch_sub_comments=False,
        callback: Optional[Callable] = None,
        max_count: int = 10,
    ):
        """
        get video all comments include sub comments
        :param video_id:
        :param crawl_interval:
        :param is_fetch_sub_comments:
        :param callback:
        max_count: 一次筆記爬取的最大評論數量

        :return:
        """
        result = []
        is_end = False
        next_page = 0
        max_retries = 3
        while not is_end and len(result) < max_count:
            comments_res = None
            for attempt in range(max_retries):
                try:
                    comments_res = await self.get_video_comments(video_id, CommentOrderType.DEFAULT, next_page)
                    break  # Success
                except DataFetchError as e:
                    if attempt < max_retries - 1:
                        delay = 5 * (2**attempt) + random.uniform(0, 1)
                        utils.logger.warning(f"[BilibiliClient.get_video_all_comments] Retrying video_id {video_id} in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        utils.logger.error(f"[BilibiliClient.get_video_all_comments] Max retries reached for video_id: {video_id}. Skipping comments. Error: {e}")
                        is_end = True
                        break
            if not comments_res:
                break

            cursor_info: Dict = comments_res.get("cursor")
            if not cursor_info:
                utils.logger.warning(f"[BilibiliClient.get_video_all_comments] Could not find 'cursor' in response for video_id: {video_id}. Skipping.")
                break

            comment_list: List[Dict] = comments_res.get("replies", [])

            # 檢查 is_end 和 next 是否存在
            if "is_end" not in cursor_info or "next" not in cursor_info:
                utils.logger.warning(f"[BilibiliClient.get_video_all_comments] 'is_end' or 'next' not in cursor for video_id: {video_id}. Assuming end of comments.")
                is_end = True
            else:
                is_end = cursor_info.get("is_end")
                next_page = cursor_info.get("next")

            if not isinstance(is_end, bool):
                utils.logger.warning(f"[BilibiliClient.get_video_all_comments] 'is_end' is not a boolean for video_id: {video_id}. Assuming end of comments.")
                is_end = True
            if is_fetch_sub_comments:
                for comment in comment_list:
                    comment_id = comment['rpid']
                    if (comment.get("rcount", 0) > 0):
                        {await self.get_video_all_level_two_comments(video_id, comment_id, CommentOrderType.DEFAULT, 10, crawl_interval, callback)}
            if len(result) + len(comment_list) > max_count:
                comment_list = comment_list[:max_count - len(result)]
            if callback:  # 如果有回調函數，就執行回調函數
                await callback(video_id, comment_list)
            await asyncio.sleep(crawl_interval)
            if not is_fetch_sub_comments:
                result.extend(comment_list)
                continue
        return result

    async def get_video_all_level_two_comments(
        self,
        video_id: str,
        level_one_comment_id: int,
        order_mode: CommentOrderType,
        ps: int = 10,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        get video all level two comments for a level one comment
        :param video_id: 視頻 ID
        :param level_one_comment_id: 一級評論 ID
        :param order_mode:
        :param ps: 一頁評論數
        :param crawl_interval:
        :param callback:
        :return:
        """

        pn = 1
        while True:
            result = await self.get_video_level_two_comments(video_id, level_one_comment_id, pn, ps, order_mode)
            comment_list: List[Dict] = result.get("replies", [])
            if callback:  # 如果有回調函數，就執行回調函數
                await callback(video_id, comment_list)
            await asyncio.sleep(crawl_interval)
            if (int(result["page"]["count"]) <= pn * ps):
                break

            pn += 1

    async def get_video_level_two_comments(
        self,
        video_id: str,
        level_one_comment_id: int,
        pn: int,
        ps: int,
        order_mode: CommentOrderType,
    ) -> Dict:
        """get video level two comments
        :param video_id: 視頻 ID
        :param level_one_comment_id: 一級評論 ID
        :param order_mode: 排序方式

        :return:
        """
        uri = "/x/v2/reply/reply"
        post_data = {
            "oid": video_id,
            "mode": order_mode.value,
            "type": 1,
            "ps": ps,
            "pn": pn,
            "root": level_one_comment_id,
        }
        result = await self.get(uri, post_data)
        return result

    async def get_creator_videos(self, creator_id: str, pn: int, ps: int = 30, order_mode: SearchOrderType = SearchOrderType.LAST_PUBLISH) -> Dict:
        """get all videos for a creator
        :param creator_id: 創作者 ID
        :param pn: 頁數
        :param ps: 一頁視頻數
        :param order_mode: 排序方式

        :return:
        """
        uri = "/x/space/wbi/arc/search"
        post_data = {
            "mid": creator_id,
            "pn": pn,
            "ps": ps,
            "order": order_mode,
        }
        return await self.get(uri, post_data)

    async def get_creator_info(self, creator_id: int) -> Dict:
        """
        get creator info
        :param creator_id: 作者 ID
        """
        uri = "/x/space/wbi/acc/info"
        post_data = {
            "mid": creator_id,
        }
        return await self.get(uri, post_data)

    async def get_creator_fans(
        self,
        creator_id: int,
        pn: int,
        ps: int = 24,
    ) -> Dict:
        """
        get creator fans
        :param creator_id: 創作者 ID
        :param pn: 開始頁數
        :param ps: 每頁數量
        :return:
        """
        uri = "/x/relation/fans"
        post_data = {
            'vmid': creator_id,
            "pn": pn,
            "ps": ps,
            "gaia_source": "main_web",
        }
        return await self.get(uri, post_data)

    async def get_creator_followings(
        self,
        creator_id: int,
        pn: int,
        ps: int = 24,
    ) -> Dict:
        """
        get creator followings
        :param creator_id: 創作者 ID
        :param pn: 開始頁數
        :param ps: 每頁數量
        :return:
        """
        uri = "/x/relation/followings"
        post_data = {
            "vmid": creator_id,
            "pn": pn,
            "ps": ps,
            "gaia_source": "main_web",
        }
        return await self.get(uri, post_data)

    async def get_creator_dynamics(self, creator_id: int, offset: str = ""):
        """
        get creator comments
        :param creator_id: 創作者 ID
        :param offset: 發送請求所需參數
        :return:
        """
        uri = "/x/polymer/web-dynamic/v1/feed/space"
        post_data = {
            "offset": offset,
            "host_mid": creator_id,
            "platform": "web",
        }

        return await self.get(uri, post_data)

    async def get_creator_all_fans(
        self,
        creator_info: Dict,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_count: int = 100,
    ) -> List:
        """
        get creator all fans
        :param creator_info:
        :param crawl_interval:
        :param callback:
        :param max_count: 一個up主爬取的最大粉絲數量

        :return: up主粉絲數列表
        """
        creator_id = creator_info["id"]
        result = []
        pn = config.START_CONTACTS_PAGE
        while len(result) < max_count:
            fans_res: Dict = await self.get_creator_fans(creator_id, pn=pn)
            fans_list: List[Dict] = fans_res.get("list", [])

            pn += 1
            if len(result) + len(fans_list) > max_count:
                fans_list = fans_list[:max_count - len(result)]
            if callback:  # 如果有回調函數，就執行回調函數
                await callback(creator_info, fans_list)
            await asyncio.sleep(crawl_interval)
            if not fans_list:
                break
            result.extend(fans_list)
        return result

    async def get_creator_all_followings(
        self,
        creator_info: Dict,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_count: int = 100,
    ) -> List:
        """
        get creator all followings
        :param creator_info:
        :param crawl_interval:
        :param callback:
        :param max_count: 一個up主爬取的最大關注者數量

        :return: up主關注者列表
        """
        creator_id = creator_info["id"]
        result = []
        pn = config.START_CONTACTS_PAGE
        while len(result) < max_count:
            followings_res: Dict = await self.get_creator_followings(creator_id, pn=pn)
            followings_list: List[Dict] = followings_res.get("list", [])

            pn += 1
            if len(result) + len(followings_list) > max_count:
                followings_list = followings_list[:max_count - len(result)]
            if callback:  # 如果有回調函數，就執行回調函數
                await callback(creator_info, followings_list)
            await asyncio.sleep(crawl_interval)
            if not followings_list:
                break
            result.extend(followings_list)
        return result

    async def get_creator_all_dynamics(
        self,
        creator_info: Dict,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_count: int = 20,
    ) -> List:
        """
        get creator all followings
        :param creator_info:
        :param crawl_interval:
        :param callback:
        :param max_count: 一個up主爬取的最大動態數量

        :return: up主關注者列表
        """
        creator_id = creator_info["id"]
        result = []
        offset = ""
        has_more = True
        while has_more and len(result) < max_count:
            dynamics_res = await self.get_creator_dynamics(creator_id, offset)
            dynamics_list: List[Dict] = dynamics_res["items"]
            has_more = dynamics_res["has_more"]
            offset = dynamics_res["offset"]
            if len(result) + len(dynamics_list) > max_count:
                dynamics_list = dynamics_list[:max_count - len(result)]
            if callback:
                await callback(creator_info, dynamics_list)
            await asyncio.sleep(crawl_interval)
            result.extend(dynamics_list)
        return result

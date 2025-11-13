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
# @Time    : 2023/12/23 15:40
# @Desc    : 微博爬蟲 API 請求 client

import asyncio
import copy
import json
import re
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import parse_qs, unquote, urlencode

import httpx
from httpx import Response
from playwright.async_api import BrowserContext, Page

import config
from tools import utils

from .exception import DataFetchError
from .field import SearchType


class WeiboClient:

    def __init__(
        self,
        timeout=60,  # 若開啓爬取媒體選項，weibo 的圖片需要更久的超時時間
        proxy=None,
        *,
        headers: Dict[str, str],
        playwright_page: Page,
        cookie_dict: Dict[str, str],
    ):
        self.proxy = proxy
        self.timeout = timeout
        self.headers = headers
        self._host = "https://m.weibo.cn"
        self.playwright_page = playwright_page
        self.cookie_dict = cookie_dict
        self._image_agent_host = "https://i1.wp.com/"

    async def request(self, method, url, **kwargs) -> Union[Response, Dict]:
        enable_return_response = kwargs.pop("return_response", False)
        async with httpx.AsyncClient(proxy=self.proxy) as client:
            response = await client.request(method, url, timeout=self.timeout, **kwargs)

        if enable_return_response:
            return response

        data: Dict = response.json()
        ok_code = data.get("ok")
        if ok_code == 0:  # response error
            utils.logger.error(f"[WeiboClient.request] request {method}:{url} err, res:{data}")
            raise DataFetchError(data.get("msg", "response error"))
        elif ok_code != 1:  # unknown error
            utils.logger.error(f"[WeiboClient.request] request {method}:{url} err, res:{data}")
            raise DataFetchError(data.get("msg", "unknown error"))
        else:  # response right
            return data.get("data", {})

    async def get(self, uri: str, params=None, headers=None, **kwargs) -> Union[Response, Dict]:
        final_uri = uri
        if isinstance(params, dict):
            final_uri = (f"{uri}?"
                         f"{urlencode(params)}")

        if headers is None:
            headers = self.headers
        return await self.request(method="GET", url=f"{self._host}{final_uri}", headers=headers, **kwargs)

    async def post(self, uri: str, data: dict) -> Dict:
        json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        return await self.request(method="POST", url=f"{self._host}{uri}", data=json_str, headers=self.headers)

    async def pong(self) -> bool:
        """get a note to check if login state is ok"""
        utils.logger.info("[WeiboClient.pong] Begin pong weibo...")
        ping_flag = False
        try:
            uri = "/api/config"
            resp_data: Dict = await self.request(method="GET", url=f"{self._host}{uri}", headers=self.headers)
            if resp_data.get("login"):
                ping_flag = True
            else:
                utils.logger.error(f"[WeiboClient.pong] cookie may be invalid and again login...")
        except Exception as e:
            utils.logger.error(f"[WeiboClient.pong] Pong weibo failed: {e}, and try to login again...")
            ping_flag = False
        return ping_flag

    async def update_cookies(self, browser_context: BrowserContext):
        cookie_str, cookie_dict = utils.convert_cookies(await browser_context.cookies())
        self.headers["Cookie"] = cookie_str
        self.cookie_dict = cookie_dict

    async def get_note_by_keyword(
        self,
        keyword: str,
        page: int = 1,
        search_type: SearchType = SearchType.DEFAULT,
    ) -> Dict:
        """
        search note by keyword
        :param keyword: 微博搜搜的關鍵詞
        :param page: 分頁參數 -當前頁碼
        :param search_type: 搜索的類型，見 weibo/filed.py 中的枚舉SearchType
        :return:
        """
        uri = "/api/container/getIndex"
        containerid = f"100103type={search_type.value}&q={keyword}"
        params = {
            "containerid": containerid,
            "page_type": "searchall",
            "page": page,
        }
        return await self.get(uri, params)

    async def get_note_comments(self, mid_id: str, max_id: int, max_id_type: int = 0) -> Dict:
        """get notes comments
        :param mid_id: 微博ID
        :param max_id: 分頁參數ID
        :param max_id_type: 分頁參數ID類型
        :return:
        """
        uri = "/comments/hotflow"
        params = {
            "id": mid_id,
            "mid": mid_id,
            "max_id_type": max_id_type,
        }
        if max_id > 0:
            params.update({"max_id": max_id})
        referer_url = f"https://m.weibo.cn/detail/{mid_id}"
        headers = copy.copy(self.headers)
        headers["Referer"] = referer_url

        return await self.get(uri, params, headers=headers)

    async def get_note_all_comments(
        self,
        note_id: str,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_count: int = 10,
    ):
        """
        get note all comments include sub comments
        :param note_id:
        :param crawl_interval:
        :param callback:
        :param max_count:
        :return:
        """
        result = []
        is_end = False
        max_id = -1
        max_id_type = 0
        while not is_end and len(result) < max_count:
            comments_res = await self.get_note_comments(note_id, max_id, max_id_type)
            max_id: int = comments_res.get("max_id")
            max_id_type: int = comments_res.get("max_id_type")
            comment_list: List[Dict] = comments_res.get("data", [])
            is_end = max_id == 0
            if len(result) + len(comment_list) > max_count:
                comment_list = comment_list[:max_count - len(result)]
            if callback:  # 如果有回調函數，就執行回調函數
                await callback(note_id, comment_list)
            await asyncio.sleep(crawl_interval)
            result.extend(comment_list)
            sub_comment_result = await self.get_comments_all_sub_comments(note_id, comment_list, callback)
            result.extend(sub_comment_result)
        return result

    @staticmethod
    async def get_comments_all_sub_comments(
        note_id: str,
        comment_list: List[Dict],
        callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        獲取評論的所有子評論
        Args:
            note_id:
            comment_list:
            callback:

        Returns:

        """
        if not config.ENABLE_GET_SUB_COMMENTS:
            utils.logger.info(f"[WeiboClient.get_comments_all_sub_comments] Crawling sub_comment mode is not enabled")
            return []

        res_sub_comments = []
        for comment in comment_list:
            sub_comments = comment.get("comments")
            if sub_comments and isinstance(sub_comments, list):
                await callback(note_id, sub_comments)
                res_sub_comments.extend(sub_comments)
        return res_sub_comments

    async def get_note_info_by_id(self, note_id: str) -> Dict:
        """
        根據帖子ID獲取詳情
        :param note_id:
        :return:
        """
        url = f"{self._host}/detail/{note_id}"
        async with httpx.AsyncClient(proxy=self.proxy) as client:
            response = await client.request("GET", url, timeout=self.timeout, headers=self.headers)
            if response.status_code != 200:
                raise DataFetchError(f"get weibo detail err: {response.text}")
            match = re.search(r'var \$render_data = (\[.*?\])\[0\]', response.text, re.DOTALL)
            if match:
                render_data_json = match.group(1)
                render_data_dict = json.loads(render_data_json)
                note_detail = render_data_dict[0].get("status")
                note_item = {"mblog": note_detail}
                return note_item
            else:
                utils.logger.info(f"[WeiboClient.get_note_info_by_id] 未找到$render_data的值")
                return dict()

    async def get_note_image(self, image_url: str) -> bytes:
        image_url = image_url[8:]  # 去掉 https://
        sub_url = image_url.split("/")
        image_url = ""
        for i in range(len(sub_url)):
            if i == 1:
                image_url += "large/"  # 都獲取高清大圖
            elif i == len(sub_url) - 1:
                image_url += sub_url[i]
            else:
                image_url += sub_url[i] + "/"
        # 微博圖牀對外存在防盜鏈，所以需要代理訪問
        # 由於微博圖片是通過 i1.wp.com 來訪問的，所以需要拼接一下
        final_uri = (f"{self._image_agent_host}"
                     f"{image_url}")
        async with httpx.AsyncClient(proxy=self.proxy) as client:
            try:
                response = await client.request("GET", final_uri, timeout=self.timeout)
                response.raise_for_status()
                if not response.reason_phrase == "OK":
                    utils.logger.error(f"[WeiboClient.get_note_image] request {final_uri} err, res:{response.text}")
                    return None
                else:
                    return response.content
            except httpx.HTTPError as exc:  # some wrong when call httpx.request method, such as connection error, client error, server error or response status code is not 2xx
                utils.logger.error(f"[DouYinClient.get_aweme_media] {exc.__class__.__name__} for {exc.request.url} - {exc}")    # 保留原始異常類型名稱，以便開發者調試
                return None

    async def get_creator_container_info(self, creator_id: str) -> Dict:
        """
        獲取用戶的容器ID, 容器信息代表着真實請求的API路徑
            fid_container_id：用戶的微博詳情API的容器ID
            lfid_container_id：用戶的微博列表API的容器ID
        Args:
            creator_id:

        Returns: {

        """
        response = await self.get(f"/u/{creator_id}", return_response=True)
        m_weibocn_params = response.cookies.get("M_WEIBOCN_PARAMS")
        if not m_weibocn_params:
            raise DataFetchError("get containerid failed")
        m_weibocn_params_dict = parse_qs(unquote(m_weibocn_params))
        return {"fid_container_id": m_weibocn_params_dict.get("fid", [""])[0], "lfid_container_id": m_weibocn_params_dict.get("lfid", [""])[0]}

    async def get_creator_info_by_id(self, creator_id: str) -> Dict:
        """
        根據用戶ID獲取用戶詳情
        Args:
            creator_id:

        Returns:

        """
        uri = "/api/container/getIndex"
        container_info = await self.get_creator_container_info(creator_id)
        if container_info.get("fid_container_id") == "" or container_info.get("lfid_container_id") == "":
            utils.logger.error(f"[WeiboClient.get_creator_info_by_id] get containerid failed")
            raise DataFetchError("get containerid failed")
        params = {
            "jumpfrom": "weibocom",
            "type": "uid",
            "value": creator_id,
            "containerid": container_info["fid_container_id"],
        }

        user_res = await self.get(uri, params)

        if user_res.get("tabsInfo"):
            tabs: List[Dict] = user_res.get("tabsInfo", {}).get("tabs", [])
            for tab in tabs:
                if tab.get("tabKey") == "weibo":
                    container_info["lfid_container_id"] = tab.get("containerid")
                    break

        user_res.update(container_info)
        return user_res

    async def get_notes_by_creator(
        self,
        creator: str,
        container_id: str,
        since_id: str = "0",
    ) -> Dict:
        """
        獲取博主的筆記
        Args:
            creator: 博主ID
            container_id: 容器ID
            since_id: 上一頁最後一條筆記的ID
        Returns:

        """

        uri = "/api/container/getIndex"
        params = {
            "jumpfrom": "weibocom",
            "type": "uid",
            "value": creator,
            "containerid": container_id,
            "since_id": since_id,
        }
        return await self.get(uri, params)

    async def get_all_notes_by_creator_id(
        self,
        creator_id: str,
        container_id: str,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        獲取指定用戶下的所有發過的帖子，該方法會一直查找一個用戶下的所有帖子信息
        Args:
            creator_id:
            container_id:
            crawl_interval:
            callback:

        Returns:

        """
        result = []
        notes_has_more = True
        since_id = ""
        crawler_total_count = 0
        while notes_has_more:
            notes_res = await self.get_notes_by_creator(creator_id, container_id, since_id)
            if not notes_res:
                utils.logger.error(f"[WeiboClient.get_notes_by_creator] The current creator may have been banned by xhs, so they cannot access the data.")
                break
            since_id = notes_res.get("cardlistInfo", {}).get("since_id", "0")
            if "cards" not in notes_res:
                utils.logger.info(f"[WeiboClient.get_all_notes_by_creator] No 'notes' key found in response: {notes_res}")
                break

            notes = notes_res["cards"]
            utils.logger.info(f"[WeiboClient.get_all_notes_by_creator] got user_id:{creator_id} notes len : {len(notes)}")
            notes = [note for note in notes if note.get("card_type") == 9]
            if callback:
                await callback(notes)
            await asyncio.sleep(crawl_interval)
            result.extend(notes)
            crawler_total_count += 10
            notes_has_more = notes_res.get("cardlistInfo", {}).get("total", 0) > crawler_total_count
        return result

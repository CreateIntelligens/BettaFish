# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlencode, quote

import requests
from playwright.async_api import BrowserContext, Page
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

import config
from base.base_crawler import AbstractApiClient
from model.m_baidu_tieba import TiebaComment, TiebaCreator, TiebaNote
from proxy.proxy_ip_pool import ProxyIpPool
from tools import utils

from .field import SearchNoteType, SearchSortType
from .help import TieBaExtractor


class BaiduTieBaClient(AbstractApiClient):

    def __init__(
        self,
        timeout=10,
        ip_pool=None,
        default_ip_proxy=None,
        headers: Dict[str, str] = None,
        playwright_page: Optional[Page] = None,
    ):
        self.ip_pool: Optional[ProxyIpPool] = ip_pool
        self.timeout = timeout
        # 使用傳入的headers(包含真實瀏覽器UA)或默認headers
        self.headers = headers or {
            "User-Agent": utils.get_user_agent(),
            "Cookie": "",
        }
        self._host = "https://tieba.baidu.com"
        self._page_extractor = TieBaExtractor()
        self.default_ip_proxy = default_ip_proxy
        self.playwright_page = playwright_page  # Playwright頁面對象

    def _sync_request(self, method, url, proxy=None, **kwargs):
        """
        同步的requests請求方法
        Args:
            method: 請求方法
            url: 請求的URL
            proxy: 代理IP
            **kwargs: 其他請求參數

        Returns:
            response對象
        """
        # 構造代理字典
        proxies = None
        if proxy:
            proxies = {
                "http": proxy,
                "https": proxy,
            }

        # 發送請求
        response = requests.request(
            method=method,
            url=url,
            headers=self.headers,
            proxies=proxies,
            timeout=self.timeout,
            **kwargs
        )
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def request(self, method, url, return_ori_content=False, proxy=None, **kwargs) -> Union[str, Any]:
        """
        封裝requests的公共請求方法，對請求響應做一些處理
        Args:
            method: 請求方法
            url: 請求的URL
            return_ori_content: 是否返回原始內容
            proxy: 代理IP
            **kwargs: 其他請求參數，例如請求頭、請求體等

        Returns:

        """
        actual_proxy = proxy if proxy else self.default_ip_proxy

        # 在線程池中執行同步的requests請求
        response = await asyncio.to_thread(
            self._sync_request,
            method,
            url,
            actual_proxy,
            **kwargs
        )

        if response.status_code != 200:
            utils.logger.error(f"Request failed, method: {method}, url: {url}, status code: {response.status_code}")
            utils.logger.error(f"Request failed, response: {response.text}")
            raise Exception(f"Request failed, method: {method}, url: {url}, status code: {response.status_code}")

        if response.text == "" or response.text == "blocked":
            utils.logger.error(f"request params incorrect, response.text: {response.text}")
            raise Exception("account blocked")

        if return_ori_content:
            return response.text

        return response.json()

    async def get(self, uri: str, params=None, return_ori_content=False, **kwargs) -> Any:
        """
        GET請求，對請求頭簽名
        Args:
            uri: 請求路由
            params: 請求參數
            return_ori_content: 是否返回原始內容

        Returns:

        """
        final_uri = uri
        if isinstance(params, dict):
            final_uri = (f"{uri}?"
                         f"{urlencode(params)}")
        try:
            res = await self.request(method="GET", url=f"{self._host}{final_uri}", return_ori_content=return_ori_content, **kwargs)
            return res
        except RetryError as e:
            if self.ip_pool:
                proxie_model = await self.ip_pool.get_proxy()
                _, proxy = utils.format_proxy_info(proxie_model)
                res = await self.request(method="GET", url=f"{self._host}{final_uri}", return_ori_content=return_ori_content, proxy=proxy, **kwargs)
                self.default_ip_proxy = proxy
                return res

            utils.logger.error(f"[BaiduTieBaClient.get] 達到了最大重試次數，IP已經被Block，請嘗試更換新的IP代理: {e}")
            raise Exception(f"[BaiduTieBaClient.get] 達到了最大重試次數，IP已經被Block，請嘗試更換新的IP代理: {e}")

    async def post(self, uri: str, data: dict, **kwargs) -> Dict:
        """
        POST請求，對請求頭簽名
        Args:
            uri: 請求路由
            data: 請求體參數

        Returns:

        """
        json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        return await self.request(method="POST", url=f"{self._host}{uri}", data=json_str, **kwargs)

    async def pong(self, browser_context: BrowserContext = None) -> bool:
        """
        用於檢查登錄態是否失效了
        使用Cookie檢測而非API調用,避免被檢測
        Args:
            browser_context: 瀏覽器上下文對象

        Returns:
            bool: True表示已登錄,False表示未登錄
        """
        utils.logger.info("[BaiduTieBaClient.pong] Begin to check tieba login state by cookies...")

        if not browser_context:
            utils.logger.warning("[BaiduTieBaClient.pong] browser_context is None, assume not logged in")
            return False

        try:
            # 從瀏覽器獲取cookies並檢查關鍵登錄cookie
            _, cookie_dict = utils.convert_cookies(await browser_context.cookies())

            # 百度貼吧的登錄標識: STOKEN 或 PTOKEN
            stoken = cookie_dict.get("STOKEN")
            ptoken = cookie_dict.get("PTOKEN")
            bduss = cookie_dict.get("BDUSS")  # 百度通用登錄cookie

            if stoken or ptoken or bduss:
                utils.logger.info(f"[BaiduTieBaClient.pong] Login state verified by cookies (STOKEN: {bool(stoken)}, PTOKEN: {bool(ptoken)}, BDUSS: {bool(bduss)})")
                return True
            else:
                utils.logger.info("[BaiduTieBaClient.pong] No valid login cookies found, need to login")
                return False

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaClient.pong] Check login state failed: {e}, assume not logged in")
            return False

    async def update_cookies(self, browser_context: BrowserContext):
        """
        API客戶端提供的更新cookies方法，一般情況下登錄成功後會調用此方法
        Args:
            browser_context: 瀏覽器上下文對象

        Returns:

        """
        cookie_str, cookie_dict = utils.convert_cookies(await browser_context.cookies())
        self.headers["Cookie"] = cookie_str
        utils.logger.info("[BaiduTieBaClient.update_cookies] Cookie has been updated")

    async def get_notes_by_keyword(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 10,
        sort: SearchSortType = SearchSortType.TIME_DESC,
        note_type: SearchNoteType = SearchNoteType.FIXED_THREAD,
    ) -> List[TiebaNote]:
        """
        根據關鍵詞搜索貼吧帖子 (使用Playwright訪問頁面,避免API檢測)
        Args:
            keyword: 關鍵詞
            page: 分頁第幾頁
            page_size: 每頁大小
            sort: 結果排序方式
            note_type: 帖子類型（主題貼｜主題+回覆混合模式）
        Returns:

        """
        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_notes_by_keyword] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based search")

        # 構造搜索URL
        # 示例: https://tieba.baidu.com/f/search/res?ie=utf-8&qw=編程
        search_url = f"{self._host}/f/search/res"
        params = {
            "ie": "utf-8",
            "qw": keyword,
            "rn": page_size,
            "pn": page,
            "sm": sort.value,
            "only_thread": note_type.value,
        }

        # 拼接完整URL
        full_url = f"{search_url}?{urlencode(params)}"
        utils.logger.info(f"[BaiduTieBaClient.get_notes_by_keyword] 訪問搜索頁面: {full_url}")

        try:
            # 使用Playwright訪問搜索頁面
            await self.playwright_page.goto(full_url, wait_until="domcontentloaded")

            # 等待頁面加載,使用配置文件中的延時設置
            await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

            # 獲取頁面HTML內容
            page_content = await self.playwright_page.content()
            utils.logger.info(f"[BaiduTieBaClient.get_notes_by_keyword] 成功獲取搜索頁面HTML,長度: {len(page_content)}")

            # 提取搜索結果
            notes = self._page_extractor.extract_search_note_list(page_content)
            utils.logger.info(f"[BaiduTieBaClient.get_notes_by_keyword] 提取到 {len(notes)} 條帖子")
            return notes

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaClient.get_notes_by_keyword] 搜索失敗: {e}")
            raise

    async def get_note_by_id(self, note_id: str) -> TiebaNote:
        """
        根據帖子ID獲取帖子詳情 (使用Playwright訪問頁面,避免API檢測)
        Args:
            note_id: 帖子ID

        Returns:
            TiebaNote: 帖子詳情對象
        """
        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_note_by_id] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based note detail fetching")

        # 構造帖子詳情URL
        note_url = f"{self._host}/p/{note_id}"
        utils.logger.info(f"[BaiduTieBaClient.get_note_by_id] 訪問帖子詳情頁面: {note_url}")

        try:
            # 使用Playwright訪問帖子詳情頁面
            await self.playwright_page.goto(note_url, wait_until="domcontentloaded")

            # 等待頁面加載,使用配置文件中的延時設置
            await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

            # 獲取頁面HTML內容
            page_content = await self.playwright_page.content()
            utils.logger.info(f"[BaiduTieBaClient.get_note_by_id] 成功獲取帖子詳情HTML,長度: {len(page_content)}")

            # 提取帖子詳情
            note_detail = self._page_extractor.extract_note_detail(page_content)
            return note_detail

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaClient.get_note_by_id] 獲取帖子詳情失敗: {e}")
            raise

    async def get_note_all_comments(
        self,
        note_detail: TiebaNote,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_count: int = 10,
    ) -> List[TiebaComment]:
        """
        獲取指定帖子下的所有一級評論 (使用Playwright訪問頁面,避免API檢測)
        Args:
            note_detail: 帖子詳情對象
            crawl_interval: 爬取一次筆記的延遲單位（秒）
            callback: 一次筆記爬取結束後的回調函數
            max_count: 一次帖子爬取的最大評論數量
        Returns:
            List[TiebaComment]: 評論列表
        """
        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_note_all_comments] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based comment fetching")

        result: List[TiebaComment] = []
        current_page = 1

        while note_detail.total_replay_page >= current_page and len(result) < max_count:
            # 構造評論頁URL
            comment_url = f"{self._host}/p/{note_detail.note_id}?pn={current_page}"
            utils.logger.info(f"[BaiduTieBaClient.get_note_all_comments] 訪問評論頁面: {comment_url}")

            try:
                # 使用Playwright訪問評論頁面
                await self.playwright_page.goto(comment_url, wait_until="domcontentloaded")

                # 等待頁面加載,使用配置文件中的延時設置
                await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

                # 獲取頁面HTML內容
                page_content = await self.playwright_page.content()

                # 提取評論
                comments = self._page_extractor.extract_tieba_note_parment_comments(
                    page_content, note_id=note_detail.note_id
                )

                if not comments:
                    utils.logger.info(f"[BaiduTieBaClient.get_note_all_comments] 第{current_page}頁沒有評論,停止爬取")
                    break

                # 限制評論數量
                if len(result) + len(comments) > max_count:
                    comments = comments[:max_count - len(result)]

                if callback:
                    await callback(note_detail.note_id, comments)

                result.extend(comments)

                # 獲取所有子評論
                await self.get_comments_all_sub_comments(
                    comments, crawl_interval=crawl_interval, callback=callback
                )

                await asyncio.sleep(crawl_interval)
                current_page += 1

            except Exception as e:
                utils.logger.error(f"[BaiduTieBaClient.get_note_all_comments] 獲取第{current_page}頁評論失敗: {e}")
                break

        utils.logger.info(f"[BaiduTieBaClient.get_note_all_comments] 共獲取 {len(result)} 條一級評論")
        return result

    async def get_comments_all_sub_comments(
        self,
        comments: List[TiebaComment],
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
    ) -> List[TiebaComment]:
        """
        獲取指定評論下的所有子評論 (使用Playwright訪問頁面,避免API檢測)
        Args:
            comments: 評論列表
            crawl_interval: 爬取一次筆記的延遲單位（秒）
            callback: 一次筆記爬取結束後的回調函數

        Returns:
            List[TiebaComment]: 子評論列表
        """
        if not config.ENABLE_GET_SUB_COMMENTS:
            return []

        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_comments_all_sub_comments] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based sub-comment fetching")

        all_sub_comments: List[TiebaComment] = []

        for parment_comment in comments:
            if parment_comment.sub_comment_count == 0:
                continue

            current_page = 1
            max_sub_page_num = parment_comment.sub_comment_count // 10 + 1

            while max_sub_page_num >= current_page:
                # 構造子評論URL
                sub_comment_url = (
                    f"{self._host}/p/comment?"
                    f"tid={parment_comment.note_id}&"
                    f"pid={parment_comment.comment_id}&"
                    f"fid={parment_comment.tieba_id}&"
                    f"pn={current_page}"
                )
                utils.logger.info(f"[BaiduTieBaClient.get_comments_all_sub_comments] 訪問子評論頁面: {sub_comment_url}")

                try:
                    # 使用Playwright訪問子評論頁面
                    await self.playwright_page.goto(sub_comment_url, wait_until="domcontentloaded")

                    # 等待頁面加載,使用配置文件中的延時設置
                    await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

                    # 獲取頁面HTML內容
                    page_content = await self.playwright_page.content()

                    # 提取子評論
                    sub_comments = self._page_extractor.extract_tieba_note_sub_comments(
                        page_content, parent_comment=parment_comment
                    )

                    if not sub_comments:
                        utils.logger.info(
                            f"[BaiduTieBaClient.get_comments_all_sub_comments] "
                            f"評論{parment_comment.comment_id}第{current_page}頁沒有子評論,停止爬取"
                        )
                        break

                    if callback:
                        await callback(parment_comment.note_id, sub_comments)

                    all_sub_comments.extend(sub_comments)
                    await asyncio.sleep(crawl_interval)
                    current_page += 1

                except Exception as e:
                    utils.logger.error(
                        f"[BaiduTieBaClient.get_comments_all_sub_comments] "
                        f"獲取評論{parment_comment.comment_id}第{current_page}頁子評論失敗: {e}"
                    )
                    break

        utils.logger.info(f"[BaiduTieBaClient.get_comments_all_sub_comments] 共獲取 {len(all_sub_comments)} 條子評論")
        return all_sub_comments

    async def get_notes_by_tieba_name(self, tieba_name: str, page_num: int) -> List[TiebaNote]:
        """
        根據貼吧名稱獲取帖子列表 (使用Playwright訪問頁面,避免API檢測)
        Args:
            tieba_name: 貼吧名稱
            page_num: 分頁頁碼

        Returns:
            List[TiebaNote]: 帖子列表
        """
        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_notes_by_tieba_name] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based tieba note fetching")

        # 構造貼吧帖子列表URL
        tieba_url = f"{self._host}/f?kw={quote(tieba_name)}&pn={page_num}"
        utils.logger.info(f"[BaiduTieBaClient.get_notes_by_tieba_name] 訪問貼吧頁面: {tieba_url}")

        try:
            # 使用Playwright訪問貼吧頁面
            await self.playwright_page.goto(tieba_url, wait_until="domcontentloaded")

            # 等待頁面加載,使用配置文件中的延時設置
            await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

            # 獲取頁面HTML內容
            page_content = await self.playwright_page.content()
            utils.logger.info(f"[BaiduTieBaClient.get_notes_by_tieba_name] 成功獲取貼吧頁面HTML,長度: {len(page_content)}")

            # 提取帖子列表
            notes = self._page_extractor.extract_tieba_note_list(page_content)
            utils.logger.info(f"[BaiduTieBaClient.get_notes_by_tieba_name] 提取到 {len(notes)} 條帖子")
            return notes

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaClient.get_notes_by_tieba_name] 獲取貼吧帖子列表失敗: {e}")
            raise

    async def get_creator_info_by_url(self, creator_url: str) -> str:
        """
        根據創作者URL獲取創作者信息 (使用Playwright訪問頁面,避免API檢測)
        Args:
            creator_url: 創作者主頁URL

        Returns:
            str: 頁面HTML內容
        """
        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_creator_info_by_url] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based creator info fetching")

        utils.logger.info(f"[BaiduTieBaClient.get_creator_info_by_url] 訪問創作者主頁: {creator_url}")

        try:
            # 使用Playwright訪問創作者主頁
            await self.playwright_page.goto(creator_url, wait_until="domcontentloaded")

            # 等待頁面加載,使用配置文件中的延時設置
            await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

            # 獲取頁面HTML內容
            page_content = await self.playwright_page.content()
            utils.logger.info(f"[BaiduTieBaClient.get_creator_info_by_url] 成功獲取創作者主頁HTML,長度: {len(page_content)}")

            return page_content

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaClient.get_creator_info_by_url] 獲取創作者主頁失敗: {e}")
            raise

    async def get_notes_by_creator(self, user_name: str, page_number: int) -> Dict:
        """
        根據創作者獲取創作者的帖子 (使用Playwright訪問頁面,避免API檢測)
        Args:
            user_name: 創作者用戶名
            page_number: 頁碼

        Returns:
            Dict: 包含帖子數據的字典
        """
        if not self.playwright_page:
            utils.logger.error("[BaiduTieBaClient.get_notes_by_creator] playwright_page is None, cannot use browser mode")
            raise Exception("playwright_page is required for browser-based creator notes fetching")

        # 構造創作者帖子列表URL
        creator_url = f"{self._host}/home/get/getthread?un={quote(user_name)}&pn={page_number}&id=utf-8&_={utils.get_current_timestamp()}"
        utils.logger.info(f"[BaiduTieBaClient.get_notes_by_creator] 訪問創作者帖子列表: {creator_url}")

        try:
            # 使用Playwright訪問創作者帖子列表頁面
            await self.playwright_page.goto(creator_url, wait_until="domcontentloaded")

            # 等待頁面加載,使用配置文件中的延時設置
            await asyncio.sleep(config.CRAWLER_MAX_SLEEP_SEC)

            # 獲取頁面內容(這個接口返回JSON)
            page_content = await self.playwright_page.content()

            # 提取JSON數據(頁面會包含<pre>標籤或直接是JSON)
            try:
                # 嘗試從頁面中提取JSON
                json_text = await self.playwright_page.evaluate("() => document.body.innerText")
                result = json.loads(json_text)
                utils.logger.info(f"[BaiduTieBaClient.get_notes_by_creator] 成功獲取創作者帖子數據")
                return result
            except json.JSONDecodeError as e:
                utils.logger.error(f"[BaiduTieBaClient.get_notes_by_creator] JSON解析失敗: {e}")
                utils.logger.error(f"[BaiduTieBaClient.get_notes_by_creator] 頁面內容: {page_content[:500]}")
                raise Exception(f"Failed to parse JSON from creator notes page: {e}")

        except Exception as e:
            utils.logger.error(f"[BaiduTieBaClient.get_notes_by_creator] 獲取創作者帖子列表失敗: {e}")
            raise

    async def get_all_notes_by_creator_user_name(
        self,
        user_name: str,
        crawl_interval: float = 1.0,
        callback: Optional[Callable] = None,
        max_note_count: int = 0,
        creator_page_html_content: str = None,
    ) -> List[TiebaNote]:
        """
        根據創作者用戶名獲取創作者所有帖子
        Args:
            user_name: 創作者用戶名
            crawl_interval: 爬取一次筆記的延遲單位（秒）
            callback: 一次筆記爬取結束後的回調函數，是一個awaitable類型的函數
            max_note_count: 帖子最大獲取數量，如果爲0則獲取所有
            creator_page_html_content: 創作者主頁HTML內容

        Returns:

        """
        # 百度貼吧比較特殊一些，前10個帖子是直接展示在主頁上的，要單獨處理，通過API獲取不到
        result: List[TiebaNote] = []
        if creator_page_html_content:
            thread_id_list = (self._page_extractor.extract_tieba_thread_id_list_from_creator_page(creator_page_html_content))
            utils.logger.info(f"[BaiduTieBaClient.get_all_notes_by_creator] got user_name:{user_name} thread_id_list len : {len(thread_id_list)}")
            note_detail_task = [self.get_note_by_id(thread_id) for thread_id in thread_id_list]
            notes = await asyncio.gather(*note_detail_task)
            if callback:
                await callback(notes)
            result.extend(notes)

        notes_has_more = 1
        page_number = 1
        page_per_count = 20
        total_get_count = 0
        while notes_has_more == 1 and (max_note_count == 0 or total_get_count < max_note_count):
            notes_res = await self.get_notes_by_creator(user_name, page_number)
            if not notes_res or notes_res.get("no") != 0:
                utils.logger.error(f"[WeiboClient.get_notes_by_creator] got user_name:{user_name} notes failed, notes_res: {notes_res}")
                break
            notes_data = notes_res.get("data")
            notes_has_more = notes_data.get("has_more")
            notes = notes_data["thread_list"]
            utils.logger.info(f"[WeiboClient.get_all_notes_by_creator] got user_name:{user_name} notes len : {len(notes)}")

            note_detail_task = [self.get_note_by_id(note['thread_id']) for note in notes]
            notes = await asyncio.gather(*note_detail_task)
            if callback:
                await callback(notes)
            await asyncio.sleep(crawl_interval)
            result.extend(notes)
            page_number += 1
            total_get_count += page_per_count
        return result

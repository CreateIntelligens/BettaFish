# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。


import os
import asyncio
import socket
import httpx
from typing import Optional, Dict, Any
from playwright.async_api import Browser, BrowserContext, Playwright

import config
from tools.browser_launcher import BrowserLauncher
from tools import utils


class CDPBrowserManager:
    """
    CDP瀏覽器管理器，負責啓動和管理通過CDP連接的瀏覽器
    """

    def __init__(self):
        self.launcher = BrowserLauncher()
        self.browser: Optional[Browser] = None
        self.browser_context: Optional[BrowserContext] = None
        self.debug_port: Optional[int] = None

    async def launch_and_connect(
        self,
        playwright: Playwright,
        playwright_proxy: Optional[Dict] = None,
        user_agent: Optional[str] = None,
        headless: bool = False,
    ) -> BrowserContext:
        """
        啓動瀏覽器並通過CDP連接
        """
        try:
            # 1. 檢測瀏覽器路徑
            browser_path = await self._get_browser_path()

            # 2. 獲取可用端口
            self.debug_port = self.launcher.find_available_port(config.CDP_DEBUG_PORT)

            # 3. 啓動瀏覽器
            await self._launch_browser(browser_path, headless)

            # 4. 通過CDP連接
            await self._connect_via_cdp(playwright)

            # 5. 創建瀏覽器上下文
            browser_context = await self._create_browser_context(
                playwright_proxy, user_agent
            )

            self.browser_context = browser_context
            return browser_context

        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] CDP瀏覽器啓動失敗: {e}")
            await self.cleanup()
            raise

    async def _get_browser_path(self) -> str:
        """
        獲取瀏覽器路徑
        """
        # 優先使用用戶自定義路徑
        if config.CUSTOM_BROWSER_PATH and os.path.isfile(config.CUSTOM_BROWSER_PATH):
            utils.logger.info(
                f"[CDPBrowserManager] 使用自定義瀏覽器路徑: {config.CUSTOM_BROWSER_PATH}"
            )
            return config.CUSTOM_BROWSER_PATH

        # 自動檢測瀏覽器路徑
        browser_paths = self.launcher.detect_browser_paths()

        if not browser_paths:
            raise RuntimeError(
                "未找到可用的瀏覽器。請確保已安裝Chrome或Edge瀏覽器，"
                "或在配置文件中設置CUSTOM_BROWSER_PATH指定瀏覽器路徑。"
            )

        browser_path = browser_paths[0]  # 使用第一個找到的瀏覽器
        browser_name, browser_version = self.launcher.get_browser_info(browser_path)

        utils.logger.info(
            f"[CDPBrowserManager] 檢測到瀏覽器: {browser_name} ({browser_version})"
        )
        utils.logger.info(f"[CDPBrowserManager] 瀏覽器路徑: {browser_path}")

        return browser_path

    async def _test_cdp_connection(self, debug_port: int) -> bool:
        """
        測試CDP連接是否可用
        """
        try:
            # 簡單的socket連接測試
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                result = s.connect_ex(("localhost", debug_port))
                if result == 0:
                    utils.logger.info(
                        f"[CDPBrowserManager] CDP端口 {debug_port} 可訪問"
                    )
                    return True
                else:
                    utils.logger.warning(
                        f"[CDPBrowserManager] CDP端口 {debug_port} 不可訪問"
                    )
                    return False
        except Exception as e:
            utils.logger.warning(f"[CDPBrowserManager] CDP連接測試失敗: {e}")
            return False

    async def _launch_browser(self, browser_path: str, headless: bool):
        """
        啓動瀏覽器進程
        """
        # 設置用戶數據目錄（如果啓用了保存登錄狀態）
        user_data_dir = None
        if config.SAVE_LOGIN_STATE:
            user_data_dir = os.path.join(
                os.getcwd(),
                "browser_data",
                f"cdp_{config.USER_DATA_DIR % config.PLATFORM}",
            )
            os.makedirs(user_data_dir, exist_ok=True)
            utils.logger.info(f"[CDPBrowserManager] 用戶數據目錄: {user_data_dir}")

        # 啓動瀏覽器
        self.launcher.browser_process = self.launcher.launch_browser(
            browser_path=browser_path,
            debug_port=self.debug_port,
            headless=headless,
            user_data_dir=user_data_dir,
        )

        # 等待瀏覽器準備就緒
        if not self.launcher.wait_for_browser_ready(
            self.debug_port, config.BROWSER_LAUNCH_TIMEOUT
        ):
            raise RuntimeError(f"瀏覽器在 {config.BROWSER_LAUNCH_TIMEOUT} 秒內未能啓動")

        # 額外等待一秒讓CDP服務完全啓動
        await asyncio.sleep(1)

        # 測試CDP連接
        if not await self._test_cdp_connection(self.debug_port):
            utils.logger.warning(
                "[CDPBrowserManager] CDP連接測試失敗，但將繼續嘗試連接"
            )

    async def _get_browser_websocket_url(self, debug_port: int) -> str:
        """
        獲取瀏覽器的WebSocket連接URL
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{debug_port}/json/version", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    ws_url = data.get("webSocketDebuggerUrl")
                    if ws_url:
                        utils.logger.info(
                            f"[CDPBrowserManager] 獲取到瀏覽器WebSocket URL: {ws_url}"
                        )
                        return ws_url
                    else:
                        raise RuntimeError("未找到webSocketDebuggerUrl")
                else:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] 獲取WebSocket URL失敗: {e}")
            raise

    async def _connect_via_cdp(self, playwright: Playwright):
        """
        通過CDP連接到瀏覽器
        """
        try:
            # 獲取正確的WebSocket URL
            ws_url = await self._get_browser_websocket_url(self.debug_port)
            utils.logger.info(f"[CDPBrowserManager] 正在通過CDP連接到瀏覽器: {ws_url}")

            # 使用Playwright的connectOverCDP方法連接
            self.browser = await playwright.chromium.connect_over_cdp(ws_url)

            if self.browser.is_connected():
                utils.logger.info("[CDPBrowserManager] 成功連接到瀏覽器")
                utils.logger.info(
                    f"[CDPBrowserManager] 瀏覽器上下文數量: {len(self.browser.contexts)}"
                )
            else:
                raise RuntimeError("CDP連接失敗")

        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] CDP連接失敗: {e}")
            raise

    async def _create_browser_context(
        self, playwright_proxy: Optional[Dict] = None, user_agent: Optional[str] = None
    ) -> BrowserContext:
        """
        創建或獲取瀏覽器上下文
        """
        if not self.browser:
            raise RuntimeError("瀏覽器未連接")

        # 獲取現有上下文或創建新的上下文
        contexts = self.browser.contexts

        if contexts:
            # 使用現有的第一個上下文
            browser_context = contexts[0]
            utils.logger.info("[CDPBrowserManager] 使用現有的瀏覽器上下文")
        else:
            # 創建新的上下文
            context_options = {
                "viewport": {"width": 1920, "height": 1080},
                "accept_downloads": True,
            }

            # 設置用戶代理
            if user_agent:
                context_options["user_agent"] = user_agent
                utils.logger.info(f"[CDPBrowserManager] 設置用戶代理: {user_agent}")

            # 注意：CDP模式下代理設置可能不生效，因爲瀏覽器已經啓動
            if playwright_proxy:
                utils.logger.warning(
                    "[CDPBrowserManager] 警告: CDP模式下代理設置可能不生效，"
                    "建議在瀏覽器啓動前配置系統代理或瀏覽器代理擴展"
                )

            browser_context = await self.browser.new_context(**context_options)
            utils.logger.info("[CDPBrowserManager] 創建新的瀏覽器上下文")

        return browser_context

    async def add_stealth_script(self, script_path: str = "libs/stealth.min.js"):
        """
        添加反檢測腳本
        """
        if self.browser_context and os.path.exists(script_path):
            try:
                await self.browser_context.add_init_script(path=script_path)
                utils.logger.info(
                    f"[CDPBrowserManager] 已添加反檢測腳本: {script_path}"
                )
            except Exception as e:
                utils.logger.warning(f"[CDPBrowserManager] 添加反檢測腳本失敗: {e}")

    async def add_cookies(self, cookies: list):
        """
        添加Cookie
        """
        if self.browser_context:
            try:
                await self.browser_context.add_cookies(cookies)
                utils.logger.info(f"[CDPBrowserManager] 已添加 {len(cookies)} 個Cookie")
            except Exception as e:
                utils.logger.warning(f"[CDPBrowserManager] 添加Cookie失敗: {e}")

    async def get_cookies(self) -> list:
        """
        獲取當前Cookie
        """
        if self.browser_context:
            try:
                cookies = await self.browser_context.cookies()
                return cookies
            except Exception as e:
                utils.logger.warning(f"[CDPBrowserManager] 獲取Cookie失敗: {e}")
                return []
        return []

    async def cleanup(self):
        """
        清理資源
        """
        try:
            # 關閉瀏覽器上下文
            if self.browser_context:
                try:
                    await self.browser_context.close()
                    utils.logger.info("[CDPBrowserManager] 瀏覽器上下文已關閉")
                except Exception as context_error:
                    utils.logger.warning(
                        f"[CDPBrowserManager] 關閉瀏覽器上下文失敗: {context_error}"
                    )
                finally:
                    self.browser_context = None

            # 斷開瀏覽器連接
            if self.browser:
                try:
                    await self.browser.close()
                    utils.logger.info("[CDPBrowserManager] 瀏覽器連接已斷開")
                except Exception as browser_error:
                    utils.logger.warning(
                        f"[CDPBrowserManager] 關閉瀏覽器連接失敗: {browser_error}"
                    )
                finally:
                    self.browser = None

            # 關閉瀏覽器進程（如果配置爲自動關閉）
            if config.AUTO_CLOSE_BROWSER:
                self.launcher.cleanup()
            else:
                utils.logger.info(
                    "[CDPBrowserManager] 瀏覽器進程保持運行（AUTO_CLOSE_BROWSER=False）"
                )

        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] 清理資源時出錯: {e}")

    def is_connected(self) -> bool:
        """
        檢查是否已連接到瀏覽器
        """
        return self.browser is not None and self.browser.is_connected()

    async def get_browser_info(self) -> Dict[str, Any]:
        """
        獲取瀏覽器信息
        """
        if not self.browser:
            return {}

        try:
            version = self.browser.version
            contexts_count = len(self.browser.contexts)

            return {
                "version": version,
                "contexts_count": contexts_count,
                "debug_port": self.debug_port,
                "is_connected": self.is_connected(),
            }
        except Exception as e:
            utils.logger.warning(f"[CDPBrowserManager] 獲取瀏覽器信息失敗: {e}")
            return {}

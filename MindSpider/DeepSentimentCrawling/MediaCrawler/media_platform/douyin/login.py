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
import functools
import sys
from typing import Optional

from playwright.async_api import BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tenacity import (RetryError, retry, retry_if_result, stop_after_attempt,
                      wait_fixed)

import config
from base.base_crawler import AbstractLogin
from cache.cache_factory import CacheFactory
from tools import utils


class DouYinLogin(AbstractLogin):

    def __init__(self,
                 login_type: str,
                 browser_context: BrowserContext, # type: ignore
                 context_page: Page, # type: ignore
                 login_phone: Optional[str] = "",
                 cookie_str: Optional[str] = ""
                 ):
        config.LOGIN_TYPE = login_type
        self.browser_context = browser_context
        self.context_page = context_page
        self.login_phone = login_phone
        self.scan_qrcode_time = 60
        self.cookie_str = cookie_str

    async def begin(self):
        """
            Start login douyin website
            滑塊中間頁面的驗證準確率不太OK... 如果沒有特俗要求，建議不開抖音登錄，或者使用cookies登錄
        """

        # popup login dialog
        await self.popup_login_dialog()

        # select login type
        if config.LOGIN_TYPE == "qrcode":
            await self.login_by_qrcode()
        elif config.LOGIN_TYPE == "phone":
            await self.login_by_mobile()
        elif config.LOGIN_TYPE == "cookie":
            await self.login_by_cookies()
        else:
            raise ValueError("[DouYinLogin.begin] Invalid Login Type Currently only supported qrcode or phone or cookie ...")

        # 如果頁面重定向到滑動驗證碼頁面，需要再次滑動滑塊
        await asyncio.sleep(6)
        current_page_title = await self.context_page.title()
        if "驗證碼中間頁" in current_page_title:
            await self.check_page_display_slider(move_step=3, slider_level="hard")

        # check login state
        utils.logger.info(f"[DouYinLogin.begin] login finished then check login state ...")
        try:
            await self.check_login_state()
        except RetryError:
            utils.logger.info("[DouYinLogin.begin] login failed please confirm ...")
            sys.exit()

        # wait for redirect
        wait_redirect_seconds = 5
        utils.logger.info(f"[DouYinLogin.begin] Login successful then wait for {wait_redirect_seconds} seconds redirect ...")
        await asyncio.sleep(wait_redirect_seconds)

    @retry(stop=stop_after_attempt(600), wait=wait_fixed(1), retry=retry_if_result(lambda value: value is False))
    async def check_login_state(self):
        """Check if the current login status is successful and return True otherwise return False"""
        current_cookie = await self.browser_context.cookies()
        _, cookie_dict = utils.convert_cookies(current_cookie)

        for page in self.browser_context.pages:
            try:
                local_storage = await page.evaluate("() => window.localStorage")
                if local_storage.get("HasUserLogin", "") == "1":
                    return True
            except Exception as e:
                # utils.logger.warn(f"[DouYinLogin] check_login_state waring: {e}")
                await asyncio.sleep(0.1)

        if cookie_dict.get("LOGIN_STATUS") == "1":
            return True

        return False

    async def popup_login_dialog(self):
        """If the login dialog box does not pop up automatically, we will manually click the login button"""
        dialog_selector = "xpath=//div[@id='login-panel-new']"
        try:
            # check dialog box is auto popup and wait for 10 seconds
            await self.context_page.wait_for_selector(dialog_selector, timeout=1000 * 10)
        except Exception as e:
            utils.logger.error(f"[DouYinLogin.popup_login_dialog] login dialog box does not pop up automatically, error: {e}")
            utils.logger.info("[DouYinLogin.popup_login_dialog] login dialog box does not pop up automatically, we will manually click the login button")
            login_button_ele = self.context_page.locator("xpath=//p[text() = '登錄']")
            await login_button_ele.click()
            await asyncio.sleep(0.5)

    async def login_by_qrcode(self):
        utils.logger.info("[DouYinLogin.login_by_qrcode] Begin login douyin by qrcode...")
        qrcode_img_selector = "xpath=//div[@id='animate_qrcode_container']//img"
        base64_qrcode_img = await utils.find_login_qrcode(
            self.context_page,
            selector=qrcode_img_selector
        )
        if not base64_qrcode_img:
            utils.logger.info("[DouYinLogin.login_by_qrcode] login qrcode not found please confirm ...")
            sys.exit()

        partial_show_qrcode = functools.partial(utils.show_qrcode, base64_qrcode_img)
        asyncio.get_running_loop().run_in_executor(executor=None, func=partial_show_qrcode)
        await asyncio.sleep(2)

    async def login_by_mobile(self):
        utils.logger.info("[DouYinLogin.login_by_mobile] Begin login douyin by mobile ...")
        mobile_tap_ele = self.context_page.locator("xpath=//li[text() = '驗證碼登錄']")
        await mobile_tap_ele.click()
        await self.context_page.wait_for_selector("xpath=//article[@class='web-login-mobile-code']")
        mobile_input_ele = self.context_page.locator("xpath=//input[@placeholder='手機號']")
        await mobile_input_ele.fill(self.login_phone)
        await asyncio.sleep(0.5)
        send_sms_code_btn = self.context_page.locator("xpath=//span[text() = '獲取驗證碼']")
        await send_sms_code_btn.click()

        # 檢查是否有滑動驗證碼
        await self.check_page_display_slider(move_step=10, slider_level="easy")
        cache_client = CacheFactory.create_cache(config.CACHE_TYPE_MEMORY)
        max_get_sms_code_time = 60 * 2  # 最長獲取驗證碼的時間爲2分鐘
        while max_get_sms_code_time > 0:
            utils.logger.info(f"[DouYinLogin.login_by_mobile] get douyin sms code from redis remaining time {max_get_sms_code_time}s ...")
            await asyncio.sleep(1)
            sms_code_key = f"dy_{self.login_phone}"
            sms_code_value = cache_client.get(sms_code_key)
            if not sms_code_value:
                max_get_sms_code_time -= 1
                continue

            sms_code_input_ele = self.context_page.locator("xpath=//input[@placeholder='請輸入驗證碼']")
            await sms_code_input_ele.fill(value=sms_code_value.decode())
            await asyncio.sleep(0.5)
            submit_btn_ele = self.context_page.locator("xpath=//button[@class='web-login-button']")
            await submit_btn_ele.click()  # 點擊登錄
            # todo ... 應該還需要檢查驗證碼的正確性有可能輸入的驗證碼不正確
            break

    async def check_page_display_slider(self, move_step: int = 10, slider_level: str = "easy"):
        """
        檢查頁面是否出現滑動驗證碼
        :return:
        """
        # 等待滑動驗證碼的出現
        back_selector = "#captcha-verify-image"
        try:
            await self.context_page.wait_for_selector(selector=back_selector, state="visible", timeout=30 * 1000)
        except PlaywrightTimeoutError:  # 沒有滑動驗證碼，直接返回
            return

        gap_selector = 'xpath=//*[@id="captcha_container"]/div/div[2]/img[2]'
        max_slider_try_times = 20
        slider_verify_success = False
        while not slider_verify_success:
            if max_slider_try_times <= 0:
                utils.logger.error("[DouYinLogin.check_page_display_slider] slider verify failed ...")
                sys.exit()
            try:
                await self.move_slider(back_selector, gap_selector, move_step, slider_level)
                await asyncio.sleep(1)

                # 如果滑塊滑動慢了，或者驗證失敗了，會提示操作過慢，這裏點一下刷新按鈕
                page_content = await self.context_page.content()
                if "操作過慢" in page_content or "提示重新操作" in page_content:
                    utils.logger.info("[DouYinLogin.check_page_display_slider] slider verify failed, retry ...")
                    await self.context_page.click(selector="//a[contains(@class, 'secsdk_captcha_refresh')]")
                    continue

                # 滑動成功後，等待滑塊消失
                await self.context_page.wait_for_selector(selector=back_selector, state="hidden", timeout=1000)
                # 如果滑塊消失了，說明驗證成功了，跳出循環，如果沒有消失，說明驗證失敗了，上面這一行代碼會拋出異常被捕獲後繼續循環滑動驗證碼
                utils.logger.info("[DouYinLogin.check_page_display_slider] slider verify success ...")
                slider_verify_success = True
            except Exception as e:
                utils.logger.error(f"[DouYinLogin.check_page_display_slider] slider verify failed, error: {e}")
                await asyncio.sleep(1)
                max_slider_try_times -= 1
                utils.logger.info(f"[DouYinLogin.check_page_display_slider] remaining slider try times: {max_slider_try_times}")
                continue

    async def move_slider(self, back_selector: str, gap_selector: str, move_step: int = 10, slider_level="easy"):
        """
        Move the slider to the right to complete the verification
        :param back_selector: 滑動驗證碼背景圖片的選擇器
        :param gap_selector:  滑動驗證碼的滑塊選擇器
        :param move_step: 是控制單次移動速度的比例是1/10 默認是1 相當於 傳入的這個距離不管多遠0.1秒鐘移動完 越大越慢
        :param slider_level: 滑塊難度 easy hard,分別對應手機驗證碼的滑塊和驗證碼中間的滑塊
        :return:
        """

        # get slider background image
        slider_back_elements = await self.context_page.wait_for_selector(
            selector=back_selector,
            timeout=1000 * 10,  # wait 10 seconds
        )
        slide_back = str(await slider_back_elements.get_property("src")) # type: ignore

        # get slider gap image
        gap_elements = await self.context_page.wait_for_selector(
            selector=gap_selector,
            timeout=1000 * 10,  # wait 10 seconds
        )
        gap_src = str(await gap_elements.get_property("src")) # type: ignore

        # 識別滑塊位置
        slide_app = utils.Slide(gap=gap_src, bg=slide_back)
        distance = slide_app.discern()

        # 獲取移動軌跡
        tracks = utils.get_tracks(distance, slider_level)
        new_1 = tracks[-1] - (sum(tracks) - distance)
        tracks.pop()
        tracks.append(new_1)

        # 根據軌跡拖拽滑塊到指定位置
        element = await self.context_page.query_selector(gap_selector)
        bounding_box = await element.bounding_box() # type: ignore

        await self.context_page.mouse.move(bounding_box["x"] + bounding_box["width"] / 2, # type: ignore
                                           bounding_box["y"] + bounding_box["height"] / 2) # type: ignore
        # 這裏獲取到x座標中心點位置
        x = bounding_box["x"] + bounding_box["width"] / 2 # type: ignore
        # 模擬滑動操作
        await element.hover() # type: ignore
        await self.context_page.mouse.down()

        for track in tracks:
            # 循環鼠標按照軌跡移動
            # steps 是控制單次移動速度的比例是1/10 默認是1 相當於 傳入的這個距離不管多遠0.1秒鐘移動完 越大越慢
            await self.context_page.mouse.move(x + track, 0, steps=move_step)
            x += track
        await self.context_page.mouse.up()

    async def login_by_cookies(self):
        utils.logger.info("[DouYinLogin.login_by_cookies] Begin login douyin by cookie ...")
        for key, value in utils.convert_str_cookie_to_dict(self.cookie_str).items():
            await self.browser_context.add_cookies([{
                'name': key,
                'value': value,
                'domain': ".douyin.com",
                'path': "/"
            }])

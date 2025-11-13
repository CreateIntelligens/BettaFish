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
import platform
import subprocess
import time
import socket
import signal
from typing import Optional, List, Tuple
import asyncio
from pathlib import Path

from tools import utils


class BrowserLauncher:
    """
    瀏覽器啓動器，用於檢測和啓動用戶的Chrome/Edge瀏覽器
    支持Windows和macOS系統
    """
    
    def __init__(self):
        self.system = platform.system()
        self.browser_process = None
        self.debug_port = None
        
    def detect_browser_paths(self) -> List[str]:
        """
        檢測系統中可用的瀏覽器路徑
        返回按優先級排序的瀏覽器路徑列表
        """
        paths = []
        
        if self.system == "Windows":
            # Windows下的常見Chrome/Edge安裝路徑
            possible_paths = [
                # Chrome路徑
                os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
                # Edge路徑
                os.path.expandvars(r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"),
                os.path.expandvars(r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"),
                # Chrome Beta/Dev/Canary
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome Beta\Application\chrome.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome Dev\Application\chrome.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome SxS\Application\chrome.exe"),
            ]
        elif self.system == "Darwin":  # macOS
            # macOS下的常見Chrome/Edge安裝路徑
            possible_paths = [
                # Chrome路徑
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
                "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
                "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
                # Edge路徑
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Microsoft Edge Beta.app/Contents/MacOS/Microsoft Edge Beta",
                "/Applications/Microsoft Edge Dev.app/Contents/MacOS/Microsoft Edge Dev",
                "/Applications/Microsoft Edge Canary.app/Contents/MacOS/Microsoft Edge Canary",
            ]
        else:
            # Linux等其他系統
            possible_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/google-chrome-beta",
                "/usr/bin/google-chrome-unstable",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/snap/bin/chromium",
                "/usr/bin/microsoft-edge",
                "/usr/bin/microsoft-edge-stable",
                "/usr/bin/microsoft-edge-beta",
                "/usr/bin/microsoft-edge-dev",
            ]
        
        # 檢查路徑是否存在且可執行
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                paths.append(path)
                
        return paths
    
    def find_available_port(self, start_port: int = 9222) -> int:
        """
        查找可用的端口
        """
        port = start_port
        while port < start_port + 100:  # 最多嘗試100個端口
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        
        raise RuntimeError(f"無法找到可用的端口，已嘗試 {start_port} 到 {port-1}")
    
    def launch_browser(self, browser_path: str, debug_port: int, headless: bool = False,
                      user_data_dir: Optional[str] = None) -> subprocess.Popen:
        """
        啓動瀏覽器進程
        """
        # 基本啓動參數
        args = [
            browser_path,
            f"--remote-debugging-port={debug_port}",
            "--remote-debugging-address=0.0.0.0",  # 允許遠程訪問
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=TranslateUI",
            "--disable-ipc-flooding-protection",
            "--disable-hang-monitor",
            "--disable-prompt-on-repost",
            "--disable-sync",
            "--disable-dev-shm-usage",  # 避免共享內存問題
            "--no-sandbox",  # 在CDP模式下關閉沙箱
            # 🔥 關鍵反檢測參數
            "--disable-blink-features=AutomationControlled",  # 禁用自動化控制標記
            "--exclude-switches=enable-automation",  # 排除自動化開關
            "--disable-infobars",  # 禁用信息欄
        ]

        # 無頭模式
        if headless:
            args.extend([
                "--headless=new",  # 使用新的headless模式
                "--disable-gpu",
            ])
        else:
            # 非無頭模式的額外參數
            args.extend([
                "--start-maximized",  # 最大化窗口,更像真實用戶
            ])
        
        # 用戶數據目錄
        if user_data_dir:
            args.append(f"--user-data-dir={user_data_dir}")
        
        utils.logger.info(f"[BrowserLauncher] 啓動瀏覽器: {browser_path}")
        utils.logger.info(f"[BrowserLauncher] 調試端口: {debug_port}")
        utils.logger.info(f"[BrowserLauncher] 無頭模式: {headless}")
        
        try:
            # 在Windows上，使用CREATE_NEW_PROCESS_GROUP避免Ctrl+C影響子進程
            if self.system == "Windows":
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # 創建新的進程組
                )

            self.browser_process = process
            return process
            
        except Exception as e:
            utils.logger.error(f"[BrowserLauncher] 啓動瀏覽器失敗: {e}")
            raise
    
    def wait_for_browser_ready(self, debug_port: int, timeout: int = 30) -> bool:
        """
        等待瀏覽器準備就緒
        """
        utils.logger.info(f"[BrowserLauncher] 等待瀏覽器在端口 {debug_port} 上準備就緒...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', debug_port))
                    if result == 0:
                        utils.logger.info(f"[BrowserLauncher] 瀏覽器已在端口 {debug_port} 上準備就緒")
                        return True
            except Exception:
                pass
            
            time.sleep(0.5)
        
        utils.logger.error(f"[BrowserLauncher] 瀏覽器在 {timeout} 秒內未能準備就緒")
        return False
    
    def get_browser_info(self, browser_path: str) -> Tuple[str, str]:
        """
        獲取瀏覽器信息（名稱和版本）
        """
        try:
            if "chrome" in browser_path.lower():
                name = "Google Chrome"
            elif "edge" in browser_path.lower() or "msedge" in browser_path.lower():
                name = "Microsoft Edge"
            elif "chromium" in browser_path.lower():
                name = "Chromium"
            else:
                name = "Unknown Browser"
            
            # 嘗試獲取版本信息
            try:
                result = subprocess.run([browser_path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                version = result.stdout.strip() if result.stdout else "Unknown Version"
            except:
                version = "Unknown Version"
            
            return name, version
            
        except Exception:
            return "Unknown Browser", "Unknown Version"
    
    def cleanup(self):
        """
        清理資源，關閉瀏覽器進程
        """
        if not self.browser_process:
            return

        process = self.browser_process

        if process.poll() is not None:
            utils.logger.info("[BrowserLauncher] 瀏覽器進程已退出，無需清理")
            self.browser_process = None
            return

        utils.logger.info("[BrowserLauncher] 正在關閉瀏覽器進程...")

        try:
            if self.system == "Windows":
                # 先嚐試正常終止
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    utils.logger.warning("[BrowserLauncher] 正常終止超時，使用taskkill強制結束")
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        capture_output=True,
                        check=False,
                    )
                    process.wait(timeout=5)
            else:
                pgid = os.getpgid(process.pid)
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    utils.logger.info("[BrowserLauncher] 瀏覽器進程組不存在，可能已退出")
                else:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        utils.logger.warning("[BrowserLauncher] 優雅關閉超時，發送SIGKILL")
                        os.killpg(pgid, signal.SIGKILL)
                        process.wait(timeout=5)

            utils.logger.info("[BrowserLauncher] 瀏覽器進程已關閉")
        except Exception as e:
            utils.logger.warning(f"[BrowserLauncher] 關閉瀏覽器進程時出錯: {e}")
        finally:
            self.browser_process = None

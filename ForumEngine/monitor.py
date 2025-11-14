"""
日誌監控器 - 實時監控三個log文件中的SummaryNode輸出
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import re
import json
from typing import Dict, Optional, List
from threading import Lock
from loguru import logger

# 導入論壇主持人模塊
try:
    from .llm_host import generate_host_speech
    HOST_AVAILABLE = True
except ImportError:
    logger.warning("ForumEngine: 論壇主持人模塊未找到，將以純監控模式運行")
    HOST_AVAILABLE = False

class LogMonitor:
    """基於文件變化的智能日誌監控器"""
   
    def __init__(self, log_dir: str = "logs"):
        """初始化日誌監控器"""
        self.log_dir = Path(log_dir)
        self.forum_log_file = self.log_dir / "forum.log"
       
        # 要監控的日誌文件
        self.monitored_logs = {
            'insight': self.log_dir / 'insight.log',
            'media': self.log_dir / 'media.log',
            'query': self.log_dir / 'query.log'
        }
       
        # 監控狀態
        self.is_monitoring = False
        self.monitor_thread = None
        self.file_positions = {}  # 記錄每個文件的讀取位置
        self.file_line_counts = {}  # 記錄每個文件的行數
        self.is_searching = False  # 是否正在搜索
        self.search_inactive_count = 0  # 搜索非活躍計數器
        self.write_lock = Lock()  # 寫入鎖，防止併發寫入衝突
        
        # 主持人相關狀態
        self.agent_speeches_buffer = []  # agent發言緩衝區
        self.host_speech_threshold = 5  # 每5條agent發言觸發一次主持人發言
        self.is_host_generating = False  # 主持人是否正在生成發言

        # 目標節點識別模式
        # 1. 類名（舊格式可能包含）
        # 2. 完整模塊路徑（實際日誌格式，包含引擎前綴）
        # 3. 部分模塊路徑（兼容性）
        # 4. 關鍵標識文本
        self.target_node_patterns = [
            'FirstSummaryNode',  # 類名
            'ReflectionSummaryNode',  # 類名
            'InsightEngine.nodes.summary_node',  # InsightEngine完整路徑
            'MediaEngine.nodes.summary_node',  # MediaEngine完整路徑
            'QueryEngine.nodes.summary_node',  # QueryEngine完整路徑
            'nodes.summary_node',  # 模塊路徑（兼容性，用於部分匹配）
            '正在生成首次段落總結',  # FirstSummaryNode的標識
            '正在生成反思總結',  # ReflectionSummaryNode的標識
        ]

        # 多行內容捕獲狀態
        self.capturing_json = {}  # 每個app的JSON捕獲狀態
        self.json_buffer = {}     # 每個app的JSON緩衝區
        self.json_start_line = {} # 每個app的JSON開始行
        self.in_error_block = {}  # 每個app是否在ERROR塊中

        # 確保logs目錄存在
        self.log_dir.mkdir(exist_ok=True)
   
    def clear_forum_log(self):
        """清空forum.log文件"""
        try:
            if self.forum_log_file.exists():
                self.forum_log_file.unlink()
           
            # 創建新的forum.log文件並寫入開始標記
            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 使用write_to_forum_log函數來寫入開始標記，確保格式一致
            with open(self.forum_log_file, 'w', encoding='utf-8') as f:
                pass  # 先創建空文件
            self.write_to_forum_log(f"=== ForumEngine 監控開始 - {start_time} ===", "SYSTEM")
               
            logger.info(f"ForumEngine: forum.log 已清空並初始化")
            
            # 重置JSON捕獲狀態
            self.capturing_json = {}
            self.json_buffer = {}
            self.json_start_line = {}
            self.in_error_block = {}
            
            # 重置主持人相關狀態
            self.agent_speeches_buffer = []
            self.is_host_generating = False
           
        except Exception as e:
            logger.exception(f"ForumEngine: 清空forum.log失敗: {e}")
   
    def write_to_forum_log(self, content: str, source: str = None):
        """寫入內容到forum.log（線程安全）"""
        try:
            with self.write_lock:  # 使用鎖確保線程安全
                with open(self.forum_log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    # 將內容中的實際換行符轉換爲\n字符串，確保整個記錄在一行
                    content_one_line = content.replace('\n', '\\n').replace('\r', '\\r')
                    # 如果提供了來源標籤，則在時間戳後添加
                    if source:
                        f.write(f"[{timestamp}] [{source}] {content_one_line}\n")
                    else:
                        f.write(f"[{timestamp}] {content_one_line}\n")
                    f.flush()
        except Exception as e:
            logger.exception(f"ForumEngine: 寫入forum.log失敗: {e}")

    def get_log_level(self, line: str) -> Optional[str]:
        """檢測日誌行的級別（INFO/ERROR/WARNING/DEBUG等）

        支持loguru格式：YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...

        Returns:
            'INFO', 'ERROR', 'WARNING', 'DEBUG' 或 None（無法識別）
        """
        # 檢查loguru格式：YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...
        # 匹配模式：| LEVEL | 或 | LEVEL     |
        match = re.search(r'\|\s*(INFO|ERROR|WARNING|DEBUG|TRACE|CRITICAL)\s*\|', line)
        if match:
            return match.group(1)
        return None

    def is_target_log_line(self, line: str) -> bool:
        """檢查是否是目標日誌行（SummaryNode）

        支持多種識別方式：
        1. 類名：FirstSummaryNode, ReflectionSummaryNode
        2. 完整模塊路徑：InsightEngine.nodes.summary_node、MediaEngine.nodes.summary_node、QueryEngine.nodes.summary_node
        3. 部分模塊路徑：nodes.summary_node（兼容性）
        4. 關鍵標識文本：正在生成首次段落總結、正在生成反思總結

        排除條件：
        - ERROR 級別的日誌（錯誤日誌不應被識別為目標節點）
        - 包含錯誤關鍵詞的日誌（JSON解析失敗、JSON修復失敗等）
        """
        # 排除 ERROR 級別的日誌
        log_level = self.get_log_level(line)
        if log_level == 'ERROR':
            return False

        # 兼容舊檢查方式
        if "| ERROR" in line or "| ERROR    |" in line:
            return False

        # 排除包含錯誤關鍵詞的日誌
        error_keywords = ["JSON解析失敗", "JSON修復失敗", "Traceback", "File \""]
        for keyword in error_keywords:
            if keyword in line:
                return False

        # 檢查是否包含目標節點模式
        for pattern in self.target_node_patterns:
            if pattern in line:
                return True
        return False
    
    def is_valuable_content(self, line: str) -> bool:
        """判斷是否是有價值的內容（排除短小的提示信息和錯誤信息）"""
        # 如果包含"清理後的輸出"，則認爲是有價值的
        if "清理後的輸出" in line:
            return True
        
        # 排除常見的短小提示信息和錯誤信息
        exclude_patterns = [
            "JSON解析失敗",
            "JSON修復失敗",
            "直接使用清理後的文本",
            "JSON解析成功",
            "成功生成",
            "已更新段落",
            "正在生成",
            "開始處理",
            "處理完成",
            "已讀取HOST發言",
            "讀取HOST發言失敗",
            "未找到HOST發言",
            "調試輸出",
            "信息記錄"
        ]
        
        for pattern in exclude_patterns:
            if pattern in line:
                return False

        # 如果行長度過短，也認為不是有價值的內容
        # 移除時間戳：支持舊格式和新格式
        clean_line = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', line)
        clean_line = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', clean_line)
        clean_line = clean_line.strip()
        if len(clean_line) < 30:  # 閾值可以調整
            return False
            
        return True
    
    def is_json_start_line(self, line: str) -> bool:
        """判斷是否是JSON開始行"""
        return "清理後的輸出: {" in line
    
    def is_json_end_line(self, line: str) -> bool:
        """判斷是否是JSON結束行

        只判斷純粹的結束標記行，不包含任何日誌格式信息（時間戳等）。
        如果行包含時間戳，應該先清理再判斷，但這裡返回False表示需要進一步處理。
        """
        stripped = line.strip()
        
        # 如果行包含时间戳（旧格式或新格式），说明不是纯粹的结束行
        # 旧格式：[HH:MM:SS]
        if re.match(r'^\[\d{2}:\d{2}:\d{2}\]', stripped):
            return False
        # 新格式：YYYY-MM-DD HH:mm:ss.SSS
        if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}', stripped):
            return False
        
        # 不包含时间戳的行，检查是否是纯结束标记
        if stripped == "}" or stripped == "] }":
            return True
        return False
    
    def extract_json_content(self, json_lines: List[str]) -> Optional[str]:
        """從多行中提取並解析JSON內容"""
        try:
            # 找到JSON開始的位置
            json_start_idx = -1
            for i, line in enumerate(json_lines):
                if "清理後的輸出: {" in line:
                    json_start_idx = i
                    break
            
            if json_start_idx == -1:
                return None
            
            # 提取JSON部分
            first_line = json_lines[json_start_idx]
            json_start_pos = first_line.find("清理後的輸出: {")
            if json_start_pos == -1:
                return None
            
            json_part = first_line[json_start_pos + len("清理後的輸出: "):]
            
            # 如果第一行就包含完整JSON，直接處理
            if json_part.strip().endswith("}") and json_part.count("{") == json_part.count("}"):
                try:
                    json_obj = json.loads(json_part.strip())
                    return self.format_json_content(json_obj)
                except json.JSONDecodeError:
                    # 單行JSON解析失敗，嘗試修復
                    fixed_json = self.fix_json_string(json_part.strip())
                    if fixed_json:
                        try:
                            json_obj = json.loads(fixed_json)
                            return self.format_json_content(json_obj)
                        except json.JSONDecodeError:
                            pass
                    return None
            
            # 處理多行JSON
            json_text = json_part
            for line in json_lines[json_start_idx + 1:]:
                # 移除時間戳：支持舊格式 [HH:MM:SS] 和新格式 loguru (YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...)
                # 舊格式：[HH:MM:SS]
                clean_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)
                # 新格式：移除 loguru 格式的时间戳和级别信息
                # 格式: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
                clean_line = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', clean_line)
                json_text += clean_line
            
            # 嘗試解析JSON
            try:
                json_obj = json.loads(json_text.strip())
                return self.format_json_content(json_obj)
            except json.JSONDecodeError:
                # 多行JSON解析失敗，嘗試修復
                fixed_json = self.fix_json_string(json_text.strip())
                if fixed_json:
                    try:
                        json_obj = json.loads(fixed_json)
                        return self.format_json_content(json_obj)
                    except json.JSONDecodeError:
                        pass
                return None
            
        except Exception as e:
            # 其他異常也不打印錯誤信息，直接返回None
            return None
    
    def format_json_content(self, json_obj: dict) -> str:
        """格式化JSON內容爲可讀形式"""
        try:
            # 提取主要內容，優先選擇反思總結，其次是首次總結
            content = None
            
            if "updated_paragraph_latest_state" in json_obj:
                content = json_obj["updated_paragraph_latest_state"]
            elif "paragraph_latest_state" in json_obj:
                content = json_obj["paragraph_latest_state"]
            
            # 如果找到了內容，直接返回（保持換行符爲\n）
            if content:
                return content
            
            # 如果沒有找到預期的字段，返回整個JSON的字符串表示
            return f"清理後的輸出: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"
            
        except Exception as e:
            logger.exception(f"ForumEngine: 格式化JSON時出錯: {e}")
            return f"清理後的輸出: {json.dumps(json_obj, ensure_ascii=False, indent=2)}"

    def extract_node_content(self, line: str) -> Optional[str]:
        """提取節點內容，去除時間戳、節點名稱等前綴"""
        content = line

        # 移除時間戳部分：支持舊格式和新格式
        # 舊格式: [HH:MM:SS]
        match_old = re.search(r'\[\d{2}:\d{2}:\d{2}\]\s*(.+)', content)
        if match_old:
            content = match_old.group(1).strip()
        else:
            # 新格式: YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
            match_new = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*(.+)', content)
            if match_new:
                content = match_new.group(1).strip()

        if not content:
            return line.strip()

        # 移除所有的方括號標籤（包括節點名稱和應用名稱）
        content = re.sub(r'^\[.*?\]\s*', '', content)

        # 繼續移除可能的多個連續標籤
        while re.match(r'^\[.*?\]\s*', content):
            content = re.sub(r'^\[.*?\]\s*', '', content)

        # 移除常見前綴（如"首次總結: "、"反思總結: "等）
        prefixes_to_remove = [
            "首次總結: ",
            "反思總結: ",
            "清理後的輸出: "
        ]

        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):]
                break

        # 移除可能存在的應用名標籤（不在方括號內的）
        app_names = ['INSIGHT', 'MEDIA', 'QUERY']
        for app_name in app_names:
            # 移除單獨的APP_NAME（在行首）
            content = re.sub(rf'^{app_name}\s+', '', content, flags=re.IGNORECASE)

        # 清理多餘的空格
        content = re.sub(r'\s+', ' ', content)

        return content.strip()
   
    def get_file_size(self, file_path: Path) -> int:
        """獲取文件大小"""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except:
            return 0
   
    def get_file_line_count(self, file_path: Path) -> int:
        """獲取文件行數"""
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
   
    def read_new_lines(self, file_path: Path, app_name: str) -> List[str]:
        """讀取文件中的新行"""
        new_lines = []
       
        try:
            if not file_path.exists():
                return new_lines
           
            current_size = self.get_file_size(file_path)
            last_position = self.file_positions.get(app_name, 0)
           
            # 如果文件變小了，說明被清空了，重新從頭開始
            if current_size < last_position:
                last_position = 0
                # 重置JSON捕獲狀態
                self.capturing_json[app_name] = False
                self.json_buffer[app_name] = []
                self.in_error_block[app_name] = False
           
            if current_size > last_position:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    new_lines = new_content.split('\n')
                   
                    # 更新位置
                    self.file_positions[app_name] = f.tell()
                   
                    # 過濾空行
                    new_lines = [line.strip() for line in new_lines if line.strip()]
                   
        except Exception as e:
            logger.exception(f"ForumEngine: 讀取{app_name}日誌失敗: {e}")
       
        return new_lines
   
    def process_lines_for_json(self, lines: List[str], app_name: str) -> List[str]:
        """處理行以捕獲多行JSON內容

        實現ERROR塊過濾：如果遇到ERROR級別的日誌，拒絕處理直到遇到下一個INFO級別的日誌
        """
        captured_contents = []

        # 初始化狀態
        if app_name not in self.capturing_json:
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
        if app_name not in self.in_error_block:
            self.in_error_block[app_name] = False
        
        for line in lines:
            if not line.strip():
                continue
            
            # 首先检查日志级别，更新ERROR块状态
            log_level = self.get_log_level(line)
            if log_level == 'ERROR':
                # 遇到ERROR，进入ERROR块状态
                self.in_error_block[app_name] = True
                # 如果正在捕获JSON，立即停止并清空缓冲区
                if self.capturing_json[app_name]:
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                # 跳过当前行，不处理
                continue
            elif log_level == 'INFO':
                # 遇到INFO，退出ERROR块状态
                self.in_error_block[app_name] = False
            # 其他级别（WARNING、DEBUG等）保持当前状态
            
            # 如果在ERROR块中，拒绝处理所有内容
            if self.in_error_block[app_name]:
                # 如果正在捕获JSON，立即停止并清空缓冲区
                if self.capturing_json[app_name]:
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
                # 跳过当前行，不处理
                continue
                
            # 檢查是否是目標節點行和JSON開始標記
            is_target = self.is_target_log_line(line)
            is_json_start = self.is_json_start_line(line)

            # 只有目標節點（SummaryNode）的JSON輸出才應該被捕獲
            # 過濾掉SearchNode等其他節點的輸出（它們不是目標節點，即使有JSON也不會被捕獲）
            if is_target and is_json_start:
                # 開始捕獲JSON（必須是目標節點且包含"清理後的輸出: {"）
                self.capturing_json[app_name] = True
                self.json_buffer[app_name] = [line]
                self.json_start_line[app_name] = line

                # 檢查是否是單行JSON
                if line.strip().endswith("}"):
                    # 單行JSON，立即處理
                    content = self.extract_json_content([line])
                    if content:  # 只有成功解析的內容才會被記錄
                        # 去除重複的標籤和格式化
                        clean_content = self._clean_content_tags(content, app_name)
                        captured_contents.append(f"{clean_content}")
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []

            elif is_target and self.is_valuable_content(line):
                # 其他有價值的SummaryNode內容（必須是目標節點且有價值）
                clean_content = self._clean_content_tags(self.extract_node_content(line), app_name)
                captured_contents.append(f"{clean_content}")
                    
            elif self.capturing_json[app_name]:
                # 正在捕獲JSON的後續行
                self.json_buffer[app_name].append(line)

                # 檢查是否是JSON結束
                # 先清理時間戳，然後判斷清理後的行是否是結束標記
                cleaned_line = line.strip()
                # 清理舊格式時間戳：[HH:MM:SS]
                cleaned_line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', cleaned_line)
                # 清理新格式時間戳：YYYY-MM-DD HH:mm:ss.SSS | LEVEL | module:function:line -
                cleaned_line = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*\|\s*[A-Z]+\s*\|\s*[^|]+?\s*-\s*', '', cleaned_line)
                cleaned_line = cleaned_line.strip()

                # 清理後判斷是否是結束標記
                if cleaned_line == "}" or cleaned_line == "] }":
                    # JSON結束，處理完整的JSON
                    content = self.extract_json_content(self.json_buffer[app_name])
                    if content:  # 只有成功解析的內容纔會被記錄
                        # 去除重複的標籤和格式化
                        clean_content = self._clean_content_tags(content, app_name)
                        captured_contents.append(f"{clean_content}")
                    
                    # 重置狀態
                    self.capturing_json[app_name] = False
                    self.json_buffer[app_name] = []
        
        return captured_contents
    
    def _trigger_host_speech(self):
        """觸發主持人發言（同步執行）"""
        if not HOST_AVAILABLE or self.is_host_generating:
            return
        
        try:
            # 設置生成標誌
            self.is_host_generating = True
            
            # 獲取緩衝區的5條發言
            recent_speeches = self.agent_speeches_buffer[:5]
            if len(recent_speeches) < 5:
                self.is_host_generating = False
                return
            
            logger.info("ForumEngine: 正在生成主持人發言...")
            
            # 調用主持人生成發言（傳入最近5條）
            host_speech = generate_host_speech(recent_speeches)
            
            if host_speech:
                # 寫入主持人發言到forum.log
                self.write_to_forum_log(host_speech, "HOST")
                logger.info(f"ForumEngine: 主持人發言已記錄")
                
                # 清空已處理的5條發言
                self.agent_speeches_buffer = self.agent_speeches_buffer[5:]
            else:
                logger.error("ForumEngine: 主持人發言生成失敗")
            
            # 重置生成標誌
            self.is_host_generating = False
                
        except Exception as e:
            logger.exception(f"ForumEngine: 觸發主持人發言時出錯: {e}")
            self.is_host_generating = False
    
    def _clean_content_tags(self, content: str, app_name: str) -> str:
        """清理內容中的重複標籤和多餘前綴"""
        if not content:
            return content
            
        # 先去除所有可能的標籤格式（包括 [INSIGHT]、[MEDIA]、[QUERY] 等）
        # 使用更強力的清理方式
        all_app_names = ['INSIGHT', 'MEDIA', 'QUERY']
        
        for name in all_app_names:
            # 去除 [APP_NAME] 格式（大小寫不敏感）
            content = re.sub(rf'\[{name}\]\s*', '', content, flags=re.IGNORECASE)
            # 去除單獨的 APP_NAME 格式
            content = re.sub(rf'^{name}\s+', '', content, flags=re.IGNORECASE)
        
        # 去除任何其他的方括號標籤
        content = re.sub(r'^\[.*?\]\s*', '', content)
        
        # 去除可能的重複空格
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
   
    def monitor_logs(self):
        """智能監控日誌文件"""
        logger.info("ForumEngine: 論壇創建中...")
       
        # 初始化文件行數和位置 - 記錄當前狀態作爲基線
        for app_name, log_file in self.monitored_logs.items():
            self.file_line_counts[app_name] = self.get_file_line_count(log_file)
            self.file_positions[app_name] = self.get_file_size(log_file)
            self.capturing_json[app_name] = False
            self.json_buffer[app_name] = []
            self.in_error_block[app_name] = False
            # logger.info(f"ForumEngine: {app_name} 基線行數: {self.file_line_counts[app_name]}")
       
        while self.is_monitoring:
            try:
                # 同時檢測三個log文件的變化
                any_growth = False
                any_shrink = False
                captured_any = False
               
                # 爲每個log文件獨立處理
                for app_name, log_file in self.monitored_logs.items():
                    current_lines = self.get_file_line_count(log_file)
                    previous_lines = self.file_line_counts.get(app_name, 0)
                   
                    if current_lines > previous_lines:
                        any_growth = True
                        # 立即讀取新增內容
                        new_lines = self.read_new_lines(log_file, app_name)
                       
                        # 先檢查是否需要觸發搜索（只觸發一次）
                        if not self.is_searching:
                            for line in new_lines:
                                # 檢查是否包含目標節點模式（支持多種格式）
                                if line.strip() and self.is_target_log_line(line):
                                    # 進一步確認是首次總結節點（FirstSummaryNode或包含"正在生成首次段落總結"）
                                    if 'FirstSummaryNode' in line or '正在生成首次段落總結' in line:
                                        logger.info(f"ForumEngine: 在{app_name}中檢測到第一次論壇發表內容")
                                        self.is_searching = True
                                        self.search_inactive_count = 0
                                        # 清空forum.log開始新會話
                                        self.clear_forum_log()
                                        break  # 找到一個就夠了，跳出循環
                       
                        # 處理所有新增內容（如果正在搜索狀態）
                        if self.is_searching:
                            # 使用新的處理邏輯
                            captured_contents = self.process_lines_for_json(new_lines, app_name)
                            
                            for content in captured_contents:
                                # 將app_name轉換爲大寫作爲標籤（如 insight -> INSIGHT）
                                source_tag = app_name.upper()
                                self.write_to_forum_log(content, source_tag)
                                # logger.info(f"ForumEngine: 捕獲 - {content}")
                                captured_any = True
                                
                                # 將發言添加到緩衝區（格式化爲完整的日誌行）
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                log_line = f"[{timestamp}] [{source_tag}] {content}"
                                self.agent_speeches_buffer.append(log_line)
                                
                                # 檢查是否需要觸發主持人發言
                                if len(self.agent_speeches_buffer) >= self.host_speech_threshold and not self.is_host_generating:
                                    # 同步觸發主持人發言
                                    self._trigger_host_speech()
                   
                    elif current_lines < previous_lines:
                        any_shrink = True
                        # logger.info(f"ForumEngine: 檢測到 {app_name} 日誌縮短，將重置基線")
                        # 重置文件位置到新的文件末尾
                        self.file_positions[app_name] = self.get_file_size(log_file)
                        # 重置JSON捕獲狀態
                        self.capturing_json[app_name] = False
                        self.json_buffer[app_name] = []
                        self.in_error_block[app_name] = False
                   
                    # 更新行數記錄
                    self.file_line_counts[app_name] = current_lines
               
                # 檢查是否應該結束當前搜索會話
                if self.is_searching:
                    if any_shrink:
                        # log變短，結束當前搜索會話，重置爲等待狀態
                        # logger.info("ForumEngine: 日誌縮短，結束當前搜索會話，回到等待狀態")
                        self.is_searching = False
                        self.search_inactive_count = 0
                        # 重置主持人相關狀態
                        self.agent_speeches_buffer = []
                        self.is_host_generating = False
                        # 寫入結束標記
                        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.write_to_forum_log(f"=== ForumEngine 論壇結束 - {end_time} ===", "SYSTEM")
                        # logger.info("ForumEngine: 已重置基線，等待下次FirstSummaryNode觸發")
                    elif not any_growth and not captured_any:
                        # 沒有增長也沒有捕獲內容，增加非活躍計數
                        self.search_inactive_count += 1
                        if self.search_inactive_count >= 7200:  # 超時無活動自動結束
                            logger.info("ForumEngine: 長時間無活動，結束論壇")
                            self.is_searching = False
                            self.search_inactive_count = 0
                            # 重置主持人相關狀態
                            self.agent_speeches_buffer = []
                            self.is_host_generating = False
                            # 寫入結束標記
                            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.write_to_forum_log(f"=== ForumEngine 論壇結束 - {end_time} ===", "SYSTEM")
                    else:
                        self.search_inactive_count = 0  # 重置計數器
               
                # 短暫休眠
                time.sleep(1)
               
            except Exception as e:
                logger.exception(f"ForumEngine: 論壇記錄中出錯: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)
       
        logger.info("ForumEngine: 停止論壇日誌文件")
   
    def start_monitoring(self):
        """開始智能監控"""
        if self.is_monitoring:
            logger.info("ForumEngine: 論壇已經在運行中")
            return False
       
        try:
            # 啓動監控
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_logs, daemon=True)
            self.monitor_thread.start()
           
            logger.info("ForumEngine: 論壇已啓動")
            return True
           
        except Exception as e:
            logger.exception(f"ForumEngine: 啓動論壇失敗: {e}")
            self.is_monitoring = False
            return False
   
    def stop_monitoring(self):
        """停止監控"""
        if not self.is_monitoring:
            logger.info("ForumEngine: 論壇未運行")
            return
       
        try:
            self.is_monitoring = False
           
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
           
            # 寫入結束標記
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.write_to_forum_log(f"=== ForumEngine 論壇結束 - {end_time} ===", "SYSTEM")
           
            logger.info("ForumEngine: 論壇已停止")
           
        except Exception as e:
            logger.exception(f"ForumEngine: 停止論壇失敗: {e}")
   
    def get_forum_log_content(self) -> List[str]:
        """獲取forum.log的內容"""
        try:
            if not self.forum_log_file.exists():
                return []
           
            with open(self.forum_log_file, 'r', encoding='utf-8') as f:
                return [line.rstrip('\n\r') for line in f.readlines()]
               
        except Exception as e:
            logger.exception(f"ForumEngine: 讀取forum.log失敗: {e}")
            return []

    def fix_json_string(self, json_text: str) -> str:
        """修復JSON字符串中的常見問題，特別是未轉義的雙引號"""
        try:
            # 嘗試直接解析，如果成功則返回原文本
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
        
        # 修復未轉義的雙引號問題
        # 這是一個更智能的修復方法，專門處理字符串值中的雙引號
        
        try:
            # 使用狀態機方法修復JSON
            # 遍歷字符，跟蹤是否在字符串值內部
            
            fixed_text = ""
            i = 0
            in_string = False
            escape_next = False
            
            while i < len(json_text):
                char = json_text[i]
                
                if escape_next:
                    # 處理轉義字符
                    fixed_text += char
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    # 轉義字符
                    fixed_text += char
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not escape_next:
                    # 遇到雙引號
                    if in_string:
                        # 在字符串內部，檢查下一個字符
                        # 如果下一個字符是冒號或者逗號或者大括號，說明這是字符串結束
                        next_char_pos = i + 1
                        while next_char_pos < len(json_text) and json_text[next_char_pos].isspace():
                            next_char_pos += 1
                        
                        if next_char_pos < len(json_text):
                            next_char = json_text[next_char_pos]
                            if next_char in [':', ',', '}']:
                                # 這是字符串結束，退出字符串狀態
                                in_string = False
                                fixed_text += char
                            else:
                                # 這是字符串內部的引號，需要轉義
                                fixed_text += '\\"'
                        else:
                            # 文件結束，退出字符串狀態
                            in_string = False
                            fixed_text += char
                    else:
                        # 字符串開始
                        in_string = True
                        fixed_text += char
                else:
                    # 其他字符
                    fixed_text += char
                
                i += 1
            
            # 嘗試解析修復後的JSON
            try:
                json.loads(fixed_text)
                return fixed_text
            except json.JSONDecodeError:
                # 修復失敗，返回None
                return None
                
        except Exception:
            return None

# 全局監控器實例
_monitor_instance = None

def get_monitor() -> LogMonitor:
    """獲取全局監控器實例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = LogMonitor()
    return _monitor_instance

def start_forum_monitoring():
    """啓動ForumEngine智能監控"""
    return get_monitor().start_monitoring()

def stop_forum_monitoring():
    """停止ForumEngine監控"""
    get_monitor().stop_monitoring()

def get_forum_log():
    """獲取forum.log內容"""
    return get_monitor().get_forum_log_content()
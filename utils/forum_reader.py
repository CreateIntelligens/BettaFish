"""
Forum日誌讀取工具
用於讀取forum.log中的最新HOST發言
"""

import re
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

def get_latest_host_speech(log_dir: str = "logs") -> Optional[str]:
    """
    獲取forum.log中最新的HOST發言
    
    Args:
        log_dir: 日誌目錄路徑
        
    Returns:
        最新的HOST發言內容，如果沒有則返回None
    """
    try:
        forum_log_path = Path(log_dir) / "forum.log"
        
        if not forum_log_path.exists():
            logger.debug("forum.log文件不存在")
            return None
            
        with open(forum_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 從後往前查找最新的HOST發言
        host_speech = None
        for line in reversed(lines):
            # 匹配格式: [時間] [HOST] 內容
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[HOST\]\s*(.+)', line)
            if match:
                _, content = match.groups()
                # 處理轉義的換行符，還原爲實際換行
                host_speech = content.replace('\\n', '\n').strip()
                break
        
        if host_speech:
            logger.info(f"找到最新的HOST發言，長度: {len(host_speech)}字符")
        else:
            logger.debug("未找到HOST發言")
            
        return host_speech
        
    except Exception as e:
        logger.error(f"讀取forum.log失敗: {str(e)}")
        return None


def get_all_host_speeches(log_dir: str = "logs") -> List[Dict[str, str]]:
    """
    獲取forum.log中所有的HOST發言
    
    Args:
        log_dir: 日誌目錄路徑
        
    Returns:
        包含所有HOST發言的列表，每個元素是包含timestamp和content的字典
    """
    try:
        forum_log_path = Path(log_dir) / "forum.log"
        
        if not forum_log_path.exists():
            logger.debug("forum.log文件不存在")
            return []
            
        with open(forum_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        host_speeches = []
        for line in lines:
            # 匹配格式: [時間] [HOST] 內容
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[HOST\]\s*(.+)', line)
            if match:
                timestamp, content = match.groups()
                # 處理轉義的換行符
                content = content.replace('\\n', '\n').strip()
                host_speeches.append({
                    'timestamp': timestamp,
                    'content': content
                })
        
        logger.info(f"找到{len(host_speeches)}條HOST發言")
        return host_speeches
        
    except Exception as e:
        logger.error(f"讀取forum.log失敗: {str(e)}")
        return []


def get_recent_agent_speeches(log_dir: str = "logs", limit: int = 5) -> List[Dict[str, str]]:
    """
    獲取forum.log中最近的Agent發言（不包括HOST）
    
    Args:
        log_dir: 日誌目錄路徑
        limit: 返回的最大發言數量
        
    Returns:
        包含最近Agent發言的列表
    """
    try:
        forum_log_path = Path(log_dir) / "forum.log"
        
        if not forum_log_path.exists():
            return []
            
        with open(forum_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        agent_speeches = []
        for line in reversed(lines):  # 從後往前讀取
            # 匹配格式: [時間] [AGENT_NAME] 內容
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[(INSIGHT|MEDIA|QUERY)\]\s*(.+)', line)
            if match:
                timestamp, agent, content = match.groups()
                # 處理轉義的換行符
                content = content.replace('\\n', '\n').strip()
                agent_speeches.append({
                    'timestamp': timestamp,
                    'agent': agent,
                    'content': content
                })
                if len(agent_speeches) >= limit:
                    break
        
        agent_speeches.reverse()  # 恢復時間順序
        return agent_speeches
        
    except Exception as e:
        logger.error(f"讀取forum.log失敗: {str(e)}")
        return []


def format_host_speech_for_prompt(host_speech: str) -> str:
    """
    格式化HOST發言，用於添加到prompt中
    
    Args:
        host_speech: HOST發言內容
        
    Returns:
        格式化後的內容
    """
    if not host_speech:
        return ""
    
    return f"""
### 論壇主持人最新總結
以下是論壇主持人對各Agent討論的最新總結和引導，請參考其中的觀點和建議：

{host_speech}

---
"""

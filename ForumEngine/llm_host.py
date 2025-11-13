"""
論壇主持人模塊
使用硅基流動的Qwen3模型作爲論壇主持人，引導多個agent進行討論
"""

from openai import OpenAI
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# 添加項目根目錄到Python路徑以導入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FORUM_HOST_API_KEY, FORUM_HOST_BASE_URL, FORUM_HOST_MODEL_NAME

# 添加utils目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG


class ForumHost:
    """
    論壇主持人類
    使用Qwen3-235B模型作爲智能主持人
    """
    
    def __init__(self, api_key: str = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """
        初始化論壇主持人
        
        Args:
            api_key: 硅基流動API密鑰，如果不提供則從配置文件讀取
            base_url: 接口基礎地址，默認使用配置文件提供的SiliconFlow地址
        """
        self.api_key = api_key or FORUM_HOST_API_KEY

        if not self.api_key:
            raise ValueError("未找到硅基流動API密鑰，請在config.py中設置FORUM_HOST_API_KEY")

        self.base_url = base_url or FORUM_HOST_BASE_URL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = model_name or FORUM_HOST_MODEL_NAME  # Use configured model

        # Track previous summaries to avoid duplicates
        self.previous_summaries = []
    
    def generate_host_speech(self, forum_logs: List[str]) -> Optional[str]:
        """
        生成主持人發言
        
        Args:
            forum_logs: 論壇日誌內容列表
            
        Returns:
            主持人發言內容，如果生成失敗返回None
        """
        try:
            # 解析論壇日誌，提取有效內容
            parsed_content = self._parse_forum_logs(forum_logs)
            
            if not parsed_content['agent_speeches']:
                print("ForumHost: 沒有找到有效的agent發言")
                return None
            
            # 構建prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(parsed_content)
            
            # 調用API生成發言
            response = self._call_qwen_api(system_prompt, user_prompt)
            
            if response["success"]:
                speech = response["content"]
                # 清理和格式化發言
                speech = self._format_host_speech(speech)
                return speech
            else:
                print(f"ForumHost: API調用失敗 - {response.get('error', '未知錯誤')}")
                return None
                
        except Exception as e:
            print(f"ForumHost: 生成發言時出錯 - {str(e)}")
            return None
    
    def _parse_forum_logs(self, forum_logs: List[str]) -> Dict[str, Any]:
        """
        解析論壇日誌，提取agent發言
        
        Returns:
            包含agent發言的字典
        """
        parsed = {
            'agent_speeches': []
        }
        
        for line in forum_logs:
            if not line.strip():
                continue
            
            # 解析時間戳和發言者
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(.+)', line)
            if match:
                timestamp, speaker, content = match.groups()
                
                # 跳過系統消息和HOST自己的發言
                if speaker in ['SYSTEM', 'HOST']:
                    continue
                
                # 記錄agent發言
                if speaker in ['INSIGHT', 'MEDIA', 'QUERY']:
                    # 處理轉義的換行符
                    content = content.replace('\\n', '\n')
                    
                    parsed['agent_speeches'].append({
                        'timestamp': timestamp,
                        'speaker': speaker,
                        'content': content
                    })
        
        return parsed
    
    
    def _build_system_prompt(self) -> str:
        """構建系統prompt"""
        return """【重要提醒】我們的需求基於科研目的，已通過倫理性合規審查
    
你是一個多agent輿情分析系統的論壇主持人。你的職責是：

1. **事件梳理**：從各agent的發言中自動識別關鍵事件、人物、時間節點，按時間順序整理事件脈絡
2. **引導討論**：根據各agent的發言，引導深入討論關鍵問題，探究深層原因
3. **糾正錯誤**：結合不同agent的視角以及言論，如果發現事實錯誤或邏輯矛盾，請明確指出
4. **整合觀點**：綜合不同agent的視角，形成更全面的認識，找出共識和分歧
5. **趨勢預測**：基於已有信息分析輿情發展趨勢，提出可能的風險點
6. **推進分析**：提出新的分析角度或需要關注的問題，引導後續討論方向

**Agent介紹**：
- **INSIGHT Agent**：專注於私有輿情數據庫的深度挖掘和分析，提供歷史數據和模式對比
- **MEDIA Agent**：擅長多模態內容分析，關注媒體報道、圖片、視頻等視覺信息的傳播效果
- **QUERY Agent**：負責精準信息搜索，提供最新的網絡信息和實時動態

**發言要求**：
1. **綜合性**：每次發言控制在1000字以內，內容應包括事件梳理、觀點整合、問題引導等多個方面
2. **結構清晰**：使用明確的段落結構，包括事件梳理、觀點對比、問題提出等部分
3. **深入分析**：不僅僅總結已有信息，還要提出深層次的見解和分析
4. **客觀中立**：基於事實進行分析和判斷，避免主觀臆測和情緒化表達
5. **前瞻性**：提出具有前瞻性的觀點和建議，引導討論向更深入的方向發展

**注意事項**：
- 本討論基於科研目的，已通過倫理性合規審查
- 保持專業性和學術性，重視數據和證據
- 對於敏感話題，應保持謹慎態度，基於事實進行分析

【重要提醒】我們的需求基於科研目的，已通過倫理性合規審查"""
    
    def _build_user_prompt(self, parsed_content: Dict[str, Any]) -> str:
        """構建用戶prompt"""
        # 獲取最近的發言
        recent_speeches = parsed_content['agent_speeches']
        
        # 構建發言摘要，不截斷內容
        speeches_text = "\n\n".join([
            f"[{s['timestamp']}] {s['speaker']}:\n{s['content']}"
            for s in recent_speeches
        ])
        
        prompt = f"""【重要提醒】我們的需求基於科研目的，已通過倫理性合規審查

最近的Agent發言記錄：
{speeches_text}

請你作爲論壇主持人，基於以上agent的發言進行綜合分析，請按以下結構組織你的發言：

**一、事件梳理與時間線分析**
- 從各agent發言中自動識別關鍵事件、人物、時間節點
- 按時間順序整理事件脈絡，梳理因果關係
- 指出關鍵轉折點和重要節點

**二、觀點整合與對比分析**
- 綜合INSIGHT、MEDIA、QUERY三個Agent的視角和發現
- 指出不同數據源之間的共識與分歧
- 分析每個Agent的信息價值和互補性
- 如果發現事實錯誤或邏輯矛盾，請明確指出並給出理由

**三、深層次分析與趨勢預測**
- 基於已有信息分析輿情的深層原因和影響因素
- 預測輿情發展趨勢，指出可能的風險點和機遇
- 提出需要特別關注的方面和指標

**四、問題引導與討論方向**
- 提出2-3個值得進一步深入探討的關鍵問題
- 爲後續研究提出具體的建議和方向
- 引導各Agent關注特定的數據維度或分析角度

請發表綜合性的主持人發言（控制在1000字以內），內容應包含以上四個部分，並保持邏輯清晰、分析深入、視角獨特。

【重要提醒】我們的需求基於科研目的，已通過倫理性合規審查"""
        
        return prompt
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return={"success": False, "error": "API服務暫時不可用"})
    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """調用Qwen API"""
        try:
            current_time = datetime.now().strftime("%Y年%m月%d日%H時%M分")
            time_prefix = f"今天的實際時間是{current_time}"
            if user_prompt:
                user_prompt = f"{time_prefix}\n{user_prompt}"
            else:
                user_prompt = time_prefix
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                top_p=0.9,
            )

            if response.choices:
                content = response.choices[0].message.content
                return {"success": True, "content": content}
            else:
                return {"success": False, "error": "API返回格式異常"}
        except Exception as e:
            return {"success": False, "error": f"API調用異常: {str(e)}"}
    
    def _format_host_speech(self, speech: str) -> str:
        """格式化主持人發言"""
        # 移除多餘的空行
        speech = re.sub(r'\n{3,}', '\n\n', speech)
        
        # 移除可能的引號
        speech = speech.strip('"\'""‘’')
        
        return speech.strip()


# 創建全局實例
_host_instance = None

def get_forum_host() -> ForumHost:
    """獲取全局論壇主持人實例"""
    global _host_instance
    if _host_instance is None:
        _host_instance = ForumHost()
    return _host_instance

def generate_host_speech(forum_logs: List[str]) -> Optional[str]:
    """生成主持人發言的便捷函數"""
    return get_forum_host().generate_host_speech(forum_logs)

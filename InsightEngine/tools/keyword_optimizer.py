"""
關鍵詞優化中間件
使用Qwen AI將Agent生成的搜索詞優化爲更適合輿情數據庫查詢的關鍵詞
"""

from openai import OpenAI
import json
import sys
import os
from typing import List, Dict, Any
from dataclasses import dataclass

# 添加項目根目錄到Python路徑以導入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings
from loguru import logger

# 添加utils目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG

@dataclass
class KeywordOptimizationResponse:
    """關鍵詞優化響應"""
    original_query: str
    optimized_keywords: List[str]
    reasoning: str
    success: bool
    error_message: str = ""

class KeywordOptimizer:
    """
    關鍵詞優化器
    使用硅基流動的Qwen3模型將Agent生成的搜索詞優化爲更貼近真實輿情的關鍵詞
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        """
        初始化關鍵詞優化器
        
        Args:
            api_key: 硅基流動API密鑰，如果不提供則從配置文件讀取
            base_url: 接口基礎地址，默認使用配置文件提供的SiliconFlow地址
        """
        self.api_key = api_key or settings.KEYWORD_OPTIMIZER_API_KEY

        if not self.api_key:
            raise ValueError("未找到硅基流動API密鑰，請在config.py中設置KEYWORD_OPTIMIZER_API_KEY")

        self.base_url = base_url or settings.KEYWORD_OPTIMIZER_BASE_URL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model = model_name or settings.KEYWORD_OPTIMIZER_MODEL_NAME
    
    def optimize_keywords(self, original_query: str, context: str = "") -> KeywordOptimizationResponse:
        """
        優化搜索關鍵詞
        
        Args:
            original_query: Agent生成的原始搜索查詢
            context: 額外的上下文信息（如段落標題、內容描述等）
            
        Returns:
            KeywordOptimizationResponse: 優化後的關鍵詞列表
        """
        logger.info(f"🔍 關鍵詞優化中間件: 處理查詢 '{original_query}'")
        
        try:
            # 構建優化prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(original_query, context)
            
            # 調用Qwen API
            response = self._call_qwen_api(system_prompt, user_prompt)
            
            if response["success"]:
                # 解析響應
                content = response["content"]
                try:
                    # 嘗試解析JSON格式的響應
                    if content.strip().startswith('{'):
                        parsed = json.loads(content)
                        keywords = parsed.get("keywords", [])
                        reasoning = parsed.get("reasoning", "")
                    else:
                        # 如果不是JSON格式，嘗試從文本中提取關鍵詞
                        keywords = self._extract_keywords_from_text(content)
                        reasoning = content
                    
                    # 驗證關鍵詞質量
                    validated_keywords = self._validate_keywords(keywords)
                    
                    logger.info(
                        f"✅ 優化成功: {len(validated_keywords)}個關鍵詞" +
                        ("" if not validated_keywords else "\n" +
                         "\n".join([f"   {i}. '{k}'" for i, k in enumerate(validated_keywords, 1)]))
                    )
                        
                    
                    
                    return KeywordOptimizationResponse(
                        original_query=original_query,
                        optimized_keywords=validated_keywords,
                        reasoning=reasoning,
                        success=True
                    )
                
                except Exception as e:
                    logger.exception(f"⚠️ 解析響應失敗，使用備用方案: {str(e)}")
                    # 備用方案：從原始查詢中提取關鍵詞
                    fallback_keywords = self._fallback_keyword_extraction(original_query)
                    return KeywordOptimizationResponse(
                        original_query=original_query,
                        optimized_keywords=fallback_keywords,
                        reasoning="API響應解析失敗，使用備用關鍵詞提取",
                        success=True
                    )
            else:
                logger.error(f"❌ API調用失敗: {response['error']}")
                # 使用備用方案
                fallback_keywords = self._fallback_keyword_extraction(original_query)
                return KeywordOptimizationResponse(
                    original_query=original_query,
                    optimized_keywords=fallback_keywords,
                    reasoning="API調用失敗，使用備用關鍵詞提取",
                    success=True,
                    error_message=response['error']
                )
                
        except Exception as e:
            logger.error(f"❌ 關鍵詞優化失敗: {str(e)}")
            # 最終備用方案
            fallback_keywords = self._fallback_keyword_extraction(original_query)
            return KeywordOptimizationResponse(
                original_query=original_query,
                optimized_keywords=fallback_keywords,
                reasoning="系統錯誤，使用備用關鍵詞提取",
                success=False,
                error_message=str(e)
            )
    
    def _build_system_prompt(self) -> str:
        """構建系統prompt"""
        return """你是一位專業的輿情數據挖掘專家。你的任務是將用戶提供的搜索查詢優化爲更適合在社交媒體輿情數據庫中查找的關鍵詞。

**核心原則**：
1. **貼近網民語言**：使用普通網友在社交媒體上會使用的詞彙
2. **避免專業術語**：不使用"輿情"、"傳播"、"傾向"、"展望"等官方詞彙
3. **簡潔具體**：每個關鍵詞要非常簡潔明瞭，便於數據庫匹配
4. **情感豐富**：包含網民常用的情感表達詞彙
5. **數量控制**：最少提供10個關鍵詞，最多提供20個關鍵詞
6. **避免重複**：不要脫離初始查詢的主題

**重要提醒**：每個關鍵詞都必須是一個不可分割的獨立詞條，嚴禁在詞條內部包含空格。例如，應使用 "雷軍班爭議" 而不是錯誤的 "雷軍班 爭議"。

**輸出格式**：
請以JSON格式返回結果：
{
    "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
    "reasoning": "選擇這些關鍵詞的理由"
}

**示例**：
輸入："武漢大學輿情管理 未來展望 發展趨勢"
輸出：
{
    "keywords": ["武大", "武漢大學", "學校管理", "大學", "教育"],
    "reasoning": "選擇'武大'和'武漢大學'作爲核心詞彙，這是網民最常使用的稱呼；'學校管理'比'輿情管理'更貼近日常表達；避免使用'未來展望'、'發展趨勢'等網民很少使用的專業術語"
}"""

    def _build_user_prompt(self, original_query: str, context: str) -> str:
        """構建用戶prompt"""
        prompt = f"請將以下搜索查詢優化爲適合輿情數據庫查詢的關鍵詞：\n\n原始查詢：{original_query}"
        
        if context:
            prompt += f"\n\n上下文信息：{context}"
        
        prompt += "\n\n請記住：要使用網民在社交媒體上真實使用的詞彙，避免官方術語和專業詞彙。"
        
        return prompt
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return={"success": False, "error": "關鍵詞優化服務暫時不可用"})
    def _call_qwen_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """調用Qwen API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )

            if response.choices:
                content = response.choices[0].message.content
                return {"success": True, "content": content}
            else:
                return {"success": False, "error": "API返回格式異常"}
        except Exception as e:
            return {"success": False, "error": f"API調用異常: {str(e)}"}
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """從文本中提取關鍵詞（當JSON解析失敗時使用）"""
        # 簡單的關鍵詞提取邏輯
        lines = text.split('\n')
        keywords = []
        
        for line in lines:
            line = line.strip()
            # 查找可能的關鍵詞
            if '：' in line or ':' in line:
                parts = line.split('：') if '：' in line else line.split(':')
                if len(parts) > 1:
                    potential_keywords = parts[1].strip()
                    # 嘗試分割關鍵詞
                    if '、' in potential_keywords:
                        keywords.extend([k.strip() for k in potential_keywords.split('、')])
                    elif ',' in potential_keywords:
                        keywords.extend([k.strip() for k in potential_keywords.split(',')])
                    else:
                        keywords.append(potential_keywords)
        
        # 如果沒有找到，嘗試其他方法
        if not keywords:
            # 查找引號中的內容
            import re
            quoted_content = re.findall(r'["""\'](.*?)["""\']', text)
            keywords.extend(quoted_content)
        
        # 清理和驗證關鍵詞
        cleaned_keywords = []
        for keyword in keywords[:20]:  # 最多20個
            keyword = keyword.strip().strip('"\'""''')
            if keyword and len(keyword) <= 20:  # 合理長度
                cleaned_keywords.append(keyword)
        
        return cleaned_keywords[:20]
    
    def _validate_keywords(self, keywords: List[str]) -> List[str]:
        """驗證和清理關鍵詞"""
        validated = []
        
        # 不良關鍵詞（過於專業或官方）
        bad_keywords = {
            '態度分析', '公衆反應', '情緒傾向',
            '未來展望', '發展趨勢', '戰略規劃', '政策導向', '管理機制'
        }
        
        for keyword in keywords:
            if isinstance(keyword, str):
                keyword = keyword.strip().strip('"\'""''')
                
                # 基本驗證
                if (keyword and 
                    len(keyword) <= 20 and 
                    len(keyword) >= 1 and
                    not any(bad_word in keyword for bad_word in bad_keywords)):
                    validated.append(keyword)
        
        return validated[:20]  # 最多返回20個關鍵詞
    
    def _fallback_keyword_extraction(self, original_query: str) -> List[str]:
        """備用關鍵詞提取方案"""
        # 簡單的關鍵詞提取邏輯
        # 移除常見的無用詞彙
        stop_words = {'、'}
        
        # 分割查詢
        import re
        # 按空格、標點分割
        tokens = re.split(r'[\s，。！？；：、]+', original_query)
        
        keywords = []
        for token in tokens:
            token = token.strip()
            if token and token not in stop_words and len(token) >= 2:
                keywords.append(token)
        
        # 如果沒有有效關鍵詞，使用原始查詢的第一個詞
        if not keywords:
            first_word = original_query.split()[0] if original_query.split() else original_query
            keywords = [first_word] if first_word else ["熱門"]
        
        return keywords[:20]

# 全局實例
keyword_optimizer = KeywordOptimizer()

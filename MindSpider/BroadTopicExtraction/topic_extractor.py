#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction模塊 - 話題提取器
基於DeepSeek直接提取關鍵詞和生成新聞總結
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
    from config import settings
except ImportError:
    raise ImportError("無法導入settings.py配置文件")

class TopicExtractor:
    """話題提取器"""

    def __init__(self):
        """初始化話題提取器"""
        self.client = OpenAI(
            api_key=settings.MINDSPIDER_API_KEY,
            base_url=settings.MINDSPIDER_BASE_URL
        )
        self.model = settings.MINDSPIDER_MODEL_NAME
    
    def extract_keywords_and_summary(self, news_list: List[Dict], max_keywords: int = 100) -> Tuple[List[str], str]:
        """
        從新聞列表中提取關鍵詞和生成總結
        
        Args:
            news_list: 新聞列表
            max_keywords: 最大關鍵詞數量
            
        Returns:
            (關鍵詞列表, 新聞分析總結)
        """
        if not news_list:
            return [], "今日暫無熱點新聞"
        
        # 構建新聞摘要文本
        news_text = self._build_news_summary(news_list)
        
        # 構建提示詞
        prompt = self._build_analysis_prompt(news_text, max_keywords)
        
        try:
            # 調用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一個專業的新聞分析師，擅長從熱點新聞中提取關鍵詞和撰寫分析總結。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # 解析返回結果
            result_text = response.choices[0].message.content
            keywords, summary = self._parse_analysis_result(result_text)
            
            print(f"成功提取 {len(keywords)} 個關鍵詞並生成新聞總結")
            return keywords[:max_keywords], summary
            
        except Exception as e:
            print(f"話題提取失敗: {e}")
            # 返回簡單的fallback結果
            fallback_keywords = self._extract_simple_keywords(news_list)
            fallback_summary = f"今日共收集到 {len(news_list)} 條熱點新聞，涵蓋多個平臺的熱門話題。"
            return fallback_keywords[:max_keywords], fallback_summary
    
    def _build_news_summary(self, news_list: List[Dict]) -> str:
        """構建新聞摘要文本"""
        news_items = []
        
        for i, news in enumerate(news_list, 1):
            title = news.get('title', '無標題')
            source = news.get('source_platform', news.get('source', '未知'))
            
            # 清理標題中的特殊字符
            title = re.sub(r'[#@]', '', title).strip()
            
            news_items.append(f"{i}. 【{source}】{title}")
        
        return "\n".join(news_items)
    
    def _build_analysis_prompt(self, news_text: str, max_keywords: int) -> str:
        """構建分析提示詞"""
        news_count = len(news_text.split('\n'))
        
        prompt = f"""
請分析以下{news_count}條今日熱點新聞，完成兩個任務：

新聞列表：
{news_text}

任務1：提取關鍵詞（最多{max_keywords}個）
- 提取能代表今日熱點話題的關鍵詞
- 關鍵詞應該適合用於社交媒體平臺搜索
- 優先選擇熱度高、討論量大的話題
- 避免過於寬泛或過於具體的詞彙

任務2：撰寫新聞分析總結（150-300字）
- 簡要概括今日熱點新聞的主要內容
- 指出當前社會關注的重點話題方向
- 分析這些熱點反映的社會現象或趨勢
- 語言簡潔明瞭，客觀中性

請嚴格按照以下JSON格式輸出：
```json
{{
  "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"],
  "summary": "今日新聞分析總結內容..."
}}
```

請直接輸出JSON格式的結果，不要包含其他文字說明。
"""
        return prompt
    
    def _parse_analysis_result(self, result_text: str) -> Tuple[List[str], str]:
        """解析分析結果"""
        try:
            # 嘗試提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # 如果沒有代碼塊，嘗試直接解析
                json_text = result_text.strip()
            
            # 解析JSON
            data = json.loads(json_text)
            
            keywords = data.get('keywords', [])
            summary = data.get('summary', '')
            
            # 驗證和清理關鍵詞
            clean_keywords = []
            for keyword in keywords:
                keyword = str(keyword).strip()
                if keyword and len(keyword) > 1 and keyword not in clean_keywords:
                    clean_keywords.append(keyword)
            
            # 驗證總結
            if not summary or len(summary.strip()) < 10:
                summary = "今日熱點新聞涵蓋多個領域，反映了當前社會的多元化關注點。"
            
            return clean_keywords, summary.strip()
            
        except json.JSONDecodeError as e:
            print(f"解析JSON失敗: {e}")
            print(f"原始返回: {result_text}")
            
            # 嘗試手動解析
            return self._manual_parse_result(result_text)
        
        except Exception as e:
            print(f"處理分析結果失敗: {e}")
            return [], "分析結果處理失敗，請稍後重試。"
    
    def _manual_parse_result(self, text: str) -> Tuple[List[str], str]:
        """手動解析結果（當JSON解析失敗時的後備方案）"""
        print("嘗試手動解析結果...")
        
        keywords = []
        summary = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尋找關鍵詞
            if '關鍵詞' in line or 'keywords' in line.lower():
                # 提取關鍵詞
                keyword_match = re.findall(r'[""](.*?)["""]', line)
                if keyword_match:
                    keywords.extend(keyword_match)
                else:
                    # 嘗試其他分隔符
                    parts = re.split(r'[,，、]', line)
                    for part in parts:
                        clean_part = re.sub(r'[關鍵詞：:keywords\[\]"]', '', part).strip()
                        if clean_part and len(clean_part) > 1:
                            keywords.append(clean_part)
            
            # 尋找總結
            elif '總結' in line or '分析' in line or 'summary' in line.lower():
                if '：' in line or ':' in line:
                    summary = line.split('：')[-1].split(':')[-1].strip()
            
            # 如果這一行看起來像總結內容
            elif len(line) > 50 and ('今日' in line or '熱點' in line or '新聞' in line):
                if not summary:
                    summary = line
        
        # 清理關鍵詞
        clean_keywords = []
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword and len(keyword) > 1 and keyword not in clean_keywords:
                clean_keywords.append(keyword)
        
        # 如果沒有找到總結，生成一個簡單的
        if not summary:
            summary = "今日熱點新聞內容豐富，涵蓋了社會各個層面的關注點。"
        
        return clean_keywords[:max_keywords], summary
    
    def _extract_simple_keywords(self, news_list: List[Dict]) -> List[str]:
        """簡單關鍵詞提取（fallback方案）"""
        keywords = []
        
        for news in news_list:
            title = news.get('title', '')
            
            # 簡單的關鍵詞提取
            # 移除常見的無意義詞彙
            title_clean = re.sub(r'[#@【】\[\]()（）]', ' ', title)
            words = title_clean.split()
            
            for word in words:
                word = word.strip()
                if (len(word) > 1 and 
                    word not in ['的', '了', '在', '和', '與', '或', '但', '是', '有', '被', '將', '已', '正在'] and
                    word not in keywords):
                    keywords.append(word)
        
        return keywords[:10]
    
    def get_search_keywords(self, keywords: List[str], limit: int = 10) -> List[str]:
        """
        獲取用於搜索的關鍵詞
        
        Args:
            keywords: 關鍵詞列表
            limit: 限制數量
            
        Returns:
            適合搜索的關鍵詞列表
        """
        # 過濾和優化關鍵詞
        search_keywords = []
        
        for keyword in keywords:
            keyword = str(keyword).strip()
            
            # 過濾條件
            if (len(keyword) > 1 and 
                len(keyword) < 20 and  # 不能太長
                keyword not in search_keywords and
                not keyword.isdigit() and  # 不是純數字
                not re.match(r'^[a-zA-Z]+$', keyword)):  # 不是純英文（除非是專有名詞）
                
                search_keywords.append(keyword)
        
        return search_keywords[:limit]

if __name__ == "__main__":
    # 測試話題提取器
    extractor = TopicExtractor()
    
    # 模擬新聞數據
    test_news = [
        {"title": "AI技術發展迅速", "source_platform": "科技新聞"},
        {"title": "股市行情分析", "source_platform": "財經新聞"},
        {"title": "明星最新動態", "source_platform": "娛樂新聞"}
    ]
    
    keywords, summary = extractor.extract_keywords_and_summary(test_news)
    
    print(f"提取的關鍵詞: {keywords}")
    print(f"新聞總結: {summary}")
    
    search_keywords = extractor.get_search_keywords(keywords)
    print(f"搜索關鍵詞: {search_keywords}")

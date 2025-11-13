"""
文本處理工具函數
用於清理LLM輸出、解析JSON等
"""

import re
import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError


def clean_json_tags(text: str) -> str:
    """
    清理文本中的JSON標籤
    
    Args:
        text: 原始文本
        
    Returns:
        清理後的文本
    """
    # 移除```json 和 ```標籤
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def clean_markdown_tags(text: str) -> str:
    """
    清理文本中的Markdown標籤
    
    Args:
        text: 原始文本
        
    Returns:
        清理後的文本
    """
    # 移除```markdown 和 ```標籤
    text = re.sub(r'```markdown\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = re.sub(r'```', '', text)
    
    return text.strip()


def remove_reasoning_from_output(text: str) -> str:
    """
    移除輸出中的推理過程文本
    
    Args:
        text: 原始文本
        
    Returns:
        清理後的文本
    """
    # 查找JSON開始位置
    json_start = -1
    
    # 嘗試找到第一個 { 或 [
    for i, char in enumerate(text):
        if char in '{[':
            json_start = i
            break
    
    if json_start != -1:
        # 從JSON開始位置截取
        return text[json_start:].strip()
    
    # 如果沒有找到JSON標記，嘗試其他方法
    # 移除常見的推理標識
    patterns = [
        r'(?:reasoning|推理|思考|分析)[:：]\s*.*?(?=\{|\[)',  # 移除推理部分
        r'(?:explanation|解釋|說明)[:：]\s*.*?(?=\{|\[)',   # 移除解釋部分
        r'^.*?(?=\{|\[)',  # 移除JSON前的所有文本
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_clean_response(text: str) -> Dict[str, Any]:
    """
    提取並清理響應中的JSON內容
    
    Args:
        text: 原始響應文本
        
    Returns:
        解析後的JSON字典
    """
    # 清理文本
    cleaned_text = clean_json_tags(text)
    cleaned_text = remove_reasoning_from_output(cleaned_text)
    
    # 嘗試直接解析
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # 嘗試修復不完整的JSON
    fixed_text = fix_incomplete_json(cleaned_text)
    if fixed_text:
        try:
            return json.loads(fixed_text)
        except JSONDecodeError:
            pass
    
    # 嘗試查找JSON對象
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 嘗試查找JSON數組
    array_pattern = r'\[.*\]'
    match = re.search(array_pattern, cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except JSONDecodeError:
            pass
    
    # 如果所有方法都失敗，返回錯誤信息
    print(f"無法解析JSON響應: {cleaned_text[:200]}...")
    return {"error": "JSON解析失敗", "raw_text": cleaned_text}


def fix_incomplete_json(text: str) -> str:
    """
    修復不完整的JSON響應
    
    Args:
        text: 原始文本
        
    Returns:
        修復後的JSON文本，如果無法修復則返回空字符串
    """
    # 移除多餘的逗號和空白
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # 檢查是否已經是有效的JSON
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        pass
    
    # 檢查是否缺少開頭的數組符號
    if text.strip().startswith('{') and not text.strip().startswith('['):
        # 如果以對象開始，嘗試包裝成數組
        if text.count('{') > 1:
            # 多個對象，包裝成數組
            text = '[' + text + ']'
        else:
            # 單個對象，包裝成數組
            text = '[' + text + ']'
    
    # 檢查是否缺少結尾的數組符號
    if text.strip().endswith('}') and not text.strip().endswith(']'):
        # 如果以對象結束，嘗試包裝成數組
        if text.count('}') > 1:
            # 多個對象，包裝成數組
            text = '[' + text + ']'
        else:
            # 單個對象，包裝成數組
            text = '[' + text + ']'
    
    # 檢查括號是否匹配
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # 修復不匹配的括號
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)
    
    # 驗證修復後的JSON是否有效
    try:
        json.loads(text)
        return text
    except JSONDecodeError:
        # 如果仍然無效，嘗試更激進的修復
        return fix_aggressive_json(text)


def fix_aggressive_json(text: str) -> str:
    """
    更激進的JSON修復方法
    
    Args:
        text: 原始文本
        
    Returns:
        修復後的JSON文本
    """
    # 查找所有可能的JSON對象
    objects = re.findall(r'\{[^{}]*\}', text)
    
    if len(objects) >= 2:
        # 如果有多個對象，包裝成數組
        return '[' + ','.join(objects) + ']'
    elif len(objects) == 1:
        # 如果只有一個對象，包裝成數組
        return '[' + objects[0] + ']'
    else:
        # 如果沒有找到對象，返回空數組
        return '[]'


def update_state_with_search_results(search_results: List[Dict[str, Any]], 
                                   paragraph_index: int, state: Any) -> Any:
    """
    將搜索結果更新到狀態中
    
    Args:
        search_results: 搜索結果列表
        paragraph_index: 段落索引
        state: 狀態對象
        
    Returns:
        更新後的狀態對象
    """
    if 0 <= paragraph_index < len(state.paragraphs):
        # 獲取最後一次搜索的查詢（假設是當前查詢）
        current_query = ""
        if search_results:
            # 從搜索結果推斷查詢（這裏需要改進以獲取實際查詢）
            current_query = "搜索查詢"
        
        # 添加搜索結果到狀態
        state.paragraphs[paragraph_index].research.add_search_results(
            current_query, search_results
        )
    
    return state


def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    驗證JSON數據是否包含必需字段
    
    Args:
        data: 要驗證的數據
        required_fields: 必需字段列表
        
    Returns:
        驗證是否通過
    """
    return all(field in data for field in required_fields)


def truncate_content(content: str, max_length: int = 20000) -> str:
    """
    截斷內容到指定長度
    
    Args:
        content: 原始內容
        max_length: 最大長度
        
    Returns:
        截斷後的內容
    """
    if len(content) <= max_length:
        return content
    
    # 嘗試在單詞邊界截斷
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # 如果最後一個空格位置合理
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."


def format_search_results_for_prompt(search_results: List[Dict[str, Any]], 
                                   max_length: int = 20000) -> List[str]:
    """
    格式化搜索結果用於提示詞
    
    Args:
        search_results: 搜索結果列表
        max_length: 每個結果的最大長度
        
    Returns:
        格式化後的內容列表
    """
    formatted_results = []
    
    for result in search_results:
        content = result.get('content', '')
        if content:
            truncated_content = truncate_content(content, max_length)
            formatted_results.append(truncated_content)
    
    return formatted_results

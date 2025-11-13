"""
GitHub Issues 工具模塊

提供創建 GitHub Issues URL 和顯示帶鏈接的錯誤信息的功能
數據模型定義位置：
- 無數據模型
"""

from datetime import datetime
from urllib.parse import quote

# GitHub 倉庫信息
GITHUB_REPO = "666ghj/BettaFish"
GITHUB_ISSUES_URL = f"https://github.com/{GITHUB_REPO}/issues/new"


def create_issue_url(title: str, body: str = "") -> str:
    """
    創建 GitHub Issues URL，預填充標題和內容
    
    Args:
        title: Issue 標題
        body: Issue 內容（可選）
    
    Returns:
        完整的 GitHub Issues URL
    """
    encoded_title = quote(title)
    encoded_body = quote(body) if body else ""
    
    if encoded_body:
        return f"{GITHUB_ISSUES_URL}?title={encoded_title}&body={encoded_body}"
    else:
        return f"{GITHUB_ISSUES_URL}?title={encoded_title}"


def error_with_issue_link(
    error_message: str,
    error_details: str = "",
    app_name: str = "Streamlit App"
) -> str:
    """
    生成帶 GitHub Issues 鏈接的錯誤信息字符串
    
    僅在通用異常處理中使用，不用於用戶配置錯誤
    
    Args:
        error_message: 錯誤消息
        error_details: 錯誤詳情（可選，用於填充到 Issue body）
        app_name: 應用名稱，用於標識錯誤來源
    
    Returns:
        包含錯誤信息和 GitHub Issues 鏈接的 Markdown 格式字符串
    """
    issue_title = f"[{app_name}] {error_message[:50]}"
    issue_body = f"## 錯誤信息\n\n{error_message}\n\n"
    
    if error_details:
        issue_body += f"## 錯誤詳情\n\n```\n{error_details}\n```\n\n"
    
    issue_body += f"## 環境信息\n\n- 應用: {app_name}\n- 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    issue_url = create_issue_url(issue_title, issue_body)
    
    # 使用 markdown 格式添加超鏈接
    error_display = f"{error_message}\n\n[📝 提交錯誤報告]({issue_url})"
    
    if error_details:
        error_display = f"{error_message}\n\n```\n{error_details}\n```\n\n[📝 提交錯誤報告]({issue_url})"
    
    return error_display


__all__ = [
    "create_issue_url",
    "error_with_issue_link",
    "GITHUB_REPO",
    "GITHUB_ISSUES_URL",
]


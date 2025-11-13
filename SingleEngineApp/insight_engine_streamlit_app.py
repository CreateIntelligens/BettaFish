"""
Streamlit Web界面
爲Insight Agent提供友好的Web界面
"""

import os
import sys
import streamlit as st
from datetime import datetime
import json
import locale
from loguru import logger

# 設置UTF-8編碼環境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 設置系統編碼
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass

# 添加src目錄到Python路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from InsightEngine import DeepSearchAgent, Settings
from config import settings
from utils.github_issues import error_with_issue_link


def main():
    """主函數"""
    st.set_page_config(
        page_title="Insight Agent",
        page_icon="",
        layout="wide"
    )

    st.title("Insight Agent")
    st.markdown("私有輿情數據庫深度分析AI代理")
    st.markdown("24小時全自動從包括微博、知乎、github、酷安等 13個 社媒平臺、技術論壇廣泛的爬取輿情數據")

    # 檢查URL參數
    try:
        # 嘗試使用新版本的query_params
        query_params = st.query_params
        auto_query = query_params.get('query', '')
        auto_search = query_params.get('auto_search', 'false').lower() == 'true'
    except AttributeError:
        # 兼容舊版本
        query_params = st.experimental_get_query_params()
        auto_query = query_params.get('query', [''])[0]
        auto_search = query_params.get('auto_search', ['false'])[0].lower() == 'true'

    # ----- 配置被硬編碼 -----
    # 強制使用 Kimi
    model_name = settings.INSIGHT_ENGINE_MODEL_NAME or "kimi-k2-0711-preview"
    # 默認高級配置
    max_reflections = 2
    max_content_length = 500000  # Kimi支持長文本

    # 簡化的研究查詢展示區域

    # 如果有自動查詢，使用它作爲默認值，否則顯示佔位符
    display_query = auto_query if auto_query else "等待從主頁面接收分析內容..."

    # 只讀的查詢展示區域
    st.text_area(
        "當前查詢",
        value=display_query,
        height=100,
        disabled=True,
        help="查詢內容由主頁面的搜索框控制",
        label_visibility="hidden"
    )

    # 自動搜索邏輯
    start_research = False
    query = auto_query

    if auto_search and auto_query and 'auto_search_executed' not in st.session_state:
        st.session_state.auto_search_executed = True
        start_research = True
    elif auto_query and not auto_search:
        st.warning("等待搜索啓動信號...")

    # 驗證配置
    if start_research:
        if not query.strip():
            st.error("請輸入研究查詢")
            logger.error("請輸入研究查詢")
            return

        # 檢查配置中的LLM密鑰
        if not settings.INSIGHT_ENGINE_API_KEY:
            st.error("請在您的環境變量中設置INSIGHT_ENGINE_API_KEY")
            logger.error("請在您的環境變量中設置INSIGHT_ENGINE_API_KEY")
            return

        # 自動使用配置文件中的API密鑰和數據庫配置
        db_host = settings.DB_HOST
        db_user = settings.DB_USER
        db_password = settings.DB_PASSWORD
        db_name = settings.DB_NAME
        db_port = settings.DB_PORT
        db_charset = settings.DB_CHARSET

        # 創建Settings配置（字段必須用大寫，以適配Settings類）
        config = Settings(
            INSIGHT_ENGINE_API_KEY=settings.INSIGHT_ENGINE_API_KEY,
            INSIGHT_ENGINE_BASE_URL=settings.INSIGHT_ENGINE_BASE_URL,
            INSIGHT_ENGINE_MODEL_NAME=model_name,
            DB_HOST=db_host,
            DB_USER=db_user,
            DB_PASSWORD=db_password,
            DB_NAME=db_name,
            DB_PORT=db_port,
            DB_CHARSET=db_charset,
            DB_DIALECT=settings.DB_DIALECT,
            MAX_REFLECTIONS=max_reflections,
            MAX_CONTENT_LENGTH=max_content_length,
            OUTPUT_DIR="insight_engine_streamlit_reports"
        )

        # 執行研究
        execute_research(query, config)


def execute_research(query: str, config: Settings):
    """執行研究"""
    try:
        # 創建進度條
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 初始化Agent
        status_text.text("正在初始化Agent...")
        agent = DeepSearchAgent(config)
        st.session_state.agent = agent

        progress_bar.progress(10)

        # 生成報告結構
        status_text.text("正在生成報告結構...")
        agent._generate_report_structure(query)
        progress_bar.progress(20)

        # 處理段落
        total_paragraphs = len(agent.state.paragraphs)
        for i in range(total_paragraphs):
            status_text.text(f"正在處理段落 {i + 1}/{total_paragraphs}: {agent.state.paragraphs[i].title}")

            # 初始搜索和總結
            agent._initial_search_and_summary(i)
            progress_value = 20 + (i + 0.5) / total_paragraphs * 60
            progress_bar.progress(int(progress_value))

            # 反思循環
            agent._reflection_loop(i)
            agent.state.paragraphs[i].research.mark_completed()

            progress_value = 20 + (i + 1) / total_paragraphs * 60
            progress_bar.progress(int(progress_value))

        # 生成最終報告
        status_text.text("正在生成最終報告...")
        final_report = agent._generate_final_report()
        progress_bar.progress(90)

        # 保存報告
        status_text.text("正在保存報告...")
        agent._save_report(final_report)
        progress_bar.progress(100)

        status_text.text("研究完成！")

        # 顯示結果
        display_results(agent, final_report)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_display = error_with_issue_link(
            f"研究過程中發生錯誤: {str(e)}",
            error_traceback,
            app_name="Insight Engine Streamlit App"
        )
        st.error(error_display)
        logger.exception(f"研究過程中發生錯誤: {str(e)}")


def display_results(agent: DeepSearchAgent, final_report: str):
    """顯示研究結果"""
    st.header("工作結束")

    # 結果標籤頁（已移除下載選項）
    tab1, tab2 = st.tabs(["研究小結", "引用信息"])

    with tab1:
        st.markdown(final_report)

    with tab2:
        # 段落詳情
        st.subheader("段落詳情")
        for i, paragraph in enumerate(agent.state.paragraphs):
            with st.expander(f"段落 {i + 1}: {paragraph.title}"):
                st.write("**預期內容:**", paragraph.content)
                st.write("**最終內容:**", paragraph.research.latest_summary[:300] + "..."
                if len(paragraph.research.latest_summary) > 300
                else paragraph.research.latest_summary)
                st.write("**搜索次數:**", paragraph.research.get_search_count())
                st.write("**反思次數:**", paragraph.research.reflection_iteration)

        # 搜索歷史
        st.subheader("搜索歷史")
        all_searches = []
        for paragraph in agent.state.paragraphs:
            all_searches.extend(paragraph.research.search_history)

        if all_searches:
            for i, search in enumerate(all_searches):
                with st.expander(f"搜索 {i + 1}: {search.query}"):
                    st.write("**URL:**", search.url)
                    st.write("**標題:**", search.title)
                    st.write("**內容預覽:**",
                             search.content[:200] + "..." if len(search.content) > 200 else search.content)
                    if search.score:
                        st.write("**相關度評分:**", search.score)


if __name__ == "__main__":
    main()

"""
Deep Search Agent主類
整合所有模塊，實現完整的深度搜索流程
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from loguru import logger

from .llms import LLMClient
from .nodes import (
    ReportStructureNode,
    FirstSearchNode, 
    ReflectionNode,
    FirstSummaryNode,
    ReflectionSummaryNode,
    ReportFormattingNode
)
from .state import State
from .tools import MediaCrawlerDB, DBResponse, keyword_optimizer, multilingual_sentiment_analyzer
from .utils.config import settings, Settings
from .utils import format_search_results_for_prompt


class DeepSearchAgent:
    """Deep Search Agent主類"""
    
    def __init__(self, config: Optional[Settings] = None):
        """
        初始化Deep Search Agent
        
        Args:
            config: 可選配置對象（不填則用全局settings）
        """
        self.config = config or settings
        
        # 初始化LLM客戶端
        self.llm_client = self._initialize_llm()
        
        
        # 初始化搜索工具集
        self.search_agency = MediaCrawlerDB()
        
        # 初始化情感分析器
        self.sentiment_analyzer = multilingual_sentiment_analyzer
        
        # 初始化節點
        self._initialize_nodes()
        
        # 狀態
        self.state = State()
        
        # 確保輸出目錄存在
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"Insight Agent已初始化")
        logger.info(f"使用LLM: {self.llm_client.get_model_info()}")
        logger.info(f"搜索工具集: MediaCrawlerDB (支持5種本地數據庫查詢工具)")
        logger.info(f"情感分析: WeiboMultilingualSentiment (支持22種語言的情感分析)")
    
    def _initialize_llm(self) -> LLMClient:
        """初始化LLM客戶端"""
        return LLMClient(
            api_key=self.config.INSIGHT_ENGINE_API_KEY,
            model_name=self.config.INSIGHT_ENGINE_MODEL_NAME,
            base_url=self.config.INSIGHT_ENGINE_BASE_URL,
        )
    
    def _initialize_nodes(self):
        """初始化處理節點"""
        self.first_search_node = FirstSearchNode(self.llm_client)
        self.reflection_node = ReflectionNode(self.llm_client)
        self.first_summary_node = FirstSummaryNode(self.llm_client)
        self.reflection_summary_node = ReflectionSummaryNode(self.llm_client)
        self.report_formatting_node = ReportFormattingNode(self.llm_client)
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        驗證日期格式是否爲YYYY-MM-DD
        
        Args:
            date_str: 日期字符串
            
        Returns:
            是否爲有效格式
        """
        if not date_str:
            return False
        
        # 檢查格式
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False
        
        # 檢查日期是否有效
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> DBResponse:
        """
        執行指定的數據庫查詢工具（集成關鍵詞優化中間件和情感分析）
        
        Args:
            tool_name: 工具名稱，可選值：
                - "search_hot_content": 查找熱點內容
                - "search_topic_globally": 全局話題搜索
                - "search_topic_by_date": 按日期搜索話題
                - "get_comments_for_topic": 獲取話題評論
                - "search_topic_on_platform": 平臺定向搜索
                - "analyze_sentiment": 對查詢結果進行情感分析
            query: 搜索關鍵詞/話題
            **kwargs: 額外參數（如start_date, end_date, platform, limit, enable_sentiment等）
                     enable_sentiment: 是否自動對搜索結果進行情感分析（默認True）
            
        Returns:
            DBResponse對象（可能包含情感分析結果）
        """
        logger.info(f"  → 執行數據庫查詢工具: {tool_name}")
        
        # 對於熱點內容搜索，不需要關鍵詞優化（因爲不需要query參數）
        if tool_name == "search_hot_content":
            time_period = kwargs.get("time_period", "week")
            limit = kwargs.get("limit", 100)
            response = self.search_agency.search_hot_content(time_period=time_period, limit=limit)
            
            # 檢查是否需要進行情感分析
            enable_sentiment = kwargs.get("enable_sentiment", True)
            if enable_sentiment and response.results and len(response.results) > 0:
                logger.info(f"  🎭 開始對熱點內容進行情感分析...")
                sentiment_analysis = self._perform_sentiment_analysis(response.results)
                if sentiment_analysis:
                    # 將情感分析結果添加到響應的parameters中
                    response.parameters["sentiment_analysis"] = sentiment_analysis
                    logger.info(f"  ✅ 情感分析完成")
            
            return response
        
        # 獨立情感分析工具
        if tool_name == "analyze_sentiment":
            texts = kwargs.get("texts", query)  # 可以通過texts參數傳遞，或使用query
            sentiment_result = self.analyze_sentiment_only(texts)
            
            # 構建DBResponse格式的響應
            return DBResponse(
                tool_name="analyze_sentiment",
                parameters={
                    "texts": texts if isinstance(texts, list) else [texts],
                    **kwargs
                },
                results=[],  # 情感分析不返回搜索結果
                results_count=0,
                metadata=sentiment_result
            )
        
        # 對於需要搜索詞的工具，使用關鍵詞優化中間件
        optimized_response = keyword_optimizer.optimize_keywords(
            original_query=query,
            context=f"使用{tool_name}工具進行查詢"
        )
        
        logger.info(f"  🔍 原始查詢: '{query}'")
        logger.info(f"  ✨ 優化後關鍵詞: {optimized_response.optimized_keywords}")
        
        # 使用優化後的關鍵詞進行多次查詢並整合結果
        all_results = []
        total_count = 0
        
        for keyword in optimized_response.optimized_keywords:
            logger.info(f"    查詢關鍵詞: '{keyword}'")
            
            try:
                if tool_name == "search_topic_globally":
                    # 使用配置文件中的默認值，忽略agent提供的limit_per_table參數
                    limit_per_table = self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE
                    response = self.search_agency.search_topic_globally(topic=keyword, limit_per_table=limit_per_table)
                elif tool_name == "search_topic_by_date":
                    start_date = kwargs.get("start_date")
                    end_date = kwargs.get("end_date")
                    # 使用配置文件中的默認值，忽略agent提供的limit_per_table參數
                    limit_per_table = self.config.DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE
                    if not start_date or not end_date:
                        raise ValueError("search_topic_by_date工具需要start_date和end_date參數")
                    response = self.search_agency.search_topic_by_date(topic=keyword, start_date=start_date, end_date=end_date, limit_per_table=limit_per_table)
                elif tool_name == "get_comments_for_topic":
                    # 使用配置文件中的默認值，按關鍵詞數量分配，但保證最小值
                    limit = self.config.DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT // len(optimized_response.optimized_keywords)
                    limit = max(limit, 50)
                    response = self.search_agency.get_comments_for_topic(topic=keyword, limit=limit)
                elif tool_name == "search_topic_on_platform":
                    platform = kwargs.get("platform")
                    start_date = kwargs.get("start_date")
                    end_date = kwargs.get("end_date")
                    # 使用配置文件中的默認值，按關鍵詞數量分配，但保證最小值
                    limit = self.config.DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT // len(optimized_response.optimized_keywords)
                    limit = max(limit, 30)
                    if not platform:
                        raise ValueError("search_topic_on_platform工具需要platform參數")
                    response = self.search_agency.search_topic_on_platform(platform=platform, topic=keyword, start_date=start_date, end_date=end_date, limit=limit)
                else:
                    logger.info(f"    未知的搜索工具: {tool_name}，使用默認全局搜索")
                    response = self.search_agency.search_topic_globally(topic=keyword, limit_per_table=self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE)
                
                # 收集結果
                if response.results:
                    logger.info(f"     找到 {len(response.results)} 條結果")
                    all_results.extend(response.results)
                    total_count += len(response.results)
                else:
                    logger.info(f"     未找到結果")
                    
            except Exception as e:
                logger.error(f"      查詢'{keyword}'時出錯: {str(e)}")
                continue
        
        # 去重和整合結果
        unique_results = self._deduplicate_results(all_results)
        logger.info(f"  總計找到 {total_count} 條結果，去重後 {len(unique_results)} 條")
        
        # 構建整合後的響應
        integrated_response = DBResponse(
            tool_name=f"{tool_name}_optimized",
            parameters={
                "original_query": query,
                "optimized_keywords": optimized_response.optimized_keywords,
                "optimization_reasoning": optimized_response.reasoning,
                **kwargs
            },
            results=unique_results,
            results_count=len(unique_results)
        )
        
        # 檢查是否需要進行情感分析
        enable_sentiment = kwargs.get("enable_sentiment", True)
        if enable_sentiment and unique_results and len(unique_results) > 0:
            logger.info(f"  🎭 開始對搜索結果進行情感分析...")
            sentiment_analysis = self._perform_sentiment_analysis(unique_results)
            if sentiment_analysis:
                # 將情感分析結果添加到響應的parameters中
                integrated_response.parameters["sentiment_analysis"] = sentiment_analysis
                logger.info(f"  ✅ 情感分析完成")
        
        return integrated_response
    
    def _deduplicate_results(self, results: List) -> List:
        """
        去重搜索結果
        """
        seen = set()
        unique_results = []
        
        for result in results:
            # 使用URL或內容作爲去重標識
            identifier = result.url if result.url else result.title_or_content[:100]
            if identifier not in seen:
                seen.add(identifier)
                unique_results.append(result)
        
        return unique_results
    
    def _perform_sentiment_analysis(self, results: List) -> Optional[Dict[str, Any]]:
        """
        對搜索結果執行情感分析
        
        Args:
            results: 搜索結果列表
            
        Returns:
            情感分析結果字典，如果失敗則返回None
        """
        try:
            # 初始化情感分析器（如果尚未初始化且未被禁用）
            if not self.sentiment_analyzer.is_initialized and not self.sentiment_analyzer.is_disabled:
                logger.info("    初始化情感分析模型...")
                if not self.sentiment_analyzer.initialize():
                    logger.info("     情感分析模型初始化失敗，將直接透傳原始文本")
            elif self.sentiment_analyzer.is_disabled:
                logger.info("     情感分析功能已禁用，直接透傳原始文本")

            # 將查詢結果轉換爲字典格式
            results_dict = []
            for result in results:
                result_dict = {
                    "content": result.title_or_content,
                    "platform": result.platform,
                    "author": result.author_nickname,
                    "url": result.url,
                    "publish_time": str(result.publish_time) if result.publish_time else None
                }
                results_dict.append(result_dict)
            
            # 執行情感分析
            sentiment_analysis = self.sentiment_analyzer.analyze_query_results(
                query_results=results_dict,
                text_field="content",
                min_confidence=0.5
            )
            
            return sentiment_analysis.get("sentiment_analysis")
            
        except Exception as e:
            logger.exception(f"    ❌ 情感分析過程中發生錯誤: {str(e)}")
            return None
    
    def analyze_sentiment_only(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        獨立的情感分析工具
        
        Args:
            texts: 單個文本或文本列表
            
        Returns:
            情感分析結果
        """
        logger.info(f"  → 執行獨立情感分析")
        
        try:
            # 初始化情感分析器（如果尚未初始化且未被禁用）
            if not self.sentiment_analyzer.is_initialized and not self.sentiment_analyzer.is_disabled:
                logger.info("    初始化情感分析模型...")
                if not self.sentiment_analyzer.initialize():
                    logger.info("     情感分析模型初始化失敗，將直接透傳原始文本")
            elif self.sentiment_analyzer.is_disabled:
                logger.warning("     情感分析功能已禁用，直接透傳原始文本")
            
            # 執行分析
            if isinstance(texts, str):
                result = self.sentiment_analyzer.analyze_single_text(texts)
                result_dict = result.__dict__
                response = {
                    "success": result.success and result.analysis_performed,
                    "total_analyzed": 1 if result.analysis_performed and result.success else 0,
                    "results": [result_dict]
                }
                if not result.analysis_performed:
                    response["success"] = False
                    response["warning"] = result.error_message or "情感分析功能不可用，已直接返回原始文本"
                return response
            else:
                texts_list = list(texts)
                batch_result = self.sentiment_analyzer.analyze_batch(texts_list, show_progress=True)
                response = {
                    "success": batch_result.analysis_performed and batch_result.success_count > 0,
                    "total_analyzed": batch_result.total_processed if batch_result.analysis_performed else 0,
                    "success_count": batch_result.success_count,
                    "failed_count": batch_result.failed_count,
                    "average_confidence": batch_result.average_confidence if batch_result.analysis_performed else 0.0,
                    "results": [result.__dict__ for result in batch_result.results]
                }
                if not batch_result.analysis_performed:
                    warning = next(
                        (r.error_message for r in batch_result.results if r.error_message),
                        "情感分析功能不可用，已直接返回原始文本"
                    )
                    response["success"] = False
                    response["warning"] = warning
                return response
                
        except Exception as e:
            logger.exception(f"    ❌ 情感分析過程中發生錯誤: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def research(self, query: str, save_report: bool = True) -> str:
        """
        執行深度研究
        
        Args:
            query: 研究查詢
            save_report: 是否保存報告到文件
            
        Returns:
            最終報告內容
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"開始深度研究: {query}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: 生成報告結構
            self._generate_report_structure(query)
            
            # Step 2: 處理每個段落
            self._process_paragraphs()
            
            # Step 3: 生成最終報告
            final_report = self._generate_final_report()
            
            # Step 4: 保存報告
            if save_report:
                self._save_report(final_report)

            logger.info("深度研究完成！")
            
            return final_report
            
        except Exception as e:
            logger.exception(f"研究過程中發生錯誤: {str(e)}")
            raise e
    
    def _generate_report_structure(self, query: str):
        """生成報告結構"""
        logger.info(f"\n[步驟 1] 生成報告結構...")
        
        # 創建報告結構節點
        report_structure_node = ReportStructureNode(self.llm_client, query)
        
        # 生成結構並更新狀態
        self.state = report_structure_node.mutate_state(state=self.state)
        
        _message = f"報告結構已生成，共 {len(self.state.paragraphs)} 個段落:"
        for i, paragraph in enumerate(self.state.paragraphs, 1):
            _message += f"\n  {i}. {paragraph.title}"
        logger.info(_message)
    
    def _process_paragraphs(self):
        """處理所有段落"""
        total_paragraphs = len(self.state.paragraphs)
        
        for i in range(total_paragraphs):
            logger.info(f"\n[步驟 2.{i+1}] 處理段落: {self.state.paragraphs[i].title}")
            logger.info("-" * 50)
            
            # 初始搜索和總結
            self._initial_search_and_summary(i)
            
            # 反思循環
            self._reflection_loop(i)
            
            # 標記段落完成
            self.state.paragraphs[i].research.mark_completed()
            
            progress = (i + 1) / total_paragraphs * 100
            logger.info(f"段落處理完成 ({progress:.1f}%)")
    
    def _initial_search_and_summary(self, paragraph_index: int):
        """執行初始搜索和總結"""
        paragraph = self.state.paragraphs[paragraph_index]
        
        # 準備搜索輸入
        search_input = {
            "title": paragraph.title,
            "content": paragraph.content
        }
        
        # 生成搜索查詢和工具選擇
        logger.info("  - 生成搜索查詢...")
        search_output = self.first_search_node.run(search_input)
        search_query = search_output["search_query"]
        search_tool = search_output.get("search_tool", "search_topic_globally")  # 默認工具
        reasoning = search_output["reasoning"]
        
        logger.info(f"  - 搜索查詢: {search_query}")
        logger.info(f"  - 選擇的工具: {search_tool}")
        logger.info(f"  - 推理: {reasoning}")
        
        # 執行搜索
        logger.info("  - 執行數據庫查詢...")
        
        # 處理特殊參數
        search_kwargs = {}
        
        # 處理需要日期的工具
        if search_tool in ["search_topic_by_date", "search_topic_on_platform"]:
            start_date = search_output.get("start_date")
            end_date = search_output.get("end_date")
            
            if start_date and end_date:
                # 驗證日期格式
                if self._validate_date_format(start_date) and self._validate_date_format(end_date):
                    search_kwargs["start_date"] = start_date
                    search_kwargs["end_date"] = end_date
                    logger.info(f"  - 時間範圍: {start_date} 到 {end_date}")
                else:
                    logger.info(f"    日期格式錯誤（應爲YYYY-MM-DD），改用全局搜索")
                    logger.info(f"      提供的日期: start_date={start_date}, end_date={end_date}")
                    search_tool = "search_topic_globally"
            elif search_tool == "search_topic_by_date":
                logger.info(f"    search_topic_by_date工具缺少時間參數，改用全局搜索")
                search_tool = "search_topic_globally"
        
        # 處理需要平臺參數的工具
        if search_tool == "search_topic_on_platform":
            platform = search_output.get("platform")
            if platform:
                search_kwargs["platform"] = platform
                logger.info(f"  - 指定平臺: {platform}")
            else:
                logger.warning(f"    search_topic_on_platform工具缺少平臺參數，改用全局搜索")
                search_tool = "search_topic_globally"
        
        # 處理限制參數，使用配置文件中的默認值而不是agent提供的參數
        if search_tool == "search_hot_content":
            time_period = search_output.get("time_period", "week")
            limit = self.config.DEFAULT_SEARCH_HOT_CONTENT_LIMIT
            search_kwargs["time_period"] = time_period
            search_kwargs["limit"] = limit
        elif search_tool in ["search_topic_globally", "search_topic_by_date"]:
            if search_tool == "search_topic_globally":
                limit_per_table = self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE
            else:  # search_topic_by_date
                limit_per_table = self.config.DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE
            search_kwargs["limit_per_table"] = limit_per_table
        elif search_tool in ["get_comments_for_topic", "search_topic_on_platform"]:
            if search_tool == "get_comments_for_topic":
                limit = self.config.DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT
            else:  # search_topic_on_platform
                limit = self.config.DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT
            search_kwargs["limit"] = limit
        
        search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
        
        # 轉換爲兼容格式
        search_results = []
        if search_response and search_response.results:
            # 使用配置文件控制傳遞給LLM的結果數量，0表示不限制
            if self.config.MAX_SEARCH_RESULTS_FOR_LLM > 0:
                max_results = min(len(search_response.results), self.config.MAX_SEARCH_RESULTS_FOR_LLM)
            else:
                max_results = len(search_response.results)  # 不限制，傳遞所有結果
            for result in search_response.results[:max_results]:
                search_results.append({
                    'title': result.title_or_content,
                    'url': result.url or "",
                    'content': result.title_or_content,
                    'score': result.hotness_score,
                    'raw_content': result.title_or_content,
                    'published_date': result.publish_time.isoformat() if result.publish_time else None,
                    'platform': result.platform,
                    'content_type': result.content_type,
                    'author': result.author_nickname,
                    'engagement': result.engagement
                })
        
        if search_results:
            _message = f"  - 找到 {len(search_results)} 個搜索結果"
            for j, result in enumerate(search_results, 1):
                date_info = f" (發佈於: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
                _message += f"\n    {j}. {result['title'][:50]}...{date_info}"
            logger.info(_message)
        else:
            logger.info("  - 未找到搜索結果")
        
        # 更新狀態中的搜索歷史
        paragraph.research.add_search_results(search_query, search_results)
        
        # 生成初始總結
        logger.info("  - 生成初始總結...")
        summary_input = {
            "title": paragraph.title,
            "content": paragraph.content,
            "search_query": search_query,
            "search_results": format_search_results_for_prompt(
                search_results, self.config.MAX_CONTENT_LENGTH
            )
        }
        
        # 更新狀態
        self.state = self.first_summary_node.mutate_state(
            summary_input, self.state, paragraph_index
        )
        
        logger.info("  - 初始總結完成")
    
    def _reflection_loop(self, paragraph_index: int):
        """執行反思循環"""
        paragraph = self.state.paragraphs[paragraph_index]
        
        for reflection_i in range(self.config.MAX_REFLECTIONS):
            logger.info(f"  - 反思 {reflection_i + 1}/{self.config.MAX_REFLECTIONS}...")
            
            # 準備反思輸入
            reflection_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "paragraph_latest_state": paragraph.research.latest_summary
            }
            
            # 生成反思搜索查詢
            reflection_output = self.reflection_node.run(reflection_input)
            search_query = reflection_output["search_query"]
            search_tool = reflection_output.get("search_tool", "search_topic_globally")  # 默認工具
            reasoning = reflection_output["reasoning"]
            
            logger.info(f"    反思查詢: {search_query}")
            logger.info(f"    選擇的工具: {search_tool}")
            logger.info(f"    反思推理: {reasoning}")
            
            # 執行反思搜索
            # 處理特殊參數
            search_kwargs = {}
            
            # 處理需要日期的工具
            if search_tool in ["search_topic_by_date", "search_topic_on_platform"]:
                start_date = reflection_output.get("start_date")
                end_date = reflection_output.get("end_date")
                
                if start_date and end_date:
                    # 驗證日期格式
                    if self._validate_date_format(start_date) and self._validate_date_format(end_date):
                        search_kwargs["start_date"] = start_date
                        search_kwargs["end_date"] = end_date
                        logger.info(f"    時間範圍: {start_date} 到 {end_date}")
                    else:
                        logger.info(f"      日期格式錯誤（應爲YYYY-MM-DD），改用全局搜索")
                        logger.info(f"        提供的日期: start_date={start_date}, end_date={end_date}")
                        search_tool = "search_topic_globally"
                elif search_tool == "search_topic_by_date":
                    logger.warning(f"      search_topic_by_date工具缺少時間參數，改用全局搜索")
                    search_tool = "search_topic_globally"
            
            # 處理需要平臺參數的工具
            if search_tool == "search_topic_on_platform":
                platform = reflection_output.get("platform")
                if platform:
                    search_kwargs["platform"] = platform
                    logger.info(f"    指定平臺: {platform}")
                else:
                    logger.warning(f"      search_topic_on_platform工具缺少平臺參數，改用全局搜索")
                    search_tool = "search_topic_globally"
            
            # 處理限制參數
            if search_tool == "search_hot_content":
                time_period = reflection_output.get("time_period", "week")
                # 使用配置文件中的默認值，不允許agent控制limit參數
                limit = self.config.DEFAULT_SEARCH_HOT_CONTENT_LIMIT
                search_kwargs["time_period"] = time_period
                search_kwargs["limit"] = limit
            elif search_tool in ["search_topic_globally", "search_topic_by_date"]:
                # 使用配置文件中的默認值，不允許agent控制limit_per_table參數
                if search_tool == "search_topic_globally":
                    limit_per_table = self.config.DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE
                else:  # search_topic_by_date
                    limit_per_table = self.config.DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE
                search_kwargs["limit_per_table"] = limit_per_table
            elif search_tool in ["get_comments_for_topic", "search_topic_on_platform"]:
                # 使用配置文件中的默認值，不允許agent控制limit參數
                if search_tool == "get_comments_for_topic":
                    limit = self.config.DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT
                else:  # search_topic_on_platform
                    limit = self.config.DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT
                search_kwargs["limit"] = limit
            
            search_response = self.execute_search_tool(search_tool, search_query, **search_kwargs)
            
            # 轉換爲兼容格式
            search_results = []
            if search_response and search_response.results:
                # 使用配置文件控制傳遞給LLM的結果數量，0表示不限制
                if self.config.MAX_SEARCH_RESULTS_FOR_LLM > 0:
                    max_results = min(len(search_response.results), self.config.MAX_SEARCH_RESULTS_FOR_LLM)
                else:
                    max_results = len(search_response.results)  # 不限制，傳遞所有結果
                for result in search_response.results[:max_results]:
                    search_results.append({
                        'title': result.title_or_content,
                        'url': result.url or "",
                        'content': result.title_or_content,
                        'score': result.hotness_score,
                        'raw_content': result.title_or_content,
                        'published_date': result.publish_time.isoformat() if result.publish_time else None,
                        'platform': result.platform,
                        'content_type': result.content_type,
                        'author': result.author_nickname,
                        'engagement': result.engagement
                    })
            
            if search_results:
                _message = f"    找到 {len(search_results)} 個反思搜索結果"
                for j, result in enumerate(search_results, 1):
                    date_info = f" (發佈於: {result.get('published_date', 'N/A')})" if result.get('published_date') else ""
                    _message += f"\n      {j}. {result['title'][:50]}...{date_info}"
                logger.info(_message)
            else:
                logger.info("    未找到反思搜索結果")
            
            # 更新搜索歷史
            paragraph.research.add_search_results(search_query, search_results)
            
            # 生成反思總結
            reflection_summary_input = {
                "title": paragraph.title,
                "content": paragraph.content,
                "search_query": search_query,
                "search_results": format_search_results_for_prompt(
                    search_results, self.config.MAX_CONTENT_LENGTH
                ),
                "paragraph_latest_state": paragraph.research.latest_summary
            }
            
            # 更新狀態
            self.state = self.reflection_summary_node.mutate_state(
                reflection_summary_input, self.state, paragraph_index
            )
            
            logger.info(f"    反思 {reflection_i + 1} 完成")
    
    def _generate_final_report(self) -> str:
        """生成最終報告"""
        logger.info(f"\n[步驟 3] 生成最終報告...")
        
        # 準備報告數據
        report_data = []
        for paragraph in self.state.paragraphs:
            report_data.append({
                "title": paragraph.title,
                "paragraph_latest_state": paragraph.research.latest_summary
            })
        
        # 格式化報告
        try:
            final_report = self.report_formatting_node.run(report_data)
        except Exception as e:
            logger.exception(f"LLM格式化失敗，使用備用方法: {str(e)}")
            final_report = self.report_formatting_node.format_report_manually(
                report_data, self.state.report_title
            )
        
        # 更新狀態
        self.state.final_report = final_report
        self.state.mark_completed()
        
        logger.info("最終報告生成完成")
        return final_report
    
    def _save_report(self, report_content: str):
        """保存報告到文件"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]
        
        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        # 保存報告
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"報告已保存到: {filepath}")
        
        # 保存狀態（如果配置允許）
        if self.config.SAVE_INTERMEDIATE_STATES:
            state_filename = f"state_{query_safe}_{timestamp}.json"
            state_filepath = os.path.join(self.config.OUTPUT_DIR, state_filename)
            self.state.save_to_file(state_filepath)
            logger.info(f"狀態已保存到: {state_filepath}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """獲取進度摘要"""
        return self.state.get_progress_summary()
    
    def load_state(self, filepath: str):
        """從文件加載狀態"""
        self.state = State.load_from_file(filepath)
        logger.info(f"狀態已從 {filepath} 加載")
    
    def save_state(self, filepath: str):
        """保存狀態到文件"""
        self.state.save_to_file(filepath)
        logger.info(f"狀態已保存到 {filepath}")


def create_agent(config_file: Optional[str] = None) -> DeepSearchAgent:
    """
    創建Deep Search Agent實例的便捷函數
    
    Args:
        config_file: 配置文件路徑
        
    Returns:
        DeepSearchAgent實例
    """
    config = settings
    return DeepSearchAgent(config)

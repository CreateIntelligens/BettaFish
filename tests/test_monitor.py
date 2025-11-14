"""
測試ForumEngine/monitor.py中的日誌解析函數

測試各種日誌格式下的解析能力，包括：
1. 舊格式：[HH:MM:SS]
2. 新格式：loguru默認格式 (YYYY-MM-DD HH:mm:ss.SSS | LEVEL | ...)
3. 只應當接收FirstSummaryNode、ReflectionSummaryNode等SummaryNode的輸出，不應當接收SearchNode的輸出
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ForumEngine.monitor import LogMonitor
from tests import forum_log_test_data as test_data


class TestLogMonitor:
    """測試LogMonitor的日誌解析功能"""
    
    def setup_method(self):
        """每個測試方法前的初始化"""
        self.monitor = LogMonitor(log_dir="tests/test_logs")
    
    def test_is_target_log_line_old_format(self):
        """測試舊格式的目標節點識別"""
        # 應該識別包含FirstSummaryNode的行
        assert self.monitor.is_target_log_line(test_data.OLD_FORMAT_FIRST_SUMMARY) == True
        # 應該識別包含ReflectionSummaryNode的行
        assert self.monitor.is_target_log_line(test_data.OLD_FORMAT_REFLECTION_SUMMARY) == True
        # 不應該識別非目標節點
        assert self.monitor.is_target_log_line(test_data.OLD_FORMAT_NON_TARGET) == False
    
    def test_is_target_log_line_new_format(self):
        """測試新格式的目標節點識別"""
        # 應該識別包含FirstSummaryNode的行
        assert self.monitor.is_target_log_line(test_data.NEW_FORMAT_FIRST_SUMMARY) == True
        # 應該識別包含ReflectionSummaryNode的行
        assert self.monitor.is_target_log_line(test_data.NEW_FORMAT_REFLECTION_SUMMARY) == True
        # 不應該識別非目標節點
        assert self.monitor.is_target_log_line(test_data.NEW_FORMAT_NON_TARGET) == False
    
    def test_is_json_start_line_old_format(self):
        """測試舊格式的JSON開始行識別"""
        assert self.monitor.is_json_start_line(test_data.OLD_FORMAT_SINGLE_LINE_JSON) == True
        assert self.monitor.is_json_start_line(test_data.OLD_FORMAT_MULTILINE_JSON[0]) == True
        assert self.monitor.is_json_start_line(test_data.OLD_FORMAT_NON_TARGET) == False
    
    def test_is_json_start_line_new_format(self):
        """測試新格式的JSON開始行識別"""
        assert self.monitor.is_json_start_line(test_data.NEW_FORMAT_SINGLE_LINE_JSON) == True
        assert self.monitor.is_json_start_line(test_data.NEW_FORMAT_MULTILINE_JSON[0]) == True
        assert self.monitor.is_json_start_line(test_data.NEW_FORMAT_NON_TARGET) == False
    
    def test_is_json_end_line(self):
        """測試JSON結束行識別"""
        assert self.monitor.is_json_end_line("}") == True
        assert self.monitor.is_json_end_line("] }") == True
        assert self.monitor.is_json_end_line("[17:42:31] }") == False  # 需要先清理時間戳
        assert self.monitor.is_json_end_line("2025-11-05 17:42:31.289 | INFO | module:function:133 - }") == False  # 需要先清理時間戳
    
    def test_extract_json_content_old_format_single_line(self):
        """測試舊格式單行JSON提取"""
        lines = [test_data.OLD_FORMAT_SINGLE_LINE_JSON]
        result = self.monitor.extract_json_content(lines)
        assert result is not None
        assert "這是首次總結內容" in result
    
    def test_extract_json_content_new_format_single_line(self):
        """測試新格式單行JSON提取"""
        lines = [test_data.NEW_FORMAT_SINGLE_LINE_JSON]
        result = self.monitor.extract_json_content(lines)
        assert result is not None
        assert "這是首次總結內容" in result
    
    def test_extract_json_content_old_format_multiline(self):
        """測試舊格式多行JSON提取"""
        result = self.monitor.extract_json_content(test_data.OLD_FORMAT_MULTILINE_JSON)
        assert result is not None
        assert "多行" in result
        assert "JSON內容" in result
    
    def test_extract_json_content_new_format_multiline(self):
        """測試新格式多行JSON提取（支持loguru格式的時間戳移除）"""
        result = self.monitor.extract_json_content(test_data.NEW_FORMAT_MULTILINE_JSON)
        assert result is not None
        assert "多行" in result
        assert "JSON內容" in result
    
    def test_extract_json_content_updated_priority(self):
        """測試updated_paragraph_latest_state優先提取"""
        result = self.monitor.extract_json_content(test_data.COMPLEX_JSON_WITH_UPDATED)
        assert result is not None
        assert "更新版" in result
        assert "核心發現" in result
    
    def test_extract_json_content_paragraph_only(self):
        """測試只有paragraph_latest_state的情況"""
        result = self.monitor.extract_json_content(test_data.COMPLEX_JSON_WITH_PARAGRAPH)
        assert result is not None
        assert "首次總結" in result or "核心發現" in result
    
    def test_format_json_content(self):
        """測試JSON內容格式化"""
        # 測試updated_paragraph_latest_state優先
        json_obj = {
            "updated_paragraph_latest_state": "更新後的內容",
            "paragraph_latest_state": "首次內容"
        }
        result = self.monitor.format_json_content(json_obj)
        assert result == "更新後的內容"
        
        # 測試只有paragraph_latest_state
        json_obj = {
            "paragraph_latest_state": "首次內容"
        }
        result = self.monitor.format_json_content(json_obj)
        assert result == "首次內容"
        
        # 測試都沒有的情況
        json_obj = {"other_field": "其他內容"}
        result = self.monitor.format_json_content(json_obj)
        assert "清理後的輸出" in result
    
    def test_extract_node_content_old_format(self):
        """測試舊格式的節點內容提取"""
        line = "[17:42:31] [INSIGHT] [FirstSummaryNode] 清理後的輸出: 這是測試內容"
        result = self.monitor.extract_node_content(line)
        assert result is not None
        assert "測試內容" in result
    
    def test_extract_node_content_new_format(self):
        """測試新格式的節點內容提取"""
        line = "2025-11-05 17:42:31.287 | INFO | InsightEngine.nodes.summary_node:process_output:131 - FirstSummaryNode 清理後的輸出: 這是測試內容"
        result = self.monitor.extract_node_content(line)
        assert result is not None
        assert "測試內容" in result
    
    def test_process_lines_for_json_old_format(self):
        """測試舊格式的完整處理流程"""
        lines = [
            test_data.OLD_FORMAT_NON_TARGET,  # 應該被忽略
            test_data.OLD_FORMAT_MULTILINE_JSON[0],
            test_data.OLD_FORMAT_MULTILINE_JSON[1],
            test_data.OLD_FORMAT_MULTILINE_JSON[2],
        ]
        result = self.monitor.process_lines_for_json(lines, "insight")
        assert len(result) > 0
        assert any("多行" in content for content in result)
    
    def test_process_lines_for_json_new_format(self):
        """測試新格式的完整處理流程"""
        lines = [
            test_data.NEW_FORMAT_NON_TARGET,  # 應該被忽略
            test_data.NEW_FORMAT_MULTILINE_JSON[0],
            test_data.NEW_FORMAT_MULTILINE_JSON[1],
            test_data.NEW_FORMAT_MULTILINE_JSON[2],
        ]
        result = self.monitor.process_lines_for_json(lines, "insight")
        assert len(result) > 0
        assert any("多行" in content for content in result)
        assert any("JSON內容" in content for content in result)
    
    def test_process_lines_for_json_mixed_format(self):
        """測試混合格式的處理"""
        result = self.monitor.process_lines_for_json(test_data.MIXED_FORMAT_LINES, "insight")
        assert len(result) > 0
        assert any("混合格式內容" in content for content in result)
    
    def test_is_valuable_content(self):
        """測試有價值內容的判斷"""
        # 包含"清理後的輸出"應該是有價值的
        assert self.monitor.is_valuable_content(test_data.OLD_FORMAT_SINGLE_LINE_JSON) == True
        
        # 排除短小提示信息
        assert self.monitor.is_valuable_content("JSON解析成功") == False
        assert self.monitor.is_valuable_content("成功生成") == False
        
        # 空行應該被過濾
        assert self.monitor.is_valuable_content("") == False
    
    def test_extract_json_content_real_query_engine(self):
        """測試QueryEngine實際生產環境日誌提取"""
        result = self.monitor.extract_json_content(test_data.REAL_QUERY_ENGINE_REFLECTION)
        assert result is not None
        assert "洛陽欒川鉬業集團" in result
        assert "CMOC" in result
        assert "updated_paragraph_latest_state" not in result  # 應該已經提取內容，不包含字段名
    
    def test_extract_json_content_real_insight_engine(self):
        """測試InsightEngine實際生產環境日誌提取（包含標識行）"""
        # 先測試能否識別標識行
        assert self.monitor.is_target_log_line(test_data.REAL_INSIGHT_ENGINE_REFLECTION[0]) == True  # 包含"正在生成反思總結"
        assert self.monitor.is_target_log_line(test_data.REAL_INSIGHT_ENGINE_REFLECTION[1]) == True  # 包含nodes.summary_node
        
        # 測試JSON提取（從第二行開始，因爲第一行是標識行）
        json_lines = test_data.REAL_INSIGHT_ENGINE_REFLECTION[1:]  # 跳過標識行
        result = self.monitor.extract_json_content(json_lines)
        assert result is not None
        assert "核心發現" in result
        assert "更新版" in result
        assert "洛陽鉬業2025年第三季度" in result
    
    def test_extract_json_content_real_media_engine(self):
        """測試MediaEngine實際生產環境日誌提取（單行JSON）"""
        # MediaEngine是單行JSON格式，需要先分割成行
        lines = test_data.REAL_MEDIA_ENGINE_REFLECTION.split('\n')
        
        # 測試能否識別標識行
        assert self.monitor.is_target_log_line(lines[0]) == True  # 包含"正在生成反思總結"
        assert self.monitor.is_target_log_line(lines[1]) == True  # 包含nodes.summary_node和"清理後的輸出"
        
        # 測試JSON提取（從包含JSON的行開始）
        json_line = lines[1]  # 第二行包含完整的單行JSON
        result = self.monitor.extract_json_content([json_line])
        assert result is not None
        assert "綜合信息概覽" in result
        assert "洛陽鉬業" in result
        assert "updated_paragraph_latest_state" not in result  # 應該已經提取內容
    
    def test_process_lines_for_json_real_query_engine(self):
        """測試QueryEngine實際日誌的完整處理流程"""
        result = self.monitor.process_lines_for_json(test_data.REAL_QUERY_ENGINE_REFLECTION, "query")
        assert len(result) > 0
        assert any("洛陽欒川鉬業集團" in content for content in result)
    
    def test_process_lines_for_json_real_insight_engine(self):
        """測試InsightEngine實際日誌的完整處理流程（包含標識行）"""
        result = self.monitor.process_lines_for_json(test_data.REAL_INSIGHT_ENGINE_REFLECTION, "insight")
        assert len(result) > 0
        assert any("核心發現" in content for content in result)
        assert any("更新版" in content for content in result)
    
    def test_process_lines_for_json_real_media_engine(self):
        """測試MediaEngine實際日誌的完整處理流程（單行JSON）"""
        # 將單行字符串分割成多行
        lines = test_data.REAL_MEDIA_ENGINE_REFLECTION.split('\n')
        result = self.monitor.process_lines_for_json(lines, "media")
        assert len(result) > 0
        assert any("綜合信息概覽" in content for content in result)
        assert any("洛陽鉬業" in content for content in result)
    
    def test_filter_search_node_output(self):
        """測試過濾SearchNode的輸出（重要：SearchNode不應進入論壇）"""
        # SearchNode的輸出包含"清理後的輸出: {"，但不包含目標節點模式
        search_lines = test_data.SEARCH_NODE_FIRST_SEARCH
        result = self.monitor.process_lines_for_json(search_lines, "insight")
        # SearchNode的輸出應該被過濾，不應該被捕獲
        assert len(result) == 0
    
    def test_filter_search_node_output_single_line(self):
        """測試過濾SearchNode的單行JSON輸出"""
        # SearchNode的單行JSON格式
        search_line = test_data.SEARCH_NODE_REFLECTION_SEARCH
        result = self.monitor.process_lines_for_json([search_line], "insight")
        # SearchNode的輸出應該被過濾
        assert len(result) == 0
    
    def test_search_node_vs_summary_node_mixed(self):
        """測試混合場景：SearchNode和SummaryNode同時存在，只捕獲SummaryNode"""
        lines = [
            # SearchNode輸出（應該被過濾）
            "[11:16:35] 2025-11-06 11:16:35.567 | INFO | InsightEngine.nodes.search_node:process_output:97 - 清理後的輸出: {",
            "[11:16:35] \"search_query\": \"測試查詢\"",
            "[11:16:35] }",
            # SummaryNode輸出（應該被捕獲）
            "[11:17:05] 2025-11-06 11:17:05.547 | INFO | InsightEngine.nodes.summary_node:process_output:131 - 清理後的輸出: {",
            "[11:17:05] \"paragraph_latest_state\": \"這是總結內容\"",
            "[11:17:05] }",
        ]
        result = self.monitor.process_lines_for_json(lines, "insight")
        # 應該只捕獲SummaryNode的輸出，不包含SearchNode的輸出
        assert len(result) > 0
        assert any("總結內容" in content for content in result)
        # 確保不包含搜索查詢內容
        assert not any("search_query" in content for content in result)
        assert not any("測試查詢" in content for content in result)
    
    def test_filter_error_logs_from_summary_node(self):
        """測試過濾SummaryNode的錯誤日誌（重要：錯誤日誌不應進入論壇）"""
        # JSON解析失敗錯誤日誌
        assert self.monitor.is_target_log_line(test_data.SUMMARY_NODE_JSON_ERROR) == False
        
        # JSON修復失敗錯誤日誌
        assert self.monitor.is_target_log_line(test_data.SUMMARY_NODE_JSON_FIX_ERROR) == False
        
        # ERROR級別日誌
        assert self.monitor.is_target_log_line(test_data.SUMMARY_NODE_ERROR_LOG) == False
        
        # Traceback錯誤日誌
        for line in test_data.SUMMARY_NODE_TRACEBACK.split('\n'):
            assert self.monitor.is_target_log_line(line) == False
    
    def test_error_logs_not_captured(self):
        """測試錯誤日誌不會被捕獲到論壇"""
        error_lines = [
            test_data.SUMMARY_NODE_JSON_ERROR,
            test_data.SUMMARY_NODE_JSON_FIX_ERROR,
            test_data.SUMMARY_NODE_ERROR_LOG,
        ]
        
        for line in error_lines:
            result = self.monitor.process_lines_for_json([line], "media")
            # 錯誤日誌不應該被捕獲
            assert len(result) == 0
    
    def test_mixed_valid_and_error_logs(self):
        """測試混合場景：有效日誌和錯誤日誌同時存在，只捕獲有效日誌"""
        lines = [
            # 錯誤日誌（應該被過濾）
            test_data.SUMMARY_NODE_JSON_ERROR,
            test_data.SUMMARY_NODE_JSON_FIX_ERROR,
            # 有效SummaryNode輸出（應該被捕獲）
            "[11:55:31] 2025-11-06 11:55:31.762 | INFO | MediaEngine.nodes.summary_node:process_output:134 - 清理後的輸出: {",
            "[11:55:31] \"paragraph_latest_state\": \"這是有效的總結內容\"",
            "[11:55:31] }",
        ]
        result = self.monitor.process_lines_for_json(lines, "media")
        # 應該只捕獲有效日誌，不包含錯誤日誌
        assert len(result) > 0
        assert any("有效的總結內容" in content for content in result)
        # 確保不包含錯誤信息
        assert not any("JSON解析失敗" in content for content in result)
        assert not any("JSON修復失敗" in content for content in result)


def run_tests():
    """運行所有測試"""
    import pytest
    
    # 運行測試
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()


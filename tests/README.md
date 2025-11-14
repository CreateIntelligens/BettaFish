# ForumEngine日誌解析測試

本測試套件用於測試 `ForumEngine/monitor.py` 中的日誌解析功能，驗證其在不同日誌格式下的正確性。

## 測試數據

`forum_log_test_data.py` 包含各種日誌格式的最小示例（論壇日誌測試數據）：

### 舊格式（[HH:MM:SS]）
- `OLD_FORMAT_SINGLE_LINE_JSON`: 單行JSON
- `OLD_FORMAT_MULTILINE_JSON`: 多行JSON
- `OLD_FORMAT_FIRST_SUMMARY`: 包含FirstSummaryNode的日誌
- `OLD_FORMAT_REFLECTION_SUMMARY`: 包含ReflectionSummaryNode的日誌

### 新格式（loguru默認格式）
- `NEW_FORMAT_SINGLE_LINE_JSON`: 單行JSON
- `NEW_FORMAT_MULTILINE_JSON`: 多行JSON
- `NEW_FORMAT_FIRST_SUMMARY`: 包含FirstSummaryNode的日誌
- `NEW_FORMAT_REFLECTION_SUMMARY`: 包含ReflectionSummaryNode的日誌

### 複雜示例
- `COMPLEX_JSON_WITH_UPDATED`: 包含updated_paragraph_latest_state的JSON
- `COMPLEX_JSON_WITH_PARAGRAPH`: 只有paragraph_latest_state的JSON
- `MIXED_FORMAT_LINES`: 混合格式的日誌行

## 運行測試

### 使用pytest（推薦）

```bash
# 安裝pytest（如果還沒有安裝）
pip install pytest

# 運行所有測試
pytest tests/test_monitor.py -v

# 運行特定測試
pytest tests/test_monitor.py::TestLogMonitor::test_extract_json_content_new_format_multiline -v
```

### 直接運行

```bash
python tests/test_monitor.py
```

## 測試覆蓋

測試覆蓋以下函數：

1. **is_target_log_line**: 識別目標節點日誌行
2. **is_json_start_line**: 識別JSON開始行
3. **is_json_end_line**: 識別JSON結束行
4. **extract_json_content**: 提取JSON內容（單行和多行）
5. **format_json_content**: 格式化JSON內容（優先提取updated_paragraph_latest_state）
6. **extract_node_content**: 提取節點內容
7. **process_lines_for_json**: 完整處理流程
8. **is_valuable_content**: 判斷內容是否有價值

## 預期問題

當前代碼可能無法正確處理loguru新格式，主要問題在於：

1. **時間戳移除**：`extract_json_content()` 中的正則 `r'^\[\d{2}:\d{2}:\d{2}\]\s*'` 只能匹配 `[HH:MM:SS]` 格式，無法匹配loguru的 `YYYY-MM-DD HH:mm:ss.SSS` 格式

2. **時間戳匹配**：`extract_node_content()` 中的正則 `r'\[\d{2}:\d{2}:\d{2}\]\s*(.+)'` 同樣只能匹配舊格式

這些測試會幫助識別這些問題，並指導後續的代碼修復。


"""
Report Engine提示詞模塊
定義報告生成各個階段使用的系統提示詞
"""

from .prompts import (
    SYSTEM_PROMPT_TEMPLATE_SELECTION,
    SYSTEM_PROMPT_HTML_GENERATION,
    output_schema_template_selection,
    input_schema_html_generation
)

__all__ = [
    "SYSTEM_PROMPT_TEMPLATE_SELECTION",
    "SYSTEM_PROMPT_HTML_GENERATION", 
    "output_schema_template_selection",
    "input_schema_html_generation"
]

"""
Report Engine
一個智能報告生成AI代理實現
基於三個子agent的輸出和論壇日誌生成綜合HTML報告
"""

from .agent import ReportAgent, create_agent

__version__ = "1.0.0"
__author__ = "Report Engine Team"

__all__ = ["ReportAgent", "create_agent"]

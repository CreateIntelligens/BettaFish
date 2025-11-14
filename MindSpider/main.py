#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindSpider - AI爬蟲項目主程序
集成BroadTopicExtraction和DeepSentimentCrawling兩個核心模塊
"""

import os
import sys
import argparse
from datetime import date, datetime
from pathlib import Path
import subprocess
import asyncio
import pymysql
from pymysql.cursors import DictCursor
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import inspect, text
from config import settings
from loguru import logger
from urllib.parse import quote_plus

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    logger.error("錯誤：無法導入config.py配置文件")
    logger.error("請確保項目根目錄下存在config.py文件，幷包含數據庫和API配置信息")
    sys.exit(1)

class MindSpider:
    """MindSpider主程序"""
    
    def __init__(self):
        """初始化MindSpider"""
        self.project_root = project_root
        self.broad_topic_path = self.project_root / "BroadTopicExtraction"
        self.deep_sentiment_path = self.project_root / "DeepSentimentCrawling"
        self.schema_path = self.project_root / "schema"
        
        logger.info("MindSpider AI爬蟲項目")
        logger.info(f"項目路徑: {self.project_root}")
    
    def check_config(self) -> bool:
        """檢查基礎配置"""
        logger.info("檢查基礎配置...")
        
        # 檢查settings配置項
        required_configs = [
            'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_CHARSET',
            'MINDSPIDER_API_KEY', 'MINDSPIDER_BASE_URL', 'MINDSPIDER_MODEL_NAME'
        ]
        
        missing_configs = []
        for config_name in required_configs:
            if not hasattr(settings, config_name) or not getattr(settings, config_name):
                missing_configs.append(config_name)
        
        if missing_configs:
            logger.error(f"配置缺失: {', '.join(missing_configs)}")
            logger.error("請檢查.env文件中的環境變量配置信息")
            return False
        
        logger.info("基礎配置檢查通過")
        return True
    
    def check_database_connection(self) -> bool:
        """檢查數據庫連接"""
        logger.info("檢查數據庫連接...")
        
        def build_async_url() -> str:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect == "postgresql":
                return (
                    f"postgresql+asyncpg://{settings.DB_USER}:"
                    f"{quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}:"
                    f"{settings.DB_PORT}/{settings.DB_NAME}"
                )
            # 默認使用 mysql 異步驅動 asyncmy
            return (
                f"mysql+asyncmy://{settings.DB_USER}:"
                f"{quote_plus(settings.DB_PASSWORD)}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            )

        async def _test_connection(db_url: str) -> None:
            engine: AsyncEngine = create_async_engine(db_url, pool_pre_ping=True)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            finally:
                await engine.dispose()

        try:
            db_url: str = build_async_url()
            asyncio.run(_test_connection(db_url))
            logger.info("數據庫連接正常")
            return True
        except Exception as e:
            logger.exception(f"數據庫連接失敗: {e}")
            return False
    
    def check_database_tables(self) -> bool:
        """檢查數據庫表是否存在"""
        logger.info("檢查數據庫表...")
        
        def build_async_url() -> str:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect == "postgresql":
                return f"postgresql+asyncpg://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            return (
                f"mysql+asyncmy://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}"
                f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            )

        async def _check_tables(db_url: str) -> list[str]:
            engine: AsyncEngine = create_async_engine(db_url, pool_pre_ping=True)
            try:
                async with engine.connect() as conn:
                    def _get_tables(sync_conn):
                        return inspect(sync_conn).get_table_names()
                    tables = await conn.run_sync(_get_tables)
                    return tables
            finally:
                await engine.dispose()

        try:
            db_url: str = build_async_url()
            existing_tables = asyncio.run(_check_tables(db_url))
            required_tables = ['daily_news', 'daily_topics']
            missing_tables = [t for t in required_tables if t not in existing_tables]
            if missing_tables:
                logger.error(f"缺少數據庫表: {', '.join(missing_tables)}")
                return False
            logger.info("數據庫表檢查通過")
            return True
        except Exception as e:
            logger.exception(f"檢查數據庫表失敗: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """初始化數據庫"""
        logger.info("初始化數據庫...")
        
        try:
            # 運行數據庫初始化腳本
            init_script = self.schema_path / "init_database.py"
            if not init_script.exists():
                logger.error("錯誤：找不到數據庫初始化腳本")
                return False
            
            result = subprocess.run(
                [sys.executable, str(init_script)],
                cwd=self.schema_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("數據庫初始化成功")
                return True
            else:
                logger.error(f"數據庫初始化失敗: {result.stderr}")
                return False
                
        except Exception as e:
            logger.exception(f"數據庫初始化異常: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """檢查依賴環境"""
        logger.info("檢查依賴環境...")
        
        # 檢查Python包
        required_packages = ['pymysql', 'requests', 'playwright']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少Python包: {', '.join(missing_packages)}")
            logger.info("請運行: pip install -r requirements.txt")
            return False
        
        # 檢查MediaCrawler依賴
        mediacrawler_path = self.deep_sentiment_path / "MediaCrawler"
        if not mediacrawler_path.exists():
            logger.error("錯誤：找不到MediaCrawler目錄")
            return False
        
        logger.info("依賴環境檢查通過")
        return True
    
    def run_broad_topic_extraction(self, extract_date: date = None, keywords_count: int = 100) -> bool:
        """運行BroadTopicExtraction模塊"""
        logger.info("運行BroadTopicExtraction模塊...")
        
        if not extract_date:
            extract_date = date.today()
        
        try:
            cmd = [
                sys.executable, "main.py",
                "--keywords", str(keywords_count)
            ]
            
            logger.info(f"執行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.broad_topic_path,
                timeout=1800  # 30分鐘超時
            )
            
            if result.returncode == 0:
                logger.info("BroadTopicExtraction模塊執行成功")
                return True
            else:
                logger.error(f"BroadTopicExtraction模塊執行失敗，返回碼: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("BroadTopicExtraction模塊執行超時")
            return False
        except Exception as e:
            logger.exception(f"BroadTopicExtraction模塊執行異常: {e}")
            return False
    
    def run_deep_sentiment_crawling(self, target_date: date = None, platforms: list = None,
                                   max_keywords: int = 50, max_notes: int = 50,
                                   test_mode: bool = False) -> bool:
        """運行DeepSentimentCrawling模塊"""
        logger.info("運行DeepSentimentCrawling模塊...")
        
        if not target_date:
            target_date = date.today()
        
        try:
            cmd = [sys.executable, "main.py"]
            
            if target_date:
                cmd.extend(["--date", target_date.strftime("%Y-%m-%d")])
            
            if platforms:
                cmd.extend(["--platforms"] + platforms)
            
            cmd.extend([
                "--max-keywords", str(max_keywords),
                "--max-notes", str(max_notes)
            ])
            
            if test_mode:
                cmd.append("--test")
            
            logger.info(f"執行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.deep_sentiment_path,
                timeout=3600  # 60分鐘超時
            )
            
            if result.returncode == 0:
                logger.info("DeepSentimentCrawling模塊執行成功")
                return True
            else:
                logger.error(f"DeepSentimentCrawling模塊執行失敗，返回碼: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("DeepSentimentCrawling模塊執行超時")
            return False
        except Exception as e:
            logger.exception(f"DeepSentimentCrawling模塊執行異常: {e}")
            return False
    
    def run_complete_workflow(self, target_date: date = None, platforms: list = None,
                             keywords_count: int = 100, max_keywords: int = 50,
                             max_notes: int = 50, test_mode: bool = False) -> bool:
        """運行完整工作流程"""
        logger.info("開始完整的MindSpider工作流程")
        
        if not target_date:
            target_date = date.today()
        
        logger.info(f"目標日期: {target_date}")
        logger.info(f"平臺列表: {platforms if platforms else '所有支持的平臺'}")
        logger.info(f"測試模式: {'是' if test_mode else '否'}")
        
        # 第一步：運行話題提取
        logger.info("=== 第一步：話題提取 ===")
        if not self.run_broad_topic_extraction(target_date, keywords_count):
            logger.error("話題提取失敗，終止流程")
            return False
        
        # 第二步：運行情感爬取
        logger.info("=== 第二步：情感爬取 ===")
        if not self.run_deep_sentiment_crawling(target_date, platforms, max_keywords, max_notes, test_mode):
            logger.error("情感爬取失敗，但話題提取已完成")
            return False
        
        logger.info("完整工作流程執行成功！")
        return True
    
    def show_status(self):
        """顯示項目狀態"""
        logger.info("MindSpider項目狀態:")
        logger.info(f"項目路徑: {self.project_root}")
        
        # 配置狀態
        config_ok = self.check_config()
        logger.info(f"配置狀態: {'正常' if config_ok else '異常'}")
        
        # 數據庫狀態
        if config_ok:
            db_conn_ok = self.check_database_connection()
            logger.info(f"數據庫連接: {'正常' if db_conn_ok else '異常'}")
            
            if db_conn_ok:
                db_tables_ok = self.check_database_tables()
                logger.info(f"數據庫表: {'正常' if db_tables_ok else '需要初始化'}")
        
        # 依賴狀態
        deps_ok = self.check_dependencies()
        logger.info(f"依賴環境: {'正常' if deps_ok else '異常'}")
        
        # 模塊狀態
        broad_topic_exists = self.broad_topic_path.exists()
        deep_sentiment_exists = self.deep_sentiment_path.exists()
        logger.info(f"BroadTopicExtraction模塊: {'存在' if broad_topic_exists else '缺失'}")
        logger.info(f"DeepSentimentCrawling模塊: {'存在' if deep_sentiment_exists else '缺失'}")
    
    def setup_project(self) -> bool:
        """項目初始化設置"""
        logger.info("開始MindSpider項目初始化...")
        
        # 1. 檢查配置
        if not self.check_config():
            return False
        
        # 2. 檢查依賴
        if not self.check_dependencies():
            return False
        
        # 3. 檢查數據庫連接
        if not self.check_database_connection():
            return False
        
        # 4. 檢查並初始化數據庫表
        if not self.check_database_tables():
            logger.info("需要初始化數據庫表...")
            if not self.initialize_database():
                return False
        
        logger.info("MindSpider項目初始化完成！")
        return True

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="MindSpider - AI爬蟲項目主程序")
    
    # 基本操作
    parser.add_argument("--setup", action="store_true", help="初始化項目設置")
    parser.add_argument("--status", action="store_true", help="顯示項目狀態")
    parser.add_argument("--init-db", action="store_true", help="初始化數據庫")
    
    # 模塊運行
    parser.add_argument("--broad-topic", action="store_true", help="只運行話題提取模塊")
    parser.add_argument("--deep-sentiment", action="store_true", help="只運行情感爬取模塊")
    parser.add_argument("--complete", action="store_true", help="運行完整工作流程")
    
    # 參數配置
    parser.add_argument("--date", type=str, help="目標日期 (YYYY-MM-DD)，默認爲今天")
    parser.add_argument("--platforms", type=str, nargs='+', 
                       choices=['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu'],
                       help="指定爬取平臺")
    parser.add_argument("--keywords-count", type=int, default=100, help="話題提取的關鍵詞數量")
    parser.add_argument("--max-keywords", type=int, default=50, help="每個平臺最大關鍵詞數量")
    parser.add_argument("--max-notes", type=int, default=50, help="每個關鍵詞最大爬取內容數量")
    parser.add_argument("--test", action="store_true", help="測試模式（少量數據）")
    
    args = parser.parse_args()
    
    # 解析日期
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("錯誤：日期格式不正確，請使用 YYYY-MM-DD 格式")
            return
    
    # 創建MindSpider實例
    spider = MindSpider()
    
    try:
        # 顯示狀態
        if args.status:
            spider.show_status()
            return
        
        # 項目設置
        if args.setup:
            if spider.setup_project():
                logger.info("項目設置完成，可以開始使用MindSpider！")
            else:
                logger.error("項目設置失敗，請檢查配置和環境")
            return
        
        # 初始化數據庫
        if args.init_db:
            if spider.initialize_database():
                logger.info("數據庫初始化成功")
            else:
                logger.error("數據庫初始化失敗")
            return
        
        # 運行模塊
        if args.broad_topic:
            spider.run_broad_topic_extraction(target_date, args.keywords_count)
        elif args.deep_sentiment:
            spider.run_deep_sentiment_crawling(
                target_date, args.platforms, args.max_keywords, args.max_notes, args.test
            )
        elif args.complete:
            spider.run_complete_workflow(
                target_date, args.platforms, args.keywords_count, 
                args.max_keywords, args.max_notes, args.test
            )
        else:
            # 默認運行完整工作流程
            logger.info("運行完整MindSpider工作流程...")
            spider.run_complete_workflow(
                target_date, args.platforms, args.keywords_count,
                args.max_keywords, args.max_notes, args.test
            )
    
    except KeyboardInterrupt:
        logger.info("用戶中斷操作")
    except Exception as e:
        logger.exception(f"執行出錯: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindSpider AI爬蟲項目 - 數據庫管理工具
提供數據庫狀態查看、數據統計、清理等功能
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from urllib.parse import quote_plus

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    logger.error("錯誤: 無法導入config.py配置文件")
    sys.exit(1)

from config import settings

class DatabaseManager:
    def __init__(self):
        self.engine: Engine = None
        self.connect()
    
    def connect(self):
        """連接數據庫"""
        try:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect in ("postgresql", "postgres"):
                url = f"postgresql+psycopg://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            else:
                url = f"mysql+pymysql://{settings.DB_USER}:{quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            self.engine = create_engine(url, future=True)
            logger.info(f"成功連接到數據庫: {settings.DB_NAME}")
        except Exception as e:
            logger.error(f"數據庫連接失敗: {e}")
            sys.exit(1)
    
    def close(self):
        """關閉數據庫連接"""
        if self.engine:
            self.engine.dispose()
    
    def show_tables(self):
        """顯示所有表"""
        data_list_message = ""
        data_list_message += "\n" + "=" * 60
        data_list_message += "數據庫表列表"
        data_list_message += "=" * 60
        logger.info(data_list_message)
        
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        
        if not tables:
            logger.info("數據庫中沒有表")
            return
        
        # 分類顯示錶
        mindspider_tables = []
        mediacrawler_tables = []
        
        for table_name in tables:
            if table_name in ['daily_news', 'daily_topics', 'topic_news_relation', 'crawling_tasks']:
                mindspider_tables.append(table_name)
            else:
                mediacrawler_tables.append(table_name)
        
        data_list_message += "MindSpider核心表:"
        data_list_message += "\n"
        for table in mindspider_tables:
            with self.engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            data_list_message += f"  - {table:<25} ({count:>6} 條記錄)"
            data_list_message += "\n"
        
        data_list_message += "\nMediaCrawler平臺表:"
        data_list_message += "\n"
        for table in mediacrawler_tables:
            try:
                with self.engine.connect() as conn:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
                data_list_message += f"  - {table:<25} ({count:>6} 條記錄)"
                data_list_message += "\n"
            except:
                data_list_message += f"  - {table:<25} (查詢失敗)"
                data_list_message += "\n"
        logger.info(data_list_message)
    
    def show_statistics(self):
        """顯示數據統計"""
        data_statistics_message = ""
        data_statistics_message += "\n" + "=" * 60
        data_statistics_message += "數據統計"
        data_statistics_message += "=" * 60
        data_statistics_message += "\n"
        
        try:
            # 新聞統計
            with self.engine.connect() as conn:
                news_count = conn.execute(text("SELECT COUNT(*) FROM daily_news")).scalar_one()
                news_days = conn.execute(text("SELECT COUNT(DISTINCT crawl_date) FROM daily_news")).scalar_one()
                platforms = conn.execute(text("SELECT COUNT(DISTINCT source_platform) FROM daily_news")).scalar_one()
            
            data_statistics_message += "新聞數據:"
            data_statistics_message += "\n"
            data_statistics_message += f"  - 總新聞數: {news_count}"
            data_statistics_message += "\n"
            data_statistics_message += f"  - 覆蓋天數: {news_days}"
            data_statistics_message += "\n"
            data_statistics_message += f"  - 新聞平臺: {platforms}"
            data_statistics_message += "\n"
            # 話題統計
            with self.engine.connect() as conn:
                topic_count = conn.execute(text("SELECT COUNT(*) FROM daily_topics")).scalar_one()
                topic_days = conn.execute(text("SELECT COUNT(DISTINCT extract_date) FROM daily_topics")).scalar_one()
            
            data_statistics_message += "話題數據:"
            data_statistics_message += "\n"
            data_statistics_message += f"  - 總話題數: {topic_count}"
            data_statistics_message += "\n"
            data_statistics_message += f"  - 提取天數: {topic_days}"
            data_statistics_message += "\n"
            
            # 爬取任務統計
            with self.engine.connect() as conn:
                task_count = conn.execute(text("SELECT COUNT(*) FROM crawling_tasks")).scalar_one()
                task_status = conn.execute(text("SELECT task_status, COUNT(*) FROM crawling_tasks GROUP BY task_status")).all()
            
            data_statistics_message += "爬取任務:"
            data_statistics_message += "\n"
            data_statistics_message += f"  - 總任務數: {task_count}"
            data_statistics_message += "\n"
            for status, count in task_status:
                data_statistics_message += f"  - {status}: {count}"
                data_statistics_message += "\n"
            
            # 爬取內容統計
            data_statistics_message += "平臺內容統計:"
            data_statistics_message += "\n"
            platform_tables = {
                'xhs_note': '小紅書',
                'douyin_aweme': '抖音',
                'kuaishou_video': '快手',
                'bilibili_video': 'B站',
                'weibo_note': '微博',
                'tieba_note': '貼吧',
                'zhihu_content': '知乎'
            }
            
            for table, platform in platform_tables.items():
                try:
                    with self.engine.connect() as conn:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
                    data_statistics_message += f"  - {platform}: {count}"
                    data_statistics_message += "\n"
                except:
                    data_statistics_message += f"  - {platform}: 表不存在"
                    data_statistics_message += "\n"
            logger.info(data_statistics_message)
        except Exception as e:
            data_statistics_message += f"統計查詢失敗: {e}"
            data_statistics_message += "\n"
            logger.error(data_statistics_message)
    
    def show_recent_data(self, days=7):
        """顯示最近幾天的數據"""
        data_recent_message = ""
        data_recent_message += "\n" + "=" * 60
        data_recent_message += "最近" + str(days) + "天的數據"
        data_recent_message += "=" * 60
        
        from datetime import date, timedelta
        start_date = date.today() - timedelta(days=days)
        # 最近的新聞
        with self.engine.connect() as conn:
            news_data = conn.execute(
                text(
                    """
                    SELECT crawl_date, COUNT(*) as news_count, COUNT(DISTINCT source_platform) as platforms
                    FROM daily_news 
                    WHERE crawl_date >= :start_date
                    GROUP BY crawl_date 
                    ORDER BY crawl_date DESC
                    """
                ),
                {"start_date": start_date},
            ).all()
        if news_data:
            data_recent_message += "每日新聞統計:"
            data_recent_message += "\n"
            for date, count, platforms in news_data:
                data_recent_message += f"  {date}: {count} 條新聞, {platforms} 個平臺"
                data_recent_message += "\n"
        
        # 最近的話題
        with self.engine.connect() as conn:
            topic_data = conn.execute(
                text(
                    """
                    SELECT extract_date, COUNT(*) as topic_count
                    FROM daily_topics 
                    WHERE extract_date >= :start_date
                    GROUP BY extract_date 
                    ORDER BY extract_date DESC
                    """
                ),
                {"start_date": start_date},
            ).all()
        if topic_data:
            data_recent_message += "每日話題統計:"
            data_recent_message += "\n"
            for date, count in topic_data:
                data_recent_message += f"  {date}: {count} 個話題"
                data_recent_message += "\n"
        logger.info(data_recent_message)
    
    def cleanup_old_data(self, days=90, dry_run=True):
        """清理舊數據"""
        cleanup_message = ""
        cleanup_message += "\n" + "=" * 60
        cleanup_message += f"清理{days}天前的數據 ({'預覽模式' if dry_run else '執行模式'})"
        cleanup_message += "=" * 60
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 檢查要刪除的數據
        cleanup_queries = [
            ("daily_news", f"SELECT COUNT(*) FROM daily_news WHERE crawl_date < '{cutoff_date.date()}'"),
            ("daily_topics", f"SELECT COUNT(*) FROM daily_topics WHERE extract_date < '{cutoff_date.date()}'"),
            ("crawling_tasks", f"SELECT COUNT(*) FROM crawling_tasks WHERE scheduled_date < '{cutoff_date.date()}'")
        ]
        
        with self.engine.begin() as conn:
            for table, query in cleanup_queries:
                count = conn.execute(text(query)).scalar_one()
                if count > 0:
                    cleanup_message += f"  {table}: {count} 條記錄將被刪除"
                    cleanup_message += "\n"
                    if not dry_run:
                        delete_query = query.replace("SELECT COUNT(*)", "DELETE")
                        conn.execute(text(delete_query))
                        cleanup_message += f"    已刪除 {count} 條記錄"
                        cleanup_message += "\n"
                else:
                    cleanup_message += f"  {table}: 無需清理"
                    cleanup_message += "\n"
        
        if dry_run:
            cleanup_message += "\n這是預覽模式，沒有實際刪除數據。使用 --execute 參數執行實際清理。"
            cleanup_message += "\n"
        logger.info(cleanup_message)

def main():
    parser = argparse.ArgumentParser(description="MindSpider數據庫管理工具")
    parser.add_argument("--tables", action="store_true", help="顯示所有表")
    parser.add_argument("--stats", action="store_true", help="顯示數據統計")
    parser.add_argument("--recent", type=int, default=7, help="顯示最近N天的數據 (默認7天)")
    parser.add_argument("--cleanup", type=int, help="清理N天前的數據")
    parser.add_argument("--execute", action="store_true", help="執行實際清理操作")
    
    args = parser.parse_args()
    
    # 如果沒有參數，顯示所有信息
    if not any([args.tables, args.stats, args.recent != 7, args.cleanup]):
        args.tables = True
        args.stats = True
    
    db_manager = DatabaseManager()
    
    try:
        if args.tables:
            db_manager.show_tables()
        
        if args.stats:
            db_manager.show_statistics()
        
        if args.recent != 7 or not any([args.tables, args.stats, args.cleanup]):
            db_manager.show_recent_data(args.recent)
        
        if args.cleanup:
            db_manager.cleanup_old_data(args.cleanup, dry_run=not args.execute)
    
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()

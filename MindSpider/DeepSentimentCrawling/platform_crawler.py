#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling模塊 - 平臺爬蟲管理器
負責配置和調用MediaCrawler進行多平臺爬取
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
from loguru import logger

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("無法導入config.py配置文件")

class PlatformCrawler:
    """平臺爬蟲管理器"""
    
    def __init__(self):
        """初始化平臺爬蟲管理器"""
        self.mediacrawler_path = Path(__file__).parent / "MediaCrawler"
        self.supported_platforms = ['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu']
        self.crawl_stats = {}
        
        # 確保MediaCrawler目錄存在
        if not self.mediacrawler_path.exists():
            raise FileNotFoundError(f"MediaCrawler目錄不存在: {self.mediacrawler_path}")
        
        logger.info(f"初始化平臺爬蟲管理器，MediaCrawler路徑: {self.mediacrawler_path}")
    
    def configure_mediacrawler_db(self):
        """配置MediaCrawler使用我們的數據庫（MySQL或PostgreSQL）"""
        try:
            # 判斷數據庫類型
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            
            # 修改MediaCrawler的數據庫配置
            db_config_path = self.mediacrawler_path / "config" / "db_config.py"
            
            # 讀取原始配置
            with open(db_config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # PostgreSQL配置值：如果使用PostgreSQL則使用MindSpider配置，否則使用默認值或環境變量
            pg_password = config.settings.DB_PASSWORD if is_postgresql else "bettafish"
            pg_user = config.settings.DB_USER if is_postgresql else "bettafish"
            pg_host = config.settings.DB_HOST if is_postgresql else "127.0.0.1"
            pg_port = config.settings.DB_PORT if is_postgresql else 5432
            pg_db_name = config.settings.DB_NAME if is_postgresql else "bettafish"
            
            # 替換數據庫配置 - 使用MindSpider的數據庫配置
            new_config = f'''# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
# 1. 不得用於任何商業用途。  
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。  
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。  
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。   
# 5. 不得用於任何非法或不當的用途。
#   
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。  
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。  


import os

# mysql config - 使用MindSpider的數據庫配置
MYSQL_DB_PWD = "{config.settings.DB_PASSWORD}"
MYSQL_DB_USER = "{config.settings.DB_USER}"
MYSQL_DB_HOST = "{config.settings.DB_HOST}"
MYSQL_DB_PORT = {config.settings.DB_PORT}
MYSQL_DB_NAME = "{config.settings.DB_NAME}"

mysql_db_config = {{
    "user": MYSQL_DB_USER,
    "password": MYSQL_DB_PWD,
    "host": MYSQL_DB_HOST,
    "port": MYSQL_DB_PORT,
    "db_name": MYSQL_DB_NAME,
}}


# redis config
REDIS_DB_HOST = "127.0.0.1"  # your redis host
REDIS_DB_PWD = os.getenv("REDIS_DB_PWD", "123456")  # your redis password
REDIS_DB_PORT = os.getenv("REDIS_DB_PORT", 6379)  # your redis port
REDIS_DB_NUM = os.getenv("REDIS_DB_NUM", 0)  # your redis db num

# cache type
CACHE_TYPE_REDIS = "redis"
CACHE_TYPE_MEMORY = "memory"

# sqlite config
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "sqlite_tables.db")

sqlite_db_config = {{
    "db_path": SQLITE_DB_PATH
}}

# postgresql config - 使用MindSpider的數據庫配置（如果DB_DIALECT是postgresql）或環境變量
POSTGRESQL_DB_PWD = os.getenv("POSTGRESQL_DB_PWD", "{pg_password}")
POSTGRESQL_DB_USER = os.getenv("POSTGRESQL_DB_USER", "{pg_user}")
POSTGRESQL_DB_HOST = os.getenv("POSTGRESQL_DB_HOST", "{pg_host}")
POSTGRESQL_DB_PORT = os.getenv("POSTGRESQL_DB_PORT", "{pg_port}")
POSTGRESQL_DB_NAME = os.getenv("POSTGRESQL_DB_NAME", "{pg_db_name}")

postgresql_db_config = {{
    "user": POSTGRESQL_DB_USER,
    "password": POSTGRESQL_DB_PWD,
    "host": POSTGRESQL_DB_HOST,
    "port": POSTGRESQL_DB_PORT,
    "db_name": POSTGRESQL_DB_NAME,
}}

'''
            
            # 寫入新配置
            with open(db_config_path, 'w', encoding='utf-8') as f:
                f.write(new_config)
            
            db_type = "PostgreSQL" if is_postgresql else "MySQL"
            logger.info(f"已配置MediaCrawler使用MindSpider {db_type}數據庫")
            return True
            
        except Exception as e:
            logger.exception(f"配置MediaCrawler數據庫失敗: {e}")
            return False
    
    def create_base_config(self, platform: str, keywords: List[str], 
                          crawler_type: str = "search", max_notes: int = 50) -> bool:
        """
        創建MediaCrawler的基礎配置
        
        Args:
            platform: 平臺名稱
            keywords: 關鍵詞列表
            crawler_type: 爬取類型
            max_notes: 最大爬取數量
        
        Returns:
            是否配置成功
        """
        try:
            # 判斷數據庫類型，確定 SAVE_DATA_OPTION
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            save_data_option = "postgresql" if is_postgresql else "db"
            
            base_config_path = self.mediacrawler_path / "config" / "base_config.py"
            
            # 將關鍵詞列表轉換爲逗號分隔的字符串
            keywords_str = ",".join(keywords)
            
            # 讀取原始配置文件
            with open(base_config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修改關鍵配置項
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.startswith('PLATFORM = '):
                    new_lines.append(f'PLATFORM = "{platform}"  # 平臺，xhs | dy | ks | bili | wb | tieba | zhihu')
                elif line.startswith('KEYWORDS = '):
                    new_lines.append(f'KEYWORDS = "{keywords_str}"  # 關鍵詞搜索配置，以英文逗號分隔')
                elif line.startswith('CRAWLER_TYPE = '):
                    new_lines.append(f'CRAWLER_TYPE = "{crawler_type}"  # 爬取類型，search(關鍵詞搜索) | detail(帖子詳情)| creator(創作者主頁數據)')
                elif line.startswith('SAVE_DATA_OPTION = '):
                    new_lines.append(f'SAVE_DATA_OPTION = "{save_data_option}"  # csv or db or json or sqlite or postgresql')
                elif line.startswith('CRAWLER_MAX_NOTES_COUNT = '):
                    new_lines.append(f'CRAWLER_MAX_NOTES_COUNT = {max_notes}')
                elif line.startswith('ENABLE_GET_COMMENTS = '):
                    new_lines.append('ENABLE_GET_COMMENTS = True')
                elif line.startswith('CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = '):
                    new_lines.append('CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 20')
                elif line.startswith('HEADLESS = '):
                    new_lines.append('HEADLESS = True')  # 使用無頭模式
                else:
                    new_lines.append(line)
            
            # 寫入新配置
            with open(base_config_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            logger.info(f"已配置 {platform} 平臺，爬取類型: {crawler_type}，關鍵詞數量: {len(keywords)}，最大爬取數量: {max_notes}，保存數據方式: {save_data_option}")
            return True
            
        except Exception as e:
            logger.exception(f"創建基礎配置失敗: {e}")
            return False
    
    def run_crawler(self, platform: str, keywords: List[str], 
                   login_type: str = "qrcode", max_notes: int = 50) -> Dict:
        """
        運行爬蟲
        
        Args:
            platform: 平臺名稱
            keywords: 關鍵詞列表
            login_type: 登錄方式
            max_notes: 最大爬取數量
        
        Returns:
            爬取結果統計
        """
        if platform not in self.supported_platforms:
            raise ValueError(f"不支持的平臺: {platform}")
        
        if not keywords:
            raise ValueError("關鍵詞列表不能爲空")
        
        start_message = f"\n開始爬取平臺: {platform}"
        start_message += f"\n關鍵詞: {keywords[:5]}{'...' if len(keywords) > 5 else ''} (共{len(keywords)}個)"
        logger.info(start_message)
        
        start_time = datetime.now()
        
        try:
            # 配置數據庫
            if not self.configure_mediacrawler_db():
                return {"success": False, "error": "數據庫配置失敗"}
            
            # 創建基礎配置
            if not self.create_base_config(platform, keywords, "search", max_notes):
                return {"success": False, "error": "基礎配置創建失敗"}
            
            # 判斷數據庫類型，確定 save_data_option
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            save_data_option = "postgresql" if is_postgresql else "db"
            
            # 構建命令
            cmd = [
                sys.executable, "main.py",
                "--platform", platform,
                "--lt", login_type,
                "--type", "search",
                "--save_data_option", save_data_option
            ]
            
            logger.info(f"執行命令: {' '.join(cmd)}")
            
            # 切換到MediaCrawler目錄並執行
            result = subprocess.run(
                cmd,
                cwd=self.mediacrawler_path,
                timeout=3600  # 60分鐘超時
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 創建統計信息
            crawl_stats = {
                "platform": platform,
                "keywords_count": len(keywords),
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "notes_count": 0,
                "comments_count": 0,
                "errors_count": 0
            }
            
            # 保存統計信息
            self.crawl_stats[platform] = crawl_stats
            
            if result.returncode == 0:
                logger.info(f"✅ {platform} 爬取完成，耗時: {duration:.1f}秒")
            else:
                logger.error(f"❌ {platform} 爬取失敗，返回碼: {result.returncode}")
            
            return crawl_stats
            
        except subprocess.TimeoutExpired:
            logger.exception(f"❌ {platform} 爬取超時")
            return {"success": False, "error": "爬取超時", "platform": platform}
        except Exception as e:
            logger.exception(f"❌ {platform} 爬取異常: {e}")
            return {"success": False, "error": str(e), "platform": platform}
    
    def _parse_crawl_output(self, output_lines: List[str], error_lines: List[str]) -> Dict:
        """解析爬取輸出，提取統計信息"""
        stats = {
            "notes_count": 0,
            "comments_count": 0,
            "errors_count": 0,
            "login_required": False
        }
        
        # 解析輸出行
        for line in output_lines:
            if "條筆記" in line or "條內容" in line:
                try:
                    # 提取數字
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        stats["notes_count"] = int(numbers[0])
                except:
                    pass
            elif "條評論" in line:
                try:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        stats["comments_count"] = int(numbers[0])
                except:
                    pass
            elif "登錄" in line or "掃碼" in line:
                stats["login_required"] = True
        
        # 解析錯誤行
        for line in error_lines:
            if "error" in line.lower() or "異常" in line:
                stats["errors_count"] += 1
        
        return stats
    
    def run_multi_platform_crawl_by_keywords(self, keywords: List[str], platforms: List[str],
                                            login_type: str = "qrcode", max_notes_per_keyword: int = 50) -> Dict:
        """
        基於關鍵詞的多平臺爬取 - 每個關鍵詞在所有平臺上都進行爬取
        
        Args:
            keywords: 關鍵詞列表
            platforms: 平臺列表
            login_type: 登錄方式
            max_notes_per_keyword: 每個關鍵詞在每個平臺的最大爬取數量
        
        Returns:
            總體爬取統計
        """
        
        start_message = f"\n🚀 開始全平臺關鍵詞爬取"
        start_message += f"\n   關鍵詞數量: {len(keywords)}"
        start_message += f"\n   平臺數量: {len(platforms)}"
        start_message += f"\n   登錄方式: {login_type}"
        start_message += f"\n   每個關鍵詞在每個平臺的最大爬取數量: {max_notes_per_keyword}"
        start_message += f"\n   總爬取任務: {len(keywords)} × {len(platforms)} = {len(keywords) * len(platforms)}"
        logger.info(start_message)
        
        total_stats = {
            "total_keywords": len(keywords),
            "total_platforms": len(platforms),
            "total_tasks": len(keywords) * len(platforms),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_notes": 0,
            "total_comments": 0,
            "keyword_results": {},
            "platform_summary": {}
        }
        
        # 初始化平臺統計
        for platform in platforms:
            total_stats["platform_summary"][platform] = {
                "successful_keywords": 0,
                "failed_keywords": 0,
                "total_notes": 0,
                "total_comments": 0
            }
        
        # 對每個平臺一次性爬取所有關鍵詞
        for platform in platforms:
            logger.info(f"\n📝 在 {platform} 平臺爬取所有關鍵詞")
            logger.info(f"   關鍵詞: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
            
            try:
                # 一次性傳遞所有關鍵詞給平臺
                result = self.run_crawler(platform, keywords, login_type, max_notes_per_keyword)
                
                if result.get("success"):
                    total_stats["successful_tasks"] += len(keywords)
                    total_stats["platform_summary"][platform]["successful_keywords"] = len(keywords)
                    
                    notes_count = result.get("notes_count", 0)
                    comments_count = result.get("comments_count", 0)
                    
                    total_stats["total_notes"] += notes_count
                    total_stats["total_comments"] += comments_count
                    total_stats["platform_summary"][platform]["total_notes"] = notes_count
                    total_stats["platform_summary"][platform]["total_comments"] = comments_count
                    
                    # 爲每個關鍵詞記錄結果
                    for keyword in keywords:
                        if keyword not in total_stats["keyword_results"]:
                            total_stats["keyword_results"][keyword] = {}
                        total_stats["keyword_results"][keyword][platform] = result
                    
                    logger.info(f"   ✅ 成功: {notes_count} 條內容, {comments_count} 條評論")
                else:
                    total_stats["failed_tasks"] += len(keywords)
                    total_stats["platform_summary"][platform]["failed_keywords"] = len(keywords)
                    
                    # 爲每個關鍵詞記錄失敗結果
                    for keyword in keywords:
                        if keyword not in total_stats["keyword_results"]:
                            total_stats["keyword_results"][keyword] = {}
                        total_stats["keyword_results"][keyword][platform] = result
                    
                    logger.error(f"   ❌ 失敗: {result.get('error', '未知錯誤')}")
            
            except Exception as e:
                total_stats["failed_tasks"] += len(keywords)
                total_stats["platform_summary"][platform]["failed_keywords"] = len(keywords)
                error_result = {"success": False, "error": str(e)}
                
                # 爲每個關鍵詞記錄異常結果
                for keyword in keywords:
                    if keyword not in total_stats["keyword_results"]:
                        total_stats["keyword_results"][keyword] = {}
                    total_stats["keyword_results"][keyword][platform] = error_result
                
                logger.error(f"   ❌ 異常: {e}")
        
        # 打印詳細統計
        finish_message = f"\n📊 全平臺關鍵詞爬取完成!"
        finish_message += f"\n   總任務: {total_stats['total_tasks']}"
        finish_message += f"\n   成功: {total_stats['successful_tasks']}"
        finish_message += f"\n   失敗: {total_stats['failed_tasks']}"
        finish_message += f"\n   成功率: {total_stats['successful_tasks']/total_stats['total_tasks']*100:.1f}%"
        finish_message += f"\n   總內容: {total_stats['total_notes']} 條"
        finish_message += f"\n   總評論: {total_stats['total_comments']} 條"
        logger.info(finish_message)
        
        platform_summary_message = f"\n� 各平臺統計:"
        for platform, stats in total_stats["platform_summary"].items():
            success_rate = stats["successful_keywords"] / len(keywords) * 100 if keywords else 0
            platform_summary_message += f"\n   {platform}: {stats['successful_keywords']}/{len(keywords)} 關鍵詞成功 ({success_rate:.1f}%), "
            platform_summary_message += f"{stats['total_notes']} 條內容"
        logger.info(platform_summary_message)
        
        return total_stats
    
    def get_crawl_statistics(self) -> Dict:
        """獲取爬取統計信息"""
        return {
            "platforms_crawled": list(self.crawl_stats.keys()),
            "total_platforms": len(self.crawl_stats),
            "detailed_stats": self.crawl_stats
        }
    
    def save_crawl_log(self, log_path: str = None):
        """保存爬取日誌"""
        if not log_path:
            log_path = f"crawl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.crawl_stats, f, ensure_ascii=False, indent=2)
            logger.info(f"爬取日誌已保存到: {log_path}")
        except Exception as e:
            logger.exception(f"保存爬取日誌失敗: {e}")

if __name__ == "__main__":
    # 測試平臺爬蟲管理器
    crawler = PlatformCrawler()
    
    # 測試配置
    test_keywords = ["科技", "AI", "編程"]
    result = crawler.run_crawler("xhs", test_keywords, max_notes=5)
    
    logger.info(f"測試結果: {result}")
    logger.info("平臺爬蟲管理器測試完成！")

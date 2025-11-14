-- MindSpider AI爬蟲項目 - 數據庫表結構
-- 基於MediaCrawler表結構擴展，添加BroadTopicExtraction模塊所需表

-- ===============================
-- BroadTopicExtraction 模塊表結構
-- ===============================

-- ----------------------------
-- Table structure for daily_news
-- 每日新聞表：存儲get_today_news.py獲取的熱點新聞
-- ----------------------------
DROP TABLE IF EXISTS `daily_news`;
CREATE TABLE `daily_news` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `news_id` varchar(128) NOT NULL COMMENT '新聞唯一ID',
    `source_platform` varchar(32) NOT NULL COMMENT '新聞源平臺(weibo|zhihu|bilibili|toutiao|douyin等)',
    `title` varchar(500) NOT NULL COMMENT '新聞標題',
    `url` varchar(512) DEFAULT NULL COMMENT '新聞鏈接',
    `description` text COMMENT '新聞描述或摘要',
    `extra_info` text COMMENT '額外信息(JSON格式存儲)',
    `crawl_date` date NOT NULL COMMENT '爬取日期',
    `rank_position` int DEFAULT NULL COMMENT '在熱榜中的排名位置',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_daily_news_unique` (`news_id`, `source_platform`, `crawl_date`),
    KEY `idx_daily_news_date` (`crawl_date`),
    KEY `idx_daily_news_platform` (`source_platform`),
    KEY `idx_daily_news_rank` (`rank_position`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='每日熱點新聞表';

-- ----------------------------
-- Table structure for daily_topics
-- 每日話題表：存儲TopicGPT提取的話題信息
-- ----------------------------
DROP TABLE IF EXISTS `daily_topics`;
CREATE TABLE `daily_topics` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `topic_id` varchar(64) NOT NULL COMMENT '話題唯一ID',
    `topic_name` varchar(255) NOT NULL COMMENT '話題名稱',
    `topic_description` text COMMENT '話題描述',
    `keywords` text COMMENT '話題關鍵詞(JSON格式存儲)',
    `extract_date` date NOT NULL COMMENT '話題提取日期',
    `relevance_score` float DEFAULT NULL COMMENT '話題相關性得分',
    `news_count` int DEFAULT 0 COMMENT '關聯的新聞數量',
    `processing_status` varchar(16) DEFAULT 'pending' COMMENT '處理狀態(pending|processing|completed|failed)',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_daily_topics_unique` (`topic_id`, `extract_date`),
    KEY `idx_daily_topics_date` (`extract_date`),
    KEY `idx_daily_topics_status` (`processing_status`),
    KEY `idx_daily_topics_score` (`relevance_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='每日提取話題表';

-- ----------------------------
-- Table structure for topic_news_relation
-- 話題新聞關聯表：記錄話題和新聞的關聯關系
-- ----------------------------
DROP TABLE IF EXISTS `topic_news_relation`;
CREATE TABLE `topic_news_relation` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `topic_id` varchar(64) NOT NULL COMMENT '話題ID',
    `news_id` varchar(128) NOT NULL COMMENT '新聞ID',
    `relation_score` float DEFAULT NULL COMMENT '關聯度得分',
    `extract_date` date NOT NULL COMMENT '關聯提取日期',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_topic_news_unique` (`topic_id`, `news_id`, `extract_date`),
    KEY `idx_topic_news_topic` (`topic_id`),
    KEY `idx_topic_news_news` (`news_id`),
    KEY `idx_topic_news_date` (`extract_date`),
    FOREIGN KEY (`topic_id`) REFERENCES `daily_topics`(`topic_id`) ON DELETE CASCADE,
    FOREIGN KEY (`news_id`) REFERENCES `daily_news`(`news_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='話題新聞關聯表';

-- ----------------------------
-- Table structure for crawling_tasks
-- 爬取任務表：記錄基於話題的平臺爬取任務
-- ----------------------------
DROP TABLE IF EXISTS `crawling_tasks`;
CREATE TABLE `crawling_tasks` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `task_id` varchar(64) NOT NULL COMMENT '任務唯一ID',
    `topic_id` varchar(64) NOT NULL COMMENT '關聯的話題ID',
    `platform` varchar(32) NOT NULL COMMENT '目標平臺(xhs|dy|ks|bili|wb|tieba|zhihu)',
    `search_keywords` text NOT NULL COMMENT '搜索關鍵詞(JSON格式存儲)',
    `task_status` varchar(16) DEFAULT 'pending' COMMENT '任務狀態(pending|running|completed|failed|paused)',
    `start_time` bigint DEFAULT NULL COMMENT '任務開始時間戳',
    `end_time` bigint DEFAULT NULL COMMENT '任務結束時間戳',
    `total_crawled` int DEFAULT 0 COMMENT '已爬取內容數量',
    `success_count` int DEFAULT 0 COMMENT '成功爬取數量',
    `error_count` int DEFAULT 0 COMMENT '錯誤數量',
    `error_message` text COMMENT '錯誤信息',
    `config_params` text COMMENT '爬取配置參數(JSON格式)',
    `scheduled_date` date NOT NULL COMMENT '計劃執行日期',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_crawling_tasks_unique` (`task_id`),
    KEY `idx_crawling_tasks_topic` (`topic_id`),
    KEY `idx_crawling_tasks_platform` (`platform`),
    KEY `idx_crawling_tasks_status` (`task_status`),
    KEY `idx_crawling_tasks_date` (`scheduled_date`),
    FOREIGN KEY (`topic_id`) REFERENCES `daily_topics`(`topic_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='爬取任務表';

-- ===============================
-- MediaCrawler表結構擴展字段
-- ===============================

-- 爲MediaCrawler現有表添加話題關聯字段，支持MindSpider功能
-- 注意：這些字段是可選的，不影響MediaCrawler原有功能

-- 爲小紅書筆記表添加話題關聯字段
ALTER TABLE `xhs_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- 爲抖音視頻表添加話題關聯字段
ALTER TABLE `douyin_aweme` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- 爲快手視頻表添加話題關聯字段
ALTER TABLE `kuaishou_video` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- 爲B站視頻表添加話題關聯字段
ALTER TABLE `bilibili_video` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- 爲微博帖子表添加話題關聯字段
ALTER TABLE `weibo_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- 爲貼吧帖子表添加話題關聯字段
ALTER TABLE `tieba_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- 爲知乎內容表添加話題關聯字段
ALTER TABLE `zhihu_content` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT '關聯的話題ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT '關聯的爬取任務ID';

-- ===============================
-- 創建視圖用於數據分析
-- ===============================

-- 話題爬取統計視圖
CREATE OR REPLACE VIEW `v_topic_crawling_stats` AS
SELECT 
    dt.topic_id,
    dt.topic_name,
    dt.extract_date,
    dt.processing_status,
    COUNT(DISTINCT ct.task_id) as total_tasks,
    SUM(CASE WHEN ct.task_status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
    SUM(CASE WHEN ct.task_status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
    SUM(ct.total_crawled) as total_content_crawled,
    SUM(ct.success_count) as total_success_count,
    SUM(ct.error_count) as total_error_count
FROM daily_topics dt
LEFT JOIN crawling_tasks ct ON dt.topic_id = ct.topic_id
GROUP BY dt.topic_id, dt.topic_name, dt.extract_date, dt.processing_status;

-- 每日數據統計視圖
CREATE OR REPLACE VIEW `v_daily_summary` AS
SELECT 
    crawl_date,
    COUNT(DISTINCT news_id) as total_news,
    COUNT(DISTINCT source_platform) as platforms_covered,
    (SELECT COUNT(*) FROM daily_topics WHERE extract_date = dn.crawl_date) as topics_extracted,
    (SELECT COUNT(*) FROM crawling_tasks WHERE scheduled_date = dn.crawl_date) as tasks_created
FROM daily_news dn
GROUP BY crawl_date
ORDER BY crawl_date DESC;

-- ===============================
-- 初始化索引優化
-- ===============================

-- 爲關聯查詢優化添加複合索引
CREATE INDEX `idx_topic_date_status` ON `daily_topics` (`extract_date`, `processing_status`);
CREATE INDEX `idx_task_topic_platform` ON `crawling_tasks` (`topic_id`, `platform`, `task_status`);
CREATE INDEX `idx_news_date_platform` ON `daily_news` (`crawl_date`, `source_platform`);

-- ===============================
-- 數據庫配置優化建議
-- ===============================

-- 設置合適的字符集和排序規則
-- ALTER DATABASE mindspider CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 建議的數據保留策略（可選）
-- 可以根據需要創建事件調度器來清理歷史數據
-- 例如：刪除90天前的新聞數據，保留話題和爬取結果數據

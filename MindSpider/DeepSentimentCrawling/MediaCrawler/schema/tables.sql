-- ----------------------------
-- Table structure for bilibili_video
-- ----------------------------
DROP TABLE IF EXISTS `bilibili_video`;
CREATE TABLE `bilibili_video`
(
    `id`               int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`          varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`         varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`           varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `add_ts`           bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`   bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `video_id`         varchar(64) NOT NULL COMMENT '視頻ID',
    `video_type`       varchar(16) NOT NULL COMMENT '視頻類型',
    `title`            varchar(500) DEFAULT NULL COMMENT '視頻標題',
    `desc`             longtext COMMENT '視頻描述',
    `create_time`      bigint      NOT NULL COMMENT '視頻發佈時間戳',
    `liked_count`      varchar(16)  DEFAULT NULL COMMENT '視頻點贊數',
    `disliked_count`   varchar(16) DEFAULT NULL COMMENT '視頻點踩數',
    `video_play_count` varchar(16)  DEFAULT NULL COMMENT '視頻播放數量',
    `video_favorite_count` varchar(16) DEFAULT NULL COMMENT '視頻收藏數量',
    `video_share_count` varchar(16) DEFAULT NULL COMMENT '視頻分享數量',
    `video_coin_count` varchar(16) DEFAULT NULL COMMENT '視頻投幣數量',
    `video_danmaku`    varchar(16)  DEFAULT NULL COMMENT '視頻彈幕數量',
    `video_comment`    varchar(16)  DEFAULT NULL COMMENT '視頻評論數量',
    `video_url`        varchar(512) DEFAULT NULL COMMENT '視頻詳情URL',
    `video_cover_url`  varchar(512) DEFAULT NULL COMMENT '視頻封面圖 URL',
    PRIMARY KEY (`id`),
    KEY                `idx_bilibili_vi_video_i_31c36e` (`video_id`),
    KEY                `idx_bilibili_vi_create__73e0ec` (`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='B站視頻';

-- ----------------------------
-- Table structure for bilibili_video_comment
-- ----------------------------
DROP TABLE IF EXISTS `bilibili_video_comment`;
CREATE TABLE `bilibili_video_comment`
(
    `id`                int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`           varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`          varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `sex`               varchar(64) DEFAULT NULL COMMENT '用戶性別',
    `sign`              text DEFAULT NULL COMMENT '用戶簽名',
    `avatar`            varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `add_ts`            bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`    bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `comment_id`        varchar(64) NOT NULL COMMENT '評論ID',
    `video_id`          varchar(64) NOT NULL COMMENT '視頻ID',
    `content`           longtext COMMENT '評論內容',
    `create_time`       bigint      NOT NULL COMMENT '評論時間戳',
    `sub_comment_count` varchar(16) NOT NULL COMMENT '評論回覆數',
    PRIMARY KEY (`id`),
    KEY                 `idx_bilibili_vi_comment_41c34e` (`comment_id`),
    KEY                 `idx_bilibili_vi_video_i_f22873` (`video_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='B 站視頻評論';

-- ----------------------------
-- Table structure for bilibili_up_info
-- ----------------------------
DROP TABLE IF EXISTS `bilibili_up_info`;
CREATE TABLE `bilibili_up_info`
(
    `id`             int    NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`        varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`       varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `sex`            varchar(64) DEFAULT NULL COMMENT '用戶性別',
    `sign`           text DEFAULT NULL COMMENT '用戶簽名',
    `avatar`         varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `add_ts`         bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    `total_fans`     bigint       DEFAULT NULL COMMENT '粉絲數',
    `total_liked`    bigint       DEFAULT NULL COMMENT '總獲贊數',
    `user_rank`      int          DEFAULT NULL COMMENT '用戶等級',
    `is_official`    int          DEFAULT NULL COMMENT '是否官號',
    PRIMARY KEY (`id`),
    KEY              `idx_bilibili_vi_user_123456` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='B 站UP主信息';

-- ----------------------------
-- Table structure for bilibili_contact_info
-- ----------------------------
DROP TABLE IF EXISTS `bilibili_contact_info`;
CREATE TABLE `bilibili_contact_info`
(
    `id`             int    NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `up_id`          varchar(64)  DEFAULT NULL COMMENT 'up主ID',
    `fan_id`         varchar(64)  DEFAULT NULL COMMENT '粉絲ID',
    `up_name`        varchar(64)  DEFAULT NULL COMMENT 'up主暱稱',
    `fan_name`       varchar(64)  DEFAULT NULL COMMENT '粉絲暱稱',
    `up_sign`        longtext     DEFAULT NULL COMMENT 'up主簽名',
    `fan_sign`       longtext     DEFAULT NULL COMMENT '粉絲簽名',
    `up_avatar`      varchar(255) DEFAULT NULL COMMENT 'up主頭像地址',
    `fan_avatar`     varchar(255) DEFAULT NULL COMMENT '粉絲頭像地址',
    `add_ts`         bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    KEY              `idx_bilibili_contact_info_up_id` (`up_id`),
    KEY              `idx_bilibili_contact_info_fan_id` (`fan_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='B 站聯繫人信息';

-- ----------------------------
-- Table structure for bilibili_up_dynamic
-- ----------------------------
DROP TABLE IF EXISTS `bilibili_up_dynamic`;
CREATE TABLE `bilibili_up_dynamic`
(
    `id`             int    NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `dynamic_id`     varchar(64)  DEFAULT NULL COMMENT '動態ID',
    `user_id`        varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `user_name`      varchar(64)  DEFAULT NULL COMMENT '用戶名',
    `text`           longtext     DEFAULT NULL COMMENT '動態文本',
    `type`           varchar(64)  DEFAULT NULL COMMENT '動態類型',
    `pub_ts`         bigint DEFAULT NULL COMMENT '動態發佈時間',
    `total_comments` bigint       DEFAULT NULL COMMENT '評論數',
    `total_forwards` bigint       DEFAULT NULL COMMENT '轉發數',
    `total_liked`    bigint       DEFAULT NULL COMMENT '點贊數',
    `add_ts`         bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    KEY              `idx_bilibili_up_dynamic_dynamic_id` (`dynamic_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='B 站up主動態信息';

-- ----------------------------
-- Table structure for douyin_aweme
-- ----------------------------
DROP TABLE IF EXISTS `douyin_aweme`;
CREATE TABLE `douyin_aweme`
(
    `id`              int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`         varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `sec_uid`         varchar(128) DEFAULT NULL COMMENT '用戶sec_uid',
    `short_user_id`   varchar(64)  DEFAULT NULL COMMENT '用戶短ID',
    `user_unique_id`  varchar(64)  DEFAULT NULL COMMENT '用戶唯一ID',
    `nickname`        varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`          varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `user_signature`  varchar(500) DEFAULT NULL COMMENT '用戶簽名',
    `ip_location`     varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`          bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`  bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `aweme_id`        varchar(64) NOT NULL COMMENT '視頻ID',
    `aweme_type`      varchar(16) NOT NULL COMMENT '視頻類型',
    `title`           varchar(1024) DEFAULT NULL COMMENT '視頻標題',
    `desc`            longtext COMMENT '視頻描述',
    `create_time`     bigint      NOT NULL COMMENT '視頻發佈時間戳',
    `liked_count`     varchar(16)  DEFAULT NULL COMMENT '視頻點贊數',
    `comment_count`   varchar(16)  DEFAULT NULL COMMENT '視頻評論數',
    `share_count`     varchar(16)  DEFAULT NULL COMMENT '視頻分享數',
    `collected_count` varchar(16)  DEFAULT NULL COMMENT '視頻收藏數',
    `aweme_url`       varchar(255) DEFAULT NULL COMMENT '視頻詳情頁URL',
    `cover_url`       varchar(500) DEFAULT NULL COMMENT '視頻封面圖URL',
    `video_download_url`       longtext COMMENT '視頻下載地址',
    `music_download_url`       longtext COMMENT '音樂下載地址',
    `note_download_url`        longtext COMMENT '筆記下載地址',
    PRIMARY KEY (`id`),
    KEY               `idx_douyin_awem_aweme_i_6f7bc6` (`aweme_id`),
    KEY               `idx_douyin_awem_create__299dfe` (`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='抖音視頻';

-- ----------------------------
-- Table structure for douyin_aweme_comment
-- ----------------------------
DROP TABLE IF EXISTS `douyin_aweme_comment`;
CREATE TABLE `douyin_aweme_comment`
(
    `id`                int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`           varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `sec_uid`           varchar(128) DEFAULT NULL COMMENT '用戶sec_uid',
    `short_user_id`     varchar(64)  DEFAULT NULL COMMENT '用戶短ID',
    `user_unique_id`    varchar(64)  DEFAULT NULL COMMENT '用戶唯一ID',
    `nickname`          varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`            varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `user_signature`    varchar(500) DEFAULT NULL COMMENT '用戶簽名',
    `ip_location`       varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`            bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`    bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `comment_id`        varchar(64) NOT NULL COMMENT '評論ID',
    `aweme_id`          varchar(64) NOT NULL COMMENT '視頻ID',
    `content`           longtext COMMENT '評論內容',
    `create_time`       bigint      NOT NULL COMMENT '評論時間戳',
    `sub_comment_count` varchar(16) NOT NULL COMMENT '評論回覆數',
    PRIMARY KEY (`id`),
    KEY                 `idx_douyin_awem_comment_fcd7e4` (`comment_id`),
    KEY                 `idx_douyin_awem_aweme_i_c50049` (`aweme_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='抖音視頻評論';

-- ----------------------------
-- Table structure for dy_creator
-- ----------------------------
DROP TABLE IF EXISTS `dy_creator`;
CREATE TABLE `dy_creator`
(
    `id`             int          NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`        varchar(128) NOT NULL COMMENT '用戶ID',
    `nickname`       varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`         varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `ip_location`    varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`         bigint       NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint       NOT NULL COMMENT '記錄最後修改時間戳',
    `desc`           longtext COMMENT '用戶描述',
    `gender`         varchar(1)   DEFAULT NULL COMMENT '性別',
    `follows`        varchar(16)  DEFAULT NULL COMMENT '關注數',
    `fans`           varchar(16)  DEFAULT NULL COMMENT '粉絲數',
    `interaction`    varchar(16)  DEFAULT NULL COMMENT '獲贊數',
    `videos_count`   varchar(16)  DEFAULT NULL COMMENT '作品數',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='抖音博主信息';

-- ----------------------------
-- Table structure for kuaishou_video
-- ----------------------------
DROP TABLE IF EXISTS `kuaishou_video`;
CREATE TABLE `kuaishou_video`
(
    `id`              int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`         varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`        varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`          varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `add_ts`          bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`  bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `video_id`        varchar(64) NOT NULL COMMENT '視頻ID',
    `video_type`      varchar(16) NOT NULL COMMENT '視頻類型',
    `title`           varchar(500) DEFAULT NULL COMMENT '視頻標題',
    `desc`            longtext COMMENT '視頻描述',
    `create_time`     bigint      NOT NULL COMMENT '視頻發佈時間戳',
    `liked_count`     varchar(16)  DEFAULT NULL COMMENT '視頻點贊數',
    `viewd_count`     varchar(16)  DEFAULT NULL COMMENT '視頻瀏覽數量',
    `video_url`       varchar(512) DEFAULT NULL COMMENT '視頻詳情URL',
    `video_cover_url` varchar(512) DEFAULT NULL COMMENT '視頻封面圖 URL',
    `video_play_url`  varchar(512) DEFAULT NULL COMMENT '視頻播放 URL',
    PRIMARY KEY (`id`),
    KEY               `idx_kuaishou_vi_video_i_c5c6a6` (`video_id`),
    KEY               `idx_kuaishou_vi_create__a10dee` (`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='快手視頻';

-- ----------------------------
-- Table structure for kuaishou_video_comment
-- ----------------------------
DROP TABLE IF EXISTS `kuaishou_video_comment`;
CREATE TABLE `kuaishou_video_comment`
(
    `id`                int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`           varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`          varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`            varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `add_ts`            bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`    bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `comment_id`        varchar(64) NOT NULL COMMENT '評論ID',
    `video_id`          varchar(64) NOT NULL COMMENT '視頻ID',
    `content`           longtext COMMENT '評論內容',
    `create_time`       bigint      NOT NULL COMMENT '評論時間戳',
    `sub_comment_count` varchar(16) NOT NULL COMMENT '評論回覆數',
    PRIMARY KEY (`id`),
    KEY                 `idx_kuaishou_vi_comment_ed48fa` (`comment_id`),
    KEY                 `idx_kuaishou_vi_video_i_e50914` (`video_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='快手視頻評論';


-- ----------------------------
-- Table structure for weibo_note
-- ----------------------------
DROP TABLE IF EXISTS `weibo_note`;
CREATE TABLE `weibo_note`
(
    `id`               int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`          varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`         varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`           varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `gender`           varchar(12)  DEFAULT NULL COMMENT '用戶性別',
    `profile_url`      varchar(255) DEFAULT NULL COMMENT '用戶主頁地址',
    `ip_location`      varchar(32)  DEFAULT '發佈微博的地理信息',
    `add_ts`           bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`   bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `note_id`          varchar(64) NOT NULL COMMENT '帖子ID',
    `content`          longtext COMMENT '帖子正文內容',
    `create_time`      bigint      NOT NULL COMMENT '帖子發佈時間戳',
    `create_date_time` varchar(32) NOT NULL COMMENT '帖子發佈日期時間',
    `liked_count`      varchar(16)  DEFAULT NULL COMMENT '帖子點贊數',
    `comments_count`   varchar(16)  DEFAULT NULL COMMENT '帖子評論數量',
    `shared_count`     varchar(16)  DEFAULT NULL COMMENT '帖子轉發數量',
    `note_url`         varchar(512) DEFAULT NULL COMMENT '帖子詳情URL',
    PRIMARY KEY (`id`),
    KEY                `idx_weibo_note_note_id_f95b1a` (`note_id`),
    KEY                `idx_weibo_note_create__692709` (`create_time`),
    KEY                `idx_weibo_note_create__d05ed2` (`create_date_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='微博帖子';

-- ----------------------------
-- Table structure for weibo_note_comment
-- ----------------------------
DROP TABLE IF EXISTS `weibo_note_comment`;
CREATE TABLE `weibo_note_comment`
(
    `id`                 int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`            varchar(64)  DEFAULT NULL COMMENT '用戶ID',
    `nickname`           varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`             varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `gender`             varchar(12)  DEFAULT NULL COMMENT '用戶性別',
    `profile_url`        varchar(255) DEFAULT NULL COMMENT '用戶主頁地址',
    `ip_location`        varchar(32)  DEFAULT '發佈微博的地理信息',
    `add_ts`             bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`     bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `comment_id`         varchar(64) NOT NULL COMMENT '評論ID',
    `note_id`            varchar(64) NOT NULL COMMENT '帖子ID',
    `content`            longtext COMMENT '評論內容',
    `create_time`        bigint      NOT NULL COMMENT '評論時間戳',
    `create_date_time`   varchar(32) NOT NULL COMMENT '評論日期時間',
    `comment_like_count` varchar(16) NOT NULL COMMENT '評論點贊數量',
    `sub_comment_count`  varchar(16) NOT NULL COMMENT '評論回覆數',
    PRIMARY KEY (`id`),
    KEY                  `idx_weibo_note__comment_c7611c` (`comment_id`),
    KEY                  `idx_weibo_note__note_id_24f108` (`note_id`),
    KEY                  `idx_weibo_note__create__667fe3` (`create_date_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='微博帖子評論';

-- ----------------------------
-- Table structure for xhs_creator
-- ----------------------------
DROP TABLE IF EXISTS `xhs_creator`;
CREATE TABLE `xhs_creator`
(
    `id`             int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`        varchar(64) NOT NULL COMMENT '用戶ID',
    `nickname`       varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`         varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `ip_location`    varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`         bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `desc`           longtext COMMENT '用戶描述',
    `gender`         varchar(1)   DEFAULT NULL COMMENT '性別',
    `follows`        varchar(16)  DEFAULT NULL COMMENT '關注數',
    `fans`           varchar(16)  DEFAULT NULL COMMENT '粉絲數',
    `interaction`    varchar(16)  DEFAULT NULL COMMENT '獲贊和收藏數',
    `tag_list`       longtext COMMENT '標籤列表',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='小紅書博主';

-- ----------------------------
-- Table structure for xhs_note
-- ----------------------------
DROP TABLE IF EXISTS `xhs_note`;
CREATE TABLE `xhs_note`
(
    `id`               int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`          varchar(64) NOT NULL COMMENT '用戶ID',
    `nickname`         varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`           varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `ip_location`      varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`           bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`   bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `note_id`          varchar(64) NOT NULL COMMENT '筆記ID',
    `type`             varchar(16)  DEFAULT NULL COMMENT '筆記類型(normal | video)',
    `title`            varchar(255) DEFAULT NULL COMMENT '筆記標題',
    `desc`             longtext COMMENT '筆記描述',
    `video_url`        longtext COMMENT '視頻地址',
    `time`             bigint      NOT NULL COMMENT '筆記發佈時間戳',
    `last_update_time` bigint      NOT NULL COMMENT '筆記最後更新時間戳',
    `liked_count`      varchar(16)  DEFAULT NULL COMMENT '筆記點贊數',
    `collected_count`  varchar(16)  DEFAULT NULL COMMENT '筆記收藏數',
    `comment_count`    varchar(16)  DEFAULT NULL COMMENT '筆記評論數',
    `share_count`      varchar(16)  DEFAULT NULL COMMENT '筆記分享數',
    `image_list`       longtext COMMENT '筆記封面圖片列表',
    `tag_list`         longtext COMMENT '標籤列表',
    `note_url`         varchar(255) DEFAULT NULL COMMENT '筆記詳情頁的URL',
    PRIMARY KEY (`id`),
    KEY                `idx_xhs_note_note_id_209457` (`note_id`),
    KEY                `idx_xhs_note_time_eaa910` (`time`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='小紅書筆記';

-- ----------------------------
-- Table structure for xhs_note_comment
-- ----------------------------
DROP TABLE IF EXISTS `xhs_note_comment`;
CREATE TABLE `xhs_note_comment`
(
    `id`                int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`           varchar(64) NOT NULL COMMENT '用戶ID',
    `nickname`          varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`            varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `ip_location`       varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`            bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`    bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `comment_id`        varchar(64) NOT NULL COMMENT '評論ID',
    `create_time`       bigint      NOT NULL COMMENT '評論時間戳',
    `note_id`           varchar(64) NOT NULL COMMENT '筆記ID',
    `content`           longtext    NOT NULL COMMENT '評論內容',
    `sub_comment_count` int         NOT NULL COMMENT '子評論數量',
    `pictures`          varchar(512) DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY                 `idx_xhs_note_co_comment_8e8349` (`comment_id`),
    KEY                 `idx_xhs_note_co_create__204f8d` (`create_time`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='小紅書筆記評論';

-- ----------------------------
-- alter table xhs_note_comment to support parent_comment_id
-- ----------------------------
ALTER TABLE `xhs_note_comment`
    ADD COLUMN `parent_comment_id` VARCHAR(64) DEFAULT NULL COMMENT '父評論ID';

ALTER TABLE `douyin_aweme_comment`
    ADD COLUMN `parent_comment_id` VARCHAR(64) DEFAULT NULL COMMENT '父評論ID';

ALTER TABLE `bilibili_video_comment`
    ADD COLUMN `parent_comment_id` VARCHAR(64) DEFAULT NULL COMMENT '父評論ID';

ALTER TABLE `weibo_note_comment`
    ADD COLUMN `parent_comment_id` VARCHAR(64) DEFAULT NULL COMMENT '父評論ID';


DROP TABLE IF EXISTS `tieba_note`;
CREATE TABLE tieba_note
(
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    note_id           VARCHAR(644) NOT NULL COMMENT '帖子ID',
    title             VARCHAR(255) NOT NULL COMMENT '帖子標題',
    `desc`            TEXT COMMENT '帖子描述',
    note_url          VARCHAR(255) NOT NULL COMMENT '帖子鏈接',
    publish_time      VARCHAR(255) NOT NULL COMMENT '發佈時間',
    user_link         VARCHAR(255) DEFAULT '' COMMENT '用戶主頁鏈接',
    user_nickname     VARCHAR(255) DEFAULT '' COMMENT '用戶暱稱',
    user_avatar       VARCHAR(255) DEFAULT '' COMMENT '用戶頭像地址',
    tieba_id          VARCHAR(255) DEFAULT '' COMMENT '貼吧ID',
    tieba_name        VARCHAR(255) NOT NULL COMMENT '貼吧名稱',
    tieba_link        VARCHAR(255) NOT NULL COMMENT '貼吧鏈接',
    total_replay_num  INT          DEFAULT 0 COMMENT '帖子回覆總數',
    total_replay_page INT          DEFAULT 0 COMMENT '帖子回覆總頁數',
    ip_location       VARCHAR(255) DEFAULT '' COMMENT 'IP地理位置',
    add_ts            BIGINT       NOT NULL COMMENT '添加時間戳',
    last_modify_ts    BIGINT       NOT NULL COMMENT '最後修改時間戳',
    KEY               `idx_tieba_note_note_id` (`note_id`),
    KEY               `idx_tieba_note_publish_time` (`publish_time`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='貼吧帖子表';

DROP TABLE IF EXISTS `tieba_comment`;
CREATE TABLE tieba_comment
(
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    comment_id        VARCHAR(255) NOT NULL COMMENT '評論ID',
    parent_comment_id VARCHAR(255) DEFAULT '' COMMENT '父評論ID',
    content           TEXT         NOT NULL COMMENT '評論內容',
    user_link         VARCHAR(255) DEFAULT '' COMMENT '用戶主頁鏈接',
    user_nickname     VARCHAR(255) DEFAULT '' COMMENT '用戶暱稱',
    user_avatar       VARCHAR(255) DEFAULT '' COMMENT '用戶頭像地址',
    tieba_id          VARCHAR(255) DEFAULT '' COMMENT '貼吧ID',
    tieba_name        VARCHAR(255) NOT NULL COMMENT '貼吧名稱',
    tieba_link        VARCHAR(255) NOT NULL COMMENT '貼吧鏈接',
    publish_time      VARCHAR(255) DEFAULT '' COMMENT '發佈時間',
    ip_location       VARCHAR(255) DEFAULT '' COMMENT 'IP地理位置',
    sub_comment_count INT          DEFAULT 0 COMMENT '子評論數',
    note_id           VARCHAR(255) NOT NULL COMMENT '帖子ID',
    note_url          VARCHAR(255) NOT NULL COMMENT '帖子鏈接',
    add_ts            BIGINT       NOT NULL COMMENT '添加時間戳',
    last_modify_ts    BIGINT       NOT NULL COMMENT '最後修改時間戳',
    KEY               `idx_tieba_comment_comment_id` (`note_id`),
    KEY               `idx_tieba_comment_note_id` (`note_id`),
    KEY               `idx_tieba_comment_publish_time` (`publish_time`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='貼吧評論表';

alter table bilibili_video add column `source_keyword` varchar(255) default '' comment '搜索來源關鍵字';
alter table douyin_aweme add column `source_keyword` varchar(255) default '' comment '搜索來源關鍵字';
alter table kuaishou_video add column `source_keyword` varchar(255) default '' comment '搜索來源關鍵字';
alter table weibo_note add column `source_keyword` varchar(255) default '' comment '搜索來源關鍵字';
alter table xhs_note add column `source_keyword` varchar(255) default '' comment '搜索來源關鍵字';
alter table tieba_note add column `source_keyword` varchar(255) default '' comment '搜索來源關鍵字';


DROP TABLE IF EXISTS `weibo_creator`;
CREATE TABLE `weibo_creator`
(
    `id`             int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`        varchar(64) NOT NULL COMMENT '用戶ID',
    `nickname`       varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`         varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `ip_location`    varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`         bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `desc`           longtext COMMENT '用戶描述',
    `gender`         varchar(2)   DEFAULT NULL COMMENT '性別',
    `follows`        varchar(16)  DEFAULT NULL COMMENT '關注數',
    `fans`           varchar(16)  DEFAULT NULL COMMENT '粉絲數',
    `tag_list`       longtext COMMENT '標籤列表',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='微博博主';


ALTER TABLE `xhs_note_comment`
    ADD COLUMN `like_count` VARCHAR(64) DEFAULT NULL COMMENT '評論點贊數量';


DROP TABLE IF EXISTS `tieba_creator`;
CREATE TABLE `tieba_creator`
(
    `id`                    int         NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id`               varchar(64) NOT NULL COMMENT '用戶ID',
    `user_name`             varchar(64) NOT NULL COMMENT '用戶名',
    `nickname`              varchar(64)  DEFAULT NULL COMMENT '用戶暱稱',
    `avatar`                varchar(255) DEFAULT NULL COMMENT '用戶頭像地址',
    `ip_location`           varchar(255) DEFAULT NULL COMMENT '評論時的IP地址',
    `add_ts`                bigint      NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts`        bigint      NOT NULL COMMENT '記錄最後修改時間戳',
    `gender`                varchar(2)   DEFAULT NULL COMMENT '性別',
    `follows`               varchar(16)  DEFAULT NULL COMMENT '關注數',
    `fans`                  varchar(16)  DEFAULT NULL COMMENT '粉絲數',
    `registration_duration` varchar(16)  DEFAULT NULL COMMENT '吧齡',
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='貼吧創作者';

DROP TABLE IF EXISTS `zhihu_content`;
CREATE TABLE `zhihu_content` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `content_id` varchar(64) NOT NULL COMMENT '內容ID',
    `content_type` varchar(16) NOT NULL COMMENT '內容類型(article | answer | zvideo)',
    `content_text` longtext COMMENT '內容文本, 如果是視頻類型這裏爲空',
    `content_url` varchar(255) NOT NULL COMMENT '內容落地鏈接',
    `question_id` varchar(64) DEFAULT NULL COMMENT '問題ID, type爲answer時有值',
    `title` varchar(255) NOT NULL COMMENT '內容標題',
    `desc` longtext COMMENT '內容描述',
    `created_time` varchar(32) NOT NULL COMMENT '創建時間',
    `updated_time` varchar(32) NOT NULL COMMENT '更新時間',
    `voteup_count` int NOT NULL DEFAULT '0' COMMENT '贊同人數',
    `comment_count` int NOT NULL DEFAULT '0' COMMENT '評論數量',
    `source_keyword` varchar(64) DEFAULT NULL COMMENT '來源關鍵詞',
    `user_id` varchar(64) NOT NULL COMMENT '用戶ID',
    `user_link` varchar(255) NOT NULL COMMENT '用戶主頁鏈接',
    `user_nickname` varchar(64) NOT NULL COMMENT '用戶暱稱',
    `user_avatar` varchar(255) NOT NULL COMMENT '用戶頭像地址',
    `user_url_token` varchar(255) NOT NULL COMMENT '用戶url_token',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    KEY `idx_zhihu_content_content_id` (`content_id`),
    KEY `idx_zhihu_content_created_time` (`created_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='知乎內容（回答、文章、視頻）';


DROP TABLE IF EXISTS `zhihu_comment`;
CREATE TABLE `zhihu_comment` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `comment_id` varchar(64) NOT NULL COMMENT '評論ID',
    `parent_comment_id` varchar(64) DEFAULT NULL COMMENT '父評論ID',
    `content` text NOT NULL COMMENT '評論內容',
    `publish_time` varchar(32) NOT NULL COMMENT '發佈時間',
    `ip_location` varchar(64) DEFAULT NULL COMMENT 'IP地理位置',
    `sub_comment_count` int NOT NULL DEFAULT '0' COMMENT '子評論數',
    `like_count` int NOT NULL DEFAULT '0' COMMENT '點贊數',
    `dislike_count` int NOT NULL DEFAULT '0' COMMENT '踩數',
    `content_id` varchar(64) NOT NULL COMMENT '內容ID',
    `content_type` varchar(16) NOT NULL COMMENT '內容類型(article | answer | zvideo)',
    `user_id` varchar(64) NOT NULL COMMENT '用戶ID',
    `user_link` varchar(255) NOT NULL COMMENT '用戶主頁鏈接',
    `user_nickname` varchar(64) NOT NULL COMMENT '用戶暱稱',
    `user_avatar` varchar(255) NOT NULL COMMENT '用戶頭像地址',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    KEY `idx_zhihu_comment_comment_id` (`comment_id`),
    KEY `idx_zhihu_comment_content_id` (`content_id`),
    KEY `idx_zhihu_comment_publish_time` (`publish_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='知乎評論';

DROP TABLE IF EXISTS `zhihu_creator`;
CREATE TABLE `zhihu_creator` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    `user_id` varchar(64) NOT NULL COMMENT '用戶ID',
    `user_link` varchar(255) NOT NULL COMMENT '用戶主頁鏈接',
    `user_nickname` varchar(64) NOT NULL COMMENT '用戶暱稱',
    `user_avatar` varchar(255) NOT NULL COMMENT '用戶頭像地址',
    `url_token` varchar(64) NOT NULL COMMENT '用戶URL Token',
    `gender` varchar(16) DEFAULT NULL COMMENT '用戶性別',
    `ip_location` varchar(64) DEFAULT NULL COMMENT 'IP地理位置',
    `follows` int NOT NULL DEFAULT 0 COMMENT '關注數',
    `fans` int NOT NULL DEFAULT 0 COMMENT '粉絲數',
    `anwser_count` int NOT NULL DEFAULT 0 COMMENT '回答數',
    `video_count` int NOT NULL DEFAULT 0 COMMENT '視頻數',
    `question_count` int NOT NULL DEFAULT 0 COMMENT '問題數',
    `article_count` int NOT NULL DEFAULT 0 COMMENT '文章數',
    `column_count` int NOT NULL DEFAULT 0 COMMENT '專欄數',
    `get_voteup_count` int NOT NULL DEFAULT 0 COMMENT '獲得的贊同數',
    `add_ts` bigint NOT NULL COMMENT '記錄添加時間戳',
    `last_modify_ts` bigint NOT NULL COMMENT '記錄最後修改時間戳',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_zhihu_creator_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='知乎創作者';


-- add column `like_count` to douyin_aweme_comment
alter table douyin_aweme_comment add column `like_count` varchar(255) NOT NULL DEFAULT '0' COMMENT '點贊數';

alter table xhs_note add column xsec_token varchar(50) default null comment '簽名算法';
alter table douyin_aweme_comment add column `pictures` varchar(500) NOT NULL DEFAULT '' COMMENT '評論圖片列表';
alter table bilibili_video_comment add column `like_count` varchar(255) NOT NULL DEFAULT '0' COMMENT '點贊數';

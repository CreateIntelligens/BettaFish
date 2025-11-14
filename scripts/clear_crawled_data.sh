#!/bin/bash
# 清空所有爬蟲收集的資料

echo "正在清空爬蟲資料..."

docker exec bettafish-db psql -U bettafish -d bettafish << EOF
-- 清空所有內容表
TRUNCATE TABLE
  weibo_note,
  xhs_note,
  bilibili_video,
  kuaishou_video,
  tieba_note,
  zhihu_content,
  daily_news
CASCADE;

-- 清空所有評論表
TRUNCATE TABLE
  weibo_note_comment,
  xhs_note_comment,
  bilibili_video_comment,
  kuaishou_video_comment,
  tieba_comment,
  zhihu_comment
CASCADE;

-- 清空創作者/UP主資訊表
TRUNCATE TABLE
  weibo_creator,
  xhs_creator,
  bilibili_up_info,
  bilibili_contact_info,
  dy_creator
CASCADE;

-- 清空其他相關表
TRUNCATE TABLE
  topic_news_relation,
  bilibili_up_dynamic
CASCADE;

EOF

echo "✓ 爬蟲資料已清空"

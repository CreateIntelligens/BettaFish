# -*- coding: utf-8 -*-
# @Author  : persist-1<persist1@126.com>
# @Time    : 2025/9/8 00:02
# @Desc    : 用於將orm映射模型（database/models.py）與兩種數據庫實際結構進行對比，並進行更新操作（連接數據庫->結構比對->差異報告->交互式同步）
# @Tips    : 該腳本需要安裝依賴'pymysql==1.1.0'

import os
import sys
from sqlalchemy import create_engine, inspect as sqlalchemy_inspect
from sqlalchemy.schema import MetaData

# 將項目根目錄添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.db_config import mysql_db_config, sqlite_db_config
from database.models import Base

def get_mysql_engine():
    """創建並返回一個MySQL數據庫引擎"""
    conn_str = f"mysql+pymysql://{mysql_db_config['user']}:{mysql_db_config['password']}@{mysql_db_config['host']}:{mysql_db_config['port']}/{mysql_db_config['db_name']}"
    return create_engine(conn_str)

def get_sqlite_engine():
    """創建並返回一個SQLite數據庫引擎"""
    conn_str = f"sqlite:///{sqlite_db_config['db_path']}"
    return create_engine(conn_str)

def get_db_schema(engine):
    """獲取數據庫的當前表結構"""
    inspector = sqlalchemy_inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        columns = {}
        for column in inspector.get_columns(table_name):
            columns[column['name']] = str(column['type'])
        schema[table_name] = columns
    return schema

def get_orm_schema():
    """獲取ORM模型的表結構"""
    schema = {}
    for table_name, table in Base.metadata.tables.items():
        columns = {}
        for column in table.columns:
            columns[column.name] = str(column.type)
        schema[table_name] = columns
    return schema

def compare_schemas(db_schema, orm_schema):
    """比較數據庫結構和ORM模型結構，返回差異"""
    db_tables = set(db_schema.keys())
    orm_tables = set(orm_schema.keys())

    added_tables = orm_tables - db_tables
    deleted_tables = db_tables - orm_tables
    common_tables = db_tables.intersection(orm_tables)

    changed_tables = {}

    for table in common_tables:
        db_cols = set(db_schema[table].keys())
        orm_cols = set(orm_schema[table].keys())
        added_cols = orm_cols - db_cols
        deleted_cols = db_cols - orm_cols
        
        modified_cols = {}
        for col in db_cols.intersection(orm_cols):
            if db_schema[table][col] != orm_schema[table][col]:
                modified_cols[col] = (db_schema[table][col], orm_schema[table][col])

        if added_cols or deleted_cols or modified_cols:
            changed_tables[table] = {
                "added": list(added_cols),
                "deleted": list(deleted_cols),
                "modified": modified_cols
            }

    return {
        "added_tables": list(added_tables),
        "deleted_tables": list(deleted_tables),
        "changed_tables": changed_tables
    }

def print_diff(db_name, diff):
    """打印差異報告"""
    print(f"--- {db_name} 數據庫結構差異報告 ---")
    if not any(diff.values()):
        print("數據庫結構與ORM模型一致，無需同步。")
        return

    if diff.get("added_tables"):
        print("\n[+] 新增的表:")
        for table in diff["added_tables"]:
            print(f"  - {table}")

    if diff.get("deleted_tables"):
        print("\n[-] 刪除的表:")
        for table in diff["deleted_tables"]:
            print(f"  - {table}")

    if diff.get("changed_tables"):
        print("\n[*] 變動的表:")
        for table, changes in diff["changed_tables"].items():
            print(f"  - {table}:")
            if changes.get("added"):
                print("    [+] 新增字段:", ", ".join(changes["added"]))
            if changes.get("deleted"):
                print("    [-] 刪除字段:", ", ".join(changes["deleted"]))
            if changes.get("modified"):
                print("    [*] 修改字段:")
                for col, types in changes["modified"].items():
                    print(f"      - {col}: {types[0]} -> {types[1]}")
    print("--- 報告結束 ---")


def sync_database(engine, diff):
    """將ORM模型同步到數據庫"""
    metadata = Base.metadata
    
    # Alembic的上下文配置
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    conn = engine.connect()
    ctx = MigrationContext.configure(conn)
    op = Operations(ctx)

    # 處理刪除的表
    for table_name in diff['deleted_tables']:
        op.drop_table(table_name)
        print(f"已刪除表: {table_name}")

    # 處理新增的表
    for table_name in diff['added_tables']:
        table = metadata.tables.get(table_name)
        if table is not None:
            table.create(engine)
            print(f"已創建表: {table_name}")

    # 處理字段變更
    for table_name, changes in diff['changed_tables'].items():
        # 刪除字段
        for col_name in changes['deleted']:
            op.drop_column(table_name, col_name)
            print(f"在表 {table_name} 中已刪除字段: {col_name}")
        # 新增字段
        for col_name in changes['added']:
            table = metadata.tables.get(table_name)
            column = table.columns.get(col_name)
            if column is not None:
                op.add_column(table_name, column)
                print(f"在表 {table_name} 中已新增字段: {col_name}")

        # 修改字段
        for col_name, types in changes['modified'].items():
            table = metadata.tables.get(table_name)
            if table is not None:
                column = table.columns.get(col_name)
                if column is not None:
                    op.alter_column(table_name, col_name, type_=column.type)
                    print(f"在表 {table_name} 中已修改字段: {col_name} (類型變爲 {column.type})")


def main():
    """主函數"""
    orm_schema = get_orm_schema()

    # 處理 MySQL
    try:
        mysql_engine = get_mysql_engine()
        mysql_schema = get_db_schema(mysql_engine)
        mysql_diff = compare_schemas(mysql_schema, orm_schema)
        print_diff("MySQL", mysql_diff)
        if any(mysql_diff.values()):
            choice = input(">>> 需要人工確認：是否要將ORM模型同步到MySQL數據庫? (y/N): ")
            if choice.lower() == 'y':
                sync_database(mysql_engine, mysql_diff)
                print("MySQL數據庫同步完成。")
    except Exception as e:
        print(f"處理MySQL時出錯: {e}")


    # 處理 SQLite
    try:
        sqlite_engine = get_sqlite_engine()
        sqlite_schema = get_db_schema(sqlite_engine)
        sqlite_diff = compare_schemas(sqlite_schema, orm_schema)
        print_diff("SQLite", sqlite_diff)
        if any(sqlite_diff.values()):
            choice = input(">>> 需要人工確認：是否要將ORM模型同步到SQLite數據庫? (y/N): ")
            if choice.lower() == 'y':
                # 注意：SQLite不支持ALTER COLUMN來修改字段類型，這裏簡化處理
                print("警告：SQLite的字段修改支持有限，此腳本不會執行修改字段類型的操作。")
                sync_database(sqlite_engine, sqlite_diff)
                print("SQLite數據庫同步完成。")
    except Exception as e:
        print(f"處理SQLite時出錯: {e}")


if __name__ == "__main__":
    main()

######################### Feedback example #########################
# [*] 變動的表:
#   - kuaishou_video:
#     [*] 修改字段:
#       - user_id: TEXT -> VARCHAR(64)
#   - xhs_note_comment:
#     [*] 修改字段:
#       - comment_id: BIGINT -> VARCHAR(255)
#   - zhihu_content:
#     [*] 修改字段:
#       - created_time: BIGINT -> VARCHAR(32)
#       - content_id: BIGINT -> VARCHAR(64)
#   - zhihu_creator:
#     [*] 修改字段:
#       - user_id: INTEGER -> VARCHAR(64)
#   - tieba_note:
#     [*] 修改字段:
#       - publish_time: BIGINT -> VARCHAR(255)
#       - tieba_id: INTEGER -> VARCHAR(255)
#       - note_id: BIGINT -> VARCHAR(644)
# --- 報告結束 ---
# >>> 需要人工確認：是否要將ORM模型同步到MySQL數據庫? (y/N): y
# 在表 kuaishou_video 中已修改字段: user_id (類型變爲 VARCHAR(64))
# 在表 xhs_note_comment 中已修改字段: comment_id (類型變爲 VARCHAR(255))
# 在表 zhihu_content 中已修改字段: created_time (類型變爲 VARCHAR(32))
# 在表 zhihu_content 中已修改字段: content_id (類型變爲 VARCHAR(64))
# 在表 zhihu_creator 中已修改字段: user_id (類型變爲 VARCHAR(64))
# 在表 tieba_note 中已修改字段: publish_time (類型變爲 VARCHAR(255))
# 在表 tieba_note 中已修改字段: tieba_id (類型變爲 VARCHAR(255))
# 在表 tieba_note 中已修改字段: note_id (類型變爲 VARCHAR(644))
# MySQL數據庫同步完成。
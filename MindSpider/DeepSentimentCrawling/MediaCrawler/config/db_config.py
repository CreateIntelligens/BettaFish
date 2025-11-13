# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：  
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
MYSQL_DB_PWD = "bettafish"
MYSQL_DB_USER = "bettafish"
MYSQL_DB_HOST = "127.0.0.1"
MYSQL_DB_PORT = 5444
MYSQL_DB_NAME = "bettafish"

mysql_db_config = {
    "user": MYSQL_DB_USER,
    "password": MYSQL_DB_PWD,
    "host": MYSQL_DB_HOST,
    "port": MYSQL_DB_PORT,
    "db_name": MYSQL_DB_NAME,
}


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

sqlite_db_config = {
    "db_path": SQLITE_DB_PATH
}

# postgresql config - 使用MindSpider的數據庫配置（如果DB_DIALECT是postgresql）或環境變量
POSTGRESQL_DB_PWD = os.getenv("POSTGRESQL_DB_PWD", "bettafish")
POSTGRESQL_DB_USER = os.getenv("POSTGRESQL_DB_USER", "bettafish")
POSTGRESQL_DB_HOST = os.getenv("POSTGRESQL_DB_HOST", "127.0.0.1")
POSTGRESQL_DB_PORT = os.getenv("POSTGRESQL_DB_PORT", "5444")
POSTGRESQL_DB_NAME = os.getenv("POSTGRESQL_DB_NAME", "bettafish")

postgresql_db_config = {
    "user": POSTGRESQL_DB_USER,
    "password": POSTGRESQL_DB_PWD,
    "host": POSTGRESQL_DB_HOST,
    "port": POSTGRESQL_DB_PORT,
    "db_name": POSTGRESQL_DB_NAME,
}


# 聲明：本代碼僅供學習和研究目的使用。使用者應遵守以下原則：
# 1. 不得用於任何商業用途。
# 2. 使用時應遵守目標平臺的使用條款和robots.txt規則。
# 3. 不得進行大規模爬取或對平臺造成運營幹擾。
# 4. 應合理控制請求頻率，避免給目標平臺帶來不必要的負擔。
# 5. 不得用於任何非法或不當的用途。
#
# 詳細許可條款請參閱項目根目錄下的LICENSE文件。
# 使用本代碼即表示您同意遵守上述原則和LICENSE中的所有條款。

# 基礎配置
PLATFORM = "bili"  # 平臺，xhs | dy | ks | bili | wb | tieba | zhihu
KEYWORDS = "電影鬼滅之刃,親屬想侵吞3姐妹亡父賠償款,網警斬斷侵害未成年人網絡黑色產業鏈,2007年後出生的人不能在馬爾代夫吸菸,沈月,是公主也是自己的騎士,以軍虐囚視頻,唐朝詭事錄,廣州地鐵回應APP乘車碼頻繁彈窗廣告,全紅嬋的減肥計劃精確到克"  # 關鍵詞搜索配置，以英文逗號分隔
LOGIN_TYPE = "qrcode"  # qrcode or phone or cookie
COOKIES = ""
CRAWLER_TYPE = "search"  # 爬取類型，search(關鍵詞搜索) | detail(帖子詳情)| creator(創作者主頁數據)

# 是否開啓 IP 代理
ENABLE_IP_PROXY = False

# 代理IP池數量
IP_PROXY_POOL_COUNT = 2

# 代理IP提供商名稱
IP_PROXY_PROVIDER_NAME = "kuaidaili"  # kuaidaili | wandouhttp

# 設置爲True不會打開瀏覽器（無頭瀏覽器）
# 設置False會打開一個瀏覽器
# 小紅書如果一直掃碼登錄不通過，打開瀏覽器手動過一下滑動驗證碼
# 抖音如果一直提示失敗，打開瀏覽器看下是否掃碼登錄之後出現了手機號驗證，如果出現了手動過一下再試。
HEADLESS = True

# 是否保存登錄狀態
SAVE_LOGIN_STATE = True

# ==================== CDP (Chrome DevTools Protocol) 配置 ====================
# 是否啓用CDP模式 - 使用用戶現有的Chrome/Edge瀏覽器進行爬取，提供更好的反檢測能力
# 啓用後將自動檢測並啓動用戶的Chrome/Edge瀏覽器，通過CDP協議進行控制
# 這種方式使用真實的瀏覽器環境，包括用戶的擴展、Cookie和設置，大大降低被檢測的風險
ENABLE_CDP_MODE = True

# CDP調試端口，用於與瀏覽器通信
# 如果端口被佔用，系統會自動嘗試下一個可用端口
CDP_DEBUG_PORT = 9222

# 自定義瀏覽器路徑（可選）
# 如果爲空，系統會自動檢測Chrome/Edge的安裝路徑
# Windows示例: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
# macOS示例: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CUSTOM_BROWSER_PATH = ""

# CDP模式下是否啓用無頭模式
# 注意：即使設置爲True，某些反檢測功能在無頭模式下可能效果不佳
CDP_HEADLESS = False

# 瀏覽器啓動超時時間（秒）
BROWSER_LAUNCH_TIMEOUT = 30

# 是否在程序結束時自動關閉瀏覽器
# 設置爲False可以保持瀏覽器運行，便於調試
AUTO_CLOSE_BROWSER = True

# 數據保存類型選項配置,支持五種類型：csv、db、json、sqlite、postgresql, 最好保存到DB，有排重的功能。
SAVE_DATA_OPTION = "postgresql"  # csv or db or json or sqlite or postgresql

# 用戶瀏覽器緩存的瀏覽器文件配置
USER_DATA_DIR = "%s_user_data_dir"  # %s will be replaced by platform name

# 爬取開始頁數 默認從第一頁開始
START_PAGE = 1

# 爬取視頻/帖子的數量控制
CRAWLER_MAX_NOTES_COUNT = 5

# 併發爬蟲數量控制
MAX_CONCURRENCY_NUM = 1

# 是否開啓爬媒體模式（包含圖片或視頻資源），默認不開啓爬媒體
ENABLE_GET_MEIDAS = False

# 是否開啓爬評論模式, 默認開啓爬評論
ENABLE_GET_COMMENTS = True

# 爬取一級評論的數量控制(單視頻/帖子)
CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 20

# 是否開啓爬二級評論模式, 默認不開啓爬二級評論
# 老版本項目使用了 db, 則需參考 schema/tables.sql line 287 增加表字段
ENABLE_GET_SUB_COMMENTS = False

# 詞雲相關
# 是否開啓生成評論詞雲圖
ENABLE_GET_WORDCLOUD = False
# 自定義詞語及其分組
# 添加規則：xx:yy 其中xx爲自定義添加的詞組，yy爲將xx該詞組分到的組名。
CUSTOM_WORDS = {
    "零幾": "年份",  # 將“零幾”識別爲一個整體
    "高頻詞": "專業術語",  # 示例自定義詞
}

# 停用(禁用)詞文件路徑
STOP_WORDS_FILE = "./docs/hit_stopwords.txt"

# 中文字體文件路徑
FONT_PATH = "./docs/STZHONGS.TTF"

# 爬取間隔時間
CRAWLER_MAX_SLEEP_SEC = 2

from .bilibili_config import *
from .xhs_config import *
from .dy_config import *
from .ks_config import *
from .weibo_config import *
from .tieba_config import *
from .zhihu_config import *

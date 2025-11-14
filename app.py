"""
Flask主應用 - 統一管理三個Streamlit應用
"""

import os
import sys
import subprocess
import time
import threading
from datetime import datetime
from queue import Queue
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import atexit
import requests
from loguru import logger
import importlib
from pathlib import Path
from MindSpider.main import MindSpider

from config import settings as app_settings

# 導入ReportEngine
try:
    from ReportEngine.flask_interface import report_bp, initialize_report_engine
    REPORT_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"ReportEngine導入失敗: {e}")
    REPORT_ENGINE_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Dedicated-to-creating-a-concise-and-versatile-public-opinion-analysis-platform'
socketio = SocketIO(app, cors_allowed_origins="*")
logger.info(f"系統時區設定為: {app_settings.TIMEZONE}")

# 註冊ReportEngine Blueprint
if REPORT_ENGINE_AVAILABLE:
    app.register_blueprint(report_bp, url_prefix='/api/report')
    logger.info("ReportEngine接口已註冊")
else:
    logger.info("ReportEngine不可用，跳過接口註冊")

# 設置UTF-8編碼環境
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 創建日誌目錄
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

CONFIG_MODULE_NAME = 'config'
CONFIG_FILE_PATH = Path(__file__).resolve().parent / 'config.py'
CONFIG_KEYS = [
    'HOST',
    'PORT',
    'DB_DIALECT',
    'DB_HOST',
    'DB_PORT',
    'DB_USER',
    'DB_PASSWORD',
    'DB_NAME',
    'DB_CHARSET',
    'INSIGHT_ENGINE_API_KEY',
    'INSIGHT_ENGINE_BASE_URL',
    'INSIGHT_ENGINE_MODEL_NAME',
    'MEDIA_ENGINE_API_KEY',
    'MEDIA_ENGINE_BASE_URL',
    'MEDIA_ENGINE_MODEL_NAME',
    'QUERY_ENGINE_API_KEY',
    'QUERY_ENGINE_BASE_URL',
    'QUERY_ENGINE_MODEL_NAME',
    'REPORT_ENGINE_API_KEY',
    'REPORT_ENGINE_BASE_URL',
    'REPORT_ENGINE_MODEL_NAME',
    'FORUM_HOST_API_KEY',
    'FORUM_HOST_BASE_URL',
    'FORUM_HOST_MODEL_NAME',
    'KEYWORD_OPTIMIZER_API_KEY',
    'KEYWORD_OPTIMIZER_BASE_URL',
    'KEYWORD_OPTIMIZER_MODEL_NAME',
    'TAVILY_API_KEY',
    'BOCHA_WEB_SEARCH_API_KEY',
    'TIMEZONE'
]


def _load_config_module():
    """Load or reload the config module to ensure latest values are available."""
    importlib.invalidate_caches()
    module = sys.modules.get(CONFIG_MODULE_NAME)
    try:
        if module is None:
            module = importlib.import_module(CONFIG_MODULE_NAME)
        else:
            module = importlib.reload(module)
    except ModuleNotFoundError:
        return None
    return module


def read_config_values():
    """Return the current configuration values that are exposed to the frontend."""
    try:
        # 重新載入配置以獲取最新的 Settings 實例
        from config import reload_settings, settings
        reload_settings()


        values = {}
        for key in CONFIG_KEYS:
            # 從 Pydantic Settings 實例讀取值
            value = getattr(settings, key, None)
            # Convert to string for uniform handling on the frontend.
            if value is None:
                values[key] = ''
            else:
                values[key] = str(value)
        return values
    except Exception as exc:
        logger.exception(f"讀取配置失敗: {exc}")
        return {}


def _serialize_config_value(value):
    """Serialize Python values back to a config.py assignment-friendly string."""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return 'None'

    value_str = str(value)
    escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def write_config_values(updates):
    """Persist configuration updates to .env file (Pydantic Settings source)."""
    from pathlib import Path
    
    # 確定 .env 文件路徑（與 config.py 中的邏輯一致）
    project_root = Path(__file__).resolve().parent
    cwd_env = Path.cwd() / ".env"
    env_file_path = cwd_env if cwd_env.exists() else (project_root / ".env")
    
    # 讀取現有的 .env 文件內容
    env_lines = []
    env_key_indices = {}  # 記錄每個鍵在文件中的索引位置
    if env_file_path.exists():
        env_lines = env_file_path.read_text(encoding='utf-8').splitlines()
        # 提取已存在的鍵及其索引
        for i, line in enumerate(env_lines):
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                if '=' in line_stripped:
                    key = line_stripped.split('=')[0].strip()
                    env_key_indices[key] = i
    
    # 更新或添加配置項
    for key, raw_value in updates.items():
        # 格式化值用於 .env 文件（不需要引號，除非是字符串且包含空格）
        if raw_value is None or raw_value == '':
            env_value = ''
        elif isinstance(raw_value, (int, float)):
            env_value = str(raw_value)
        elif isinstance(raw_value, bool):
            env_value = 'True' if raw_value else 'False'
        else:
            value_str = str(raw_value)
            # 如果包含空格或特殊字符，需要引號
            if ' ' in value_str or '\n' in value_str or '#' in value_str:
                escaped = value_str.replace('\\', '\\\\').replace('"', '\\"')
                env_value = f'"{escaped}"'
            else:
                env_value = value_str
        
        # 更新或添加配置項
        if key in env_key_indices:
            # 更新現有行
            env_lines[env_key_indices[key]] = f'{key}={env_value}'
        else:
            # 添加新行到文件末尾
            env_lines.append(f'{key}={env_value}')
    
    # 寫入 .env 文件
    env_file_path.parent.mkdir(parents=True, exist_ok=True)
    env_file_path.write_text('\n'.join(env_lines) + '\n', encoding='utf-8')
    
    # 重新加載配置模塊（這會重新讀取 .env 文件並創建新的 Settings 實例）
    _load_config_module()


system_state_lock = threading.Lock()
system_state = {
    'started': False,
    'starting': False
}


def _set_system_state(*, started=None, starting=None):
    """Safely update the cached system state flags."""
    with system_state_lock:
        if started is not None:
            system_state['started'] = started
        if starting is not None:
            system_state['starting'] = starting


def _get_system_state():
    """Return a shallow copy of the system state flags."""
    with system_state_lock:
        return system_state.copy()


def _prepare_system_start():
    """Mark the system as starting if it is not already running or starting."""
    with system_state_lock:
        if system_state['started']:
            return False, '系統已啓動'
        if system_state['starting']:
            return False, '系統正在啓動'
        system_state['starting'] = True
        return True, None


def initialize_system_components():
    """啓動所有依賴組件（Streamlit 子應用、ForumEngine、ReportEngine）。"""
    logs = []
    errors = []
    
    spider = MindSpider()
    if spider.initialize_database():
        logger.info("資料庫初始化成功")
    else:
        logger.error("資料庫初始化失敗")

    try:
        stop_forum_engine()
        logs.append("已停止 ForumEngine 監控器以避免文件衝突")
    except Exception as exc:  # pragma: no cover - 安全捕獲
        message = f"停止 ForumEngine 時發生異常: {exc}"
        logs.append(message)
        logger.exception(message)

    processes['forum']['status'] = 'stopped'

    for app_name, script_path in STREAMLIT_SCRIPTS.items():
        logs.append(f"檢查文件: {script_path}")
        if os.path.exists(script_path):
            success, message = start_streamlit_app(app_name, script_path, processes[app_name]['port'])
            logs.append(f"{app_name}: {message}")
            if success:
                startup_success, startup_message = wait_for_app_startup(app_name, 30)
                logs.append(f"{app_name} 啓動檢查: {startup_message}")
                if not startup_success:
                    errors.append(f"{app_name} 啓動失敗: {startup_message}")
            else:
                errors.append(f"{app_name} 啓動失敗: {message}")
        else:
            msg = f"文件不存在: {script_path}"
            logs.append(f"錯誤: {msg}")
            errors.append(f"{app_name}: {msg}")

    forum_started = False
    try:
        start_forum_engine()
        processes['forum']['status'] = 'running'
        logs.append("ForumEngine 啓動完成")
        forum_started = True
    except Exception as exc:  # pragma: no cover - 保底捕獲
        error_msg = f"ForumEngine 啓動失敗: {exc}"
        logs.append(error_msg)
        errors.append(error_msg)

    if REPORT_ENGINE_AVAILABLE:
        try:
            if initialize_report_engine():
                logs.append("ReportEngine 初始化成功")
            else:
                msg = "ReportEngine 初始化失敗"
                logs.append(msg)
                errors.append(msg)
        except Exception as exc:  # pragma: no cover
            msg = f"ReportEngine 初始化異常: {exc}"
            logs.append(msg)
            errors.append(msg)

    if errors:
        cleanup_processes()
        processes['forum']['status'] = 'stopped'
        if forum_started:
            try:
                stop_forum_engine()
            except Exception:  # pragma: no cover
                logger.exception("停止ForumEngine失敗")
        return False, logs, errors

    return True, logs, []

# 初始化ForumEngine的forum.log文件
def init_forum_log():
    """初始化forum.log文件"""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        # 檢查文件不存在則創建並且寫一個開始，存在就清空寫一個開始
        if not forum_log_file.exists():
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEngine 系統初始化 - {start_time} ===\n")
            logger.info(f"ForumEngine: forum.log 已初始化")
        else:
            with open(forum_log_file, 'w', encoding='utf-8') as f:
                start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== ForumEngine 系統初始化 - {start_time} ===\n")
            logger.info(f"ForumEngine: forum.log 已初始化")
    except Exception as e:
        logger.exception(f"ForumEngine: 初始化forum.log失敗: {e}")

# 初始化forum.log
init_forum_log()

# 啓動ForumEngine智能監控
def start_forum_engine():
    """啓動ForumEngine論壇"""
    try:
        from ForumEngine.monitor import start_forum_monitoring
        logger.info("ForumEngine: 啓動論壇...")
        success = start_forum_monitoring()
        if not success:
            logger.info("ForumEngine: 論壇啓動失敗")
    except Exception as e:
        logger.exception(f"ForumEngine: 啓動論壇失敗: {e}")

# 停止ForumEngine智能監控
def stop_forum_engine():
    """停止ForumEngine論壇"""
    try:
        from ForumEngine.monitor import stop_forum_monitoring
        logger.info("ForumEngine: 停止論壇...")
        stop_forum_monitoring()
        logger.info("ForumEngine: 論壇已停止")
    except Exception as e:
        logger.exception(f"ForumEngine: 停止論壇失敗: {e}")

def parse_forum_log_line(line):
    """解析forum.log行內容，提取對話信息"""
    import re
    
    # 匹配格式: [時間] [來源] 內容
    pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*\[([A-Z]+)\]\s*(.*)'
    match = re.match(pattern, line)
    
    if match:
        timestamp, source, content = match.groups()
        
        # 過濾掉系統消息和空內容
        if source == 'SYSTEM' or not content.strip():
            return None
        
        # 只處理三個Engine的消息
        if source not in ['QUERY', 'INSIGHT', 'MEDIA']:
            return None
        
        # 根據來源確定消息類型和發送者
        message_type = 'agent'
        sender = f'{source} Engine'
        
        return {
            'type': message_type,
            'sender': sender,
            'content': content.strip(),
            'timestamp': timestamp,
            'source': source
        }
    
    return None

# Forum日誌監聽器
def monitor_forum_log():
    """監聽forum.log文件變化並推送到前端"""
    import time
    from pathlib import Path
    
    forum_log_file = LOG_DIR / "forum.log"
    last_position = 0
    processed_lines = set()  # 用於跟蹤已處理的行，避免重複
    
    # 如果文件存在，獲取初始位置
    if forum_log_file.exists():
        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 初始化時讀取所有現有行，避免重複處理
            existing_lines = f.readlines()
            for line in existing_lines:
                line_hash = hash(line.strip())
                processed_lines.add(line_hash)
            last_position = f.tell()
    
    while True:
        try:
            if forum_log_file.exists():
                with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    
                    if new_lines:
                        for line in new_lines:
                            line = line.rstrip('\n\r')
                            if line.strip():
                                line_hash = hash(line.strip())
                                
                                # 避免重複處理同一行
                                if line_hash in processed_lines:
                                    continue
                                
                                processed_lines.add(line_hash)
                                
                                # 解析日誌行併發送forum消息
                                parsed_message = parse_forum_log_line(line)
                                if parsed_message:
                                    socketio.emit('forum_message', parsed_message)
                                
                                # 只有在控制檯顯示forum時才發送控制檯消息
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                formatted_line = f"[{timestamp}] {line}"
                                socketio.emit('console_output', {
                                    'app': 'forum',
                                    'line': formatted_line
                                })
                        
                        last_position = f.tell()
                        
                        # 清理processed_lines集合，避免內存泄漏（保留最近1000行的哈希）
                        if len(processed_lines) > 1000:
                            processed_lines.clear()
            
            time.sleep(1)  # 每秒檢查一次
        except Exception as e:
            logger.error(f"Forum日誌監聽錯誤: {e}")
            time.sleep(5)

# 啓動Forum日誌監聽線程
forum_monitor_thread = threading.Thread(target=monitor_forum_log, daemon=True)
forum_monitor_thread.start()

# 全局變量存儲進程信息
processes = {
    'insight': {'process': None, 'port': 8501, 'status': 'stopped', 'output': [], 'log_file': None},
    'media': {'process': None, 'port': 8502, 'status': 'stopped', 'output': [], 'log_file': None},
    'query': {'process': None, 'port': 8503, 'status': 'stopped', 'output': [], 'log_file': None},
    'forum': {'process': None, 'port': None, 'status': 'stopped', 'output': [], 'log_file': None}  # 啓動後標記爲 running
}

STREAMLIT_SCRIPTS = {
    'insight': 'SingleEngineApp/insight_engine_streamlit_app.py',
    'media': 'SingleEngineApp/media_engine_streamlit_app.py',
    'query': 'SingleEngineApp/query_engine_streamlit_app.py'
}

# 輸出隊列
output_queues = {
    'insight': Queue(),
    'media': Queue(),
    'query': Queue(),
    'forum': Queue()
}

def write_log_to_file(app_name, line):
    """將日誌寫入文件"""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
    except Exception as e:
        logger.error(f"Error writing log for {app_name}: {e}")

def read_log_from_file(app_name, tail_lines=None):
    """從文件讀取日誌"""
    try:
        log_file_path = LOG_DIR / f"{app_name}.log"
        if not log_file_path.exists():
            return []
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
            
            if tail_lines:
                return lines[-tail_lines:]
            return lines
    except Exception as e:
        logger.exception(f"Error reading log for {app_name}: {e}")
        return []

def read_process_output(process, app_name):
    """讀取進程輸出並寫入文件"""
    import select
    import sys
    
    while True:
        try:
            if process.poll() is not None:
                # 進程結束，讀取剩餘輸出
                remaining_output = process.stdout.read()
                if remaining_output:
                    lines = remaining_output.decode('utf-8', errors='replace').split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            write_log_to_file(app_name, formatted_line)
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                break
            
            # 使用非阻塞讀取
            if sys.platform == 'win32':
                # Windows下使用不同的方法
                output = process.stdout.readline()
                if output:
                    line = output.decode('utf-8', errors='replace').strip()
                    if line:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        formatted_line = f"[{timestamp}] {line}"
                        
                        # 寫入日誌文件
                        write_log_to_file(app_name, formatted_line)
                        
                        # 發送到前端
                        socketio.emit('console_output', {
                            'app': app_name,
                            'line': formatted_line
                        })
                else:
                    # 沒有輸出時短暫休眠
                    time.sleep(0.1)
            else:
                # Unix系統使用select
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    output = process.stdout.readline()
                    if output:
                        line = output.decode('utf-8', errors='replace').strip()
                        if line:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            formatted_line = f"[{timestamp}] {line}"
                            
                            # 寫入日誌文件
                            write_log_to_file(app_name, formatted_line)
                            
                            # 發送到前端
                            socketio.emit('console_output', {
                                'app': app_name,
                                'line': formatted_line
                            })
                            
        except Exception as e:
            error_msg = f"Error reading output for {app_name}: {e}"
            logger.exception(error_msg)
            write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
            break

def start_streamlit_app(app_name, script_path, port):
    """啓動Streamlit應用"""
    try:
        if processes[app_name]['process'] is not None:
            return False, "應用已經在運行"
        
        # 檢查文件是否存在
        if not os.path.exists(script_path):
            return False, f"文件不存在: {script_path}"
        
        # 清空之前的日誌文件
        log_file_path = LOG_DIR / f"{app_name}.log"
        if log_file_path.exists():
            log_file_path.unlink()
        
        # 創建啓動日誌
        start_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 啓動 {app_name} 應用..."
        write_log_to_file(app_name, start_msg)
        
        # 不設置 baseUrlPath,讓 Streamlit 在根路徑提供服務
        # Nginx 會通過 rewrite 規則剝除 /insight, /media, /query 前綴
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            script_path,
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--logger.level', 'info',
            '--server.enableCORS', 'false',  # 禁用 CORS,由 Nginx 處理
            '--server.enableXsrfProtection', 'false',  # 禁用 XSRF,透過 Nginx 反向代理時會阻止 WebSocket
        ]
        
        # 設置環境變量確保UTF-8編碼和減少緩衝
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
            'PYTHONUNBUFFERED': '1',  # 禁用Python緩衝
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
        })
        
        # 使用當前工作目錄而不是腳本目錄
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,  # 無緩衝
            universal_newlines=False,
            cwd=os.getcwd(),
            env=env,
            encoding=None,  # 讓我們手動處理編碼
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        processes[app_name]['process'] = process
        processes[app_name]['status'] = 'starting'
        processes[app_name]['output'] = []
        
        # 啓動輸出讀取線程
        output_thread = threading.Thread(
            target=read_process_output,
            args=(process, app_name),
            daemon=True
        )
        output_thread.start()
        
        return True, f"{app_name} 應用啓動中..."
        
    except Exception as e:
        error_msg = f"啓動失敗: {str(e)}"
        write_log_to_file(app_name, f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
        return False, error_msg

def stop_streamlit_app(app_name):
    """停止Streamlit應用"""
    try:
        if processes[app_name]['process'] is None:
            return False, "應用未運行"
        
        process = processes[app_name]['process']
        process.terminate()
        
        # 等待進程結束
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        processes[app_name]['process'] = None
        processes[app_name]['status'] = 'stopped'
        
        return True, f"{app_name} 應用已停止"
        
    except Exception as e:
        return False, f"停止失敗: {str(e)}"

HEALTHCHECK_PATH = "/_stcore/health"
HEALTHCHECK_PROXIES = {'http': None, 'https': None}


def _build_healthcheck_url(port):
    return f"http://127.0.0.1:{port}{HEALTHCHECK_PATH}"


def check_app_status():
    """檢查應用狀態"""
    for app_name, info in processes.items():
        if info['process'] is not None:
            if info['process'].poll() is None:
                # 進程仍在運行，檢查端口是否可訪問
                try:
                    response = requests.get(
                        _build_healthcheck_url(info['port']),
                        timeout=2,
                        proxies=HEALTHCHECK_PROXIES
                    )
                    if response.status_code == 200:
                        info['status'] = 'running'
                    else:
                        info['status'] = 'starting'
                except Exception as exc:
                    logger.warning(f"{app_name} 健康检查失败: {exc}")
                    info['status'] = 'starting'
            else:
                # 進程已結束
                info['process'] = None
                info['status'] = 'stopped'

def wait_for_app_startup(app_name, max_wait_time=90):
    """等待應用啟動完成"""
    import time
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        info = processes[app_name]
        if info['process'] is None:
            return False, "進程已停止"

        if info['process'].poll() is not None:
            return False, "進程啓動失敗"

        try:
            response = requests.get(
                _build_healthcheck_url(info['port']),
                timeout=2,
                proxies=HEALTHCHECK_PROXIES
            )
            if response.status_code == 200:
                info['status'] = 'running'
                return True, "啟動成功"
        except Exception as exc:
            logger.warning(f"{app_name} 健康檢查失敗: {exc}")

        time.sleep(1)

    return False, "啟動超時"

def cleanup_processes():
    """清理所有進程"""
    for app_name in STREAMLIT_SCRIPTS:
        stop_streamlit_app(app_name)

    processes['forum']['status'] = 'stopped'
    try:
        stop_forum_engine()
    except Exception:  # pragma: no cover
        logger.exception("停止ForumEngine失敗")
    _set_system_state(started=False, starting=False)

# 註冊清理函數
atexit.register(cleanup_processes)

@app.route('/')
def index():
    """主頁"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """獲取所有應用狀態"""
    check_app_status()
    return jsonify({
        app_name: {
            'status': info['status'],
            'port': info['port'],
            'output_lines': len(info['output'])
        }
        for app_name, info in processes.items()
    })

@app.route('/api/start/<app_name>')
def start_app(app_name):
    """啓動指定應用"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知應用'})

    if app_name == 'forum':
        try:
            start_forum_engine()
            processes['forum']['status'] = 'running'
            return jsonify({'success': True, 'message': 'ForumEngine已啓動'})
        except Exception as exc:  # pragma: no cover
            logger.exception("手動啓動ForumEngine失敗")
            return jsonify({'success': False, 'message': f'ForumEngine啓動失敗: {exc}'})

    script_path = STREAMLIT_SCRIPTS.get(app_name)
    if not script_path:
        return jsonify({'success': False, 'message': '該應用不支持啓動操作'})

    success, message = start_streamlit_app(
        app_name,
        script_path,
        processes[app_name]['port']
    )

    if success:
        # 等待應用啓動
        startup_success, startup_message = wait_for_app_startup(app_name, 15)
        if not startup_success:
            message += f" 但啓動檢查失敗: {startup_message}"
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/stop/<app_name>')
def stop_app(app_name):
    """停止指定應用"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知應用'})

    if app_name == 'forum':
        try:
            stop_forum_engine()
            processes['forum']['status'] = 'stopped'
            return jsonify({'success': True, 'message': 'ForumEngine已停止'})
        except Exception as exc:  # pragma: no cover
            logger.exception("手動停止ForumEngine失敗")
            return jsonify({'success': False, 'message': f'ForumEngine停止失敗: {exc}'})

    success, message = stop_streamlit_app(app_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/output/<app_name>')
def get_output(app_name):
    """獲取應用輸出"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知應用'})
    
    # 特殊處理Forum Engine
    if app_name == 'forum':
        try:
            forum_log_content = read_log_from_file('forum')
            return jsonify({
                'success': True,
                'output': forum_log_content,
                'total_lines': len(forum_log_content)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'讀取forum日誌失敗: {str(e)}'})
    
    # 從文件讀取完整日誌
    output_lines = read_log_from_file(app_name)
    
    return jsonify({
        'success': True,
        'output': output_lines
    })

@app.route('/api/test_log/<app_name>')
def test_log(app_name):
    """測試日誌寫入功能"""
    if app_name not in processes:
        return jsonify({'success': False, 'message': '未知應用'})
    
    # 寫入測試消息
    test_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 測試日誌消息 - {datetime.now()}"
    write_log_to_file(app_name, test_msg)
    
    # 通過Socket.IO發送
    socketio.emit('console_output', {
        'app': app_name,
        'line': test_msg
    })
    
    return jsonify({
        'success': True,
        'message': f'測試消息已寫入 {app_name} 日誌'
    })

@app.route('/api/forum/start')
def start_forum_monitoring_api():
    """手動啓動ForumEngine論壇"""
    try:
        from ForumEngine.monitor import start_forum_monitoring
        success = start_forum_monitoring()
        if success:
            return jsonify({'success': True, 'message': 'ForumEngine論壇已啓動'})
        else:
            return jsonify({'success': False, 'message': 'ForumEngine論壇啓動失敗'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'啓動論壇失敗: {str(e)}'})

@app.route('/api/forum/stop')
def stop_forum_monitoring_api():
    """手動停止ForumEngine論壇"""
    try:
        from ForumEngine.monitor import stop_forum_monitoring
        stop_forum_monitoring()
        return jsonify({'success': True, 'message': 'ForumEngine論壇已停止'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'停止論壇失敗: {str(e)}'})

@app.route('/api/forum/log')
def get_forum_log():
    """獲取ForumEngine的forum.log內容"""
    try:
        forum_log_file = LOG_DIR / "forum.log"
        if not forum_log_file.exists():
            return jsonify({
                'success': True,
                'log_lines': [],
                'parsed_messages': [],
                'total_lines': 0
            })
        
        with open(forum_log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
        
        # 解析每一行日誌並提取對話信息
        parsed_messages = []
        for line in lines:
            parsed_message = parse_forum_log_line(line)
            if parsed_message:
                parsed_messages.append(parsed_message)
        
        return jsonify({
            'success': True,
            'log_lines': lines,
            'parsed_messages': parsed_messages,
            'total_lines': len(lines)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'讀取forum.log失敗: {str(e)}'})

@app.route('/api/search', methods=['POST'])
def search():
    """統一搜索接口"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'success': False, 'message': '搜索查詢不能爲空'})
    
    # ForumEngine論壇已經在後臺運行，會自動檢測搜索活動
    # logger.info("ForumEngine: 搜索請求已收到，論壇將自動檢測日誌變化")
    
    # 檢查哪些應用正在運行
    check_app_status()
    running_apps = [name for name, info in processes.items() if info['status'] == 'running']
    
    if not running_apps:
        return jsonify({'success': False, 'message': '沒有運行中的應用'})
    
    # 向運行中的應用發送搜索請求
    results = {}
    api_ports = {'insight': 8601, 'media': 8602, 'query': 8603}
    
    for app_name in running_apps:
        try:
            api_port = api_ports[app_name]
            # 調用Streamlit應用的API端點
            response = requests.post(
                f"http://localhost:{api_port}/api/search",
                json={'query': query},
                timeout=10
            )
            if response.status_code == 200:
                results[app_name] = response.json()
            else:
                results[app_name] = {'success': False, 'message': 'API調用失敗'}
        except Exception as e:
            results[app_name] = {'success': False, 'message': str(e)}
    
    # 搜索完成後可以選擇停止監控，或者讓它繼續運行以捕獲後續的處理日誌
    # 這裏我們讓監控繼續運行，用戶可以通過其他接口手動停止
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Expose selected configuration values to the frontend."""
    try:
        config_values = read_config_values()
        return jsonify({'success': True, 'config': config_values})
    except Exception as exc:
        logger.exception("讀取配置失敗")
        return jsonify({'success': False, 'message': f'讀取配置失敗: {exc}'}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration values and persist them to config.py."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload:
        return jsonify({'success': False, 'message': '請求體不能爲空'}), 400

    updates = {}
    for key, value in payload.items():
        if key in CONFIG_KEYS:
            updates[key] = value if value is not None else ''

    if not updates:
        return jsonify({'success': False, 'message': '沒有可更新的配置項'}), 400

    try:
        write_config_values(updates)
        updated_config = read_config_values()
        return jsonify({'success': True, 'config': updated_config})
    except Exception as exc:
        logger.exception("更新配置失敗")
        return jsonify({'success': False, 'message': f'更新配置失敗: {exc}'}), 500


@app.route('/api/system/status')
def get_system_status():
    """返回系統啓動狀態。"""
    state = _get_system_state()
    return jsonify({
        'success': True,
        'started': state['started'],
        'starting': state['starting']
    })


@app.route('/api/system/start', methods=['POST'])
def start_system():
    """在接收到請求後啓動完整系統。"""
    allowed, message = _prepare_system_start()
    if not allowed:
        return jsonify({'success': False, 'message': message}), 400

    try:
        success, logs, errors = initialize_system_components()
        if success:
            _set_system_state(started=True)
            return jsonify({'success': True, 'message': '系統啓動成功', 'logs': logs})

        _set_system_state(started=False)
        return jsonify({
            'success': False,
            'message': '系統啓動失敗',
            'logs': logs,
            'errors': errors
        }), 500
    except Exception as exc:  # pragma: no cover - 保底捕獲
        logger.exception("系統啓動過程中出現異常")
        _set_system_state(started=False)
        return jsonify({'success': False, 'message': f'系統啓動異常: {exc}'}), 500
    finally:
        _set_system_state(starting=False)

@socketio.on('connect')
def handle_connect():
    """客戶端連接"""
    emit('status', 'Connected to Flask server')

@socketio.on('request_status')
def handle_status_request():
    """請求狀態更新"""
    check_app_status()
    emit('status_update', {
        app_name: {
            'status': info['status'],
            'port': info['port']
        }
        for app_name, info in processes.items()
    })

if __name__ == '__main__':
    # 從配置文件讀取 HOST 和 PORT
    from config import settings
    HOST = settings.HOST
    PORT = settings.PORT

    logger.info("等待配置確認，系統將在前端指令後啟動組件...")
    logger.info(f"Flask服務器已啟動，訪問地址: http://{HOST}:{PORT}")
    
    try:
        socketio.run(app, host=HOST, port=PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("\n正在關閉應用...")
        cleanup_processes()
        
    

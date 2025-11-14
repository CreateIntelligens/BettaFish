"""
Report Engine Flask接口
提供HTTP API用於報告生成
"""

import os
import json
import threading
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, Response, send_file
from typing import Dict, Any
from loguru import logger
from .agent import ReportAgent, create_agent
from .utils.config import settings


# 創建Blueprint
report_bp = Blueprint('report_engine', __name__)

# 全局變量
report_agent = None
current_task = None
task_lock = threading.Lock()


def initialize_report_engine():
    """初始化Report Engine"""
    global report_agent
    try:
        report_agent = create_agent()
        logger.info("Report Engine初始化成功")
        return True
    except Exception as e:
        logger.exception(f"Report Engine初始化失敗: {str(e)}")
        return False


class ReportTask:
    """報告生成任務"""

    def __init__(self, query: str, task_id: str, custom_template: str = ""):
        self.task_id = task_id
        self.query = query
        self.custom_template = custom_template
        self.status = "pending"  # pending, running, completed, error
        self.progress = 0
        self.result = None
        self.error_message = ""
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.html_content = ""
        self.report_file_path = ""
        self.report_file_relative_path = ""
        self.report_file_name = ""
        self.state_file_path = ""
        self.state_file_relative_path = ""

    def update_status(self, status: str, progress: int = None, error_message: str = ""):
        """更新任務狀態"""
        self.status = status
        if progress is not None:
            self.progress = progress
        if error_message:
            self.error_message = error_message
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            'task_id': self.task_id,
            'query': self.query,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'has_result': bool(self.html_content),
            'report_file_ready': bool(self.report_file_path),
            'report_file_name': self.report_file_name,
            'report_file_path': self.report_file_relative_path
        }


def check_engines_ready() -> Dict[str, Any]:
    """檢查三個子引擎是否都有新文件"""
    directories = {
        'insight': 'insight_engine_streamlit_reports',
        'media': 'media_engine_streamlit_reports',
        'query': 'query_engine_streamlit_reports'
    }

    forum_log_path = 'logs/forum.log'

    if not report_agent:
        return {
            'ready': False,
            'error': 'Report Engine未初始化'
        }

    return report_agent.check_input_files(
        directories['insight'],
        directories['media'],
        directories['query'],
        forum_log_path
    )


def run_report_generation(task: ReportTask, query: str, custom_template: str = ""):
    """在後臺線程中運行報告生成"""
    global current_task

    try:
        task.update_status("running", 10)

        # 檢查輸入文件
        check_result = check_engines_ready()
        if not check_result['ready']:
            task.update_status("error", 0, f"輸入文件未準備就緒: {check_result.get('missing_files', [])}")
            return

        task.update_status("running", 30)

        # 加載輸入文件
        content = report_agent.load_input_files(check_result['latest_files'])

        task.update_status("running", 50)

        # 生成報告
        generation_result = report_agent.generate_report(
            query=query,
            reports=content['reports'],
            forum_logs=content['forum_logs'],
            custom_template=custom_template,
            save_report=True
        )

        html_report = generation_result.get('html_content', '')

        task.update_status("running", 90)

        # 保存結果
        task.html_content = html_report
        task.report_file_path = generation_result.get('report_filepath', '')
        task.report_file_relative_path = generation_result.get('report_relative_path', '')
        task.report_file_name = generation_result.get('report_filename', '')
        task.state_file_path = generation_result.get('state_filepath', '')
        task.state_file_relative_path = generation_result.get('state_relative_path', '')
        task.update_status("completed", 100)

    except Exception as e:
        logger.exception(f"報告生成過程中發生錯誤: {str(e)}")
        task.update_status("error", 0, str(e))
        # 只在出錯時清理任務
        with task_lock:
            if current_task and current_task.task_id == task.task_id:
                current_task = None


@report_bp.route('/status', methods=['GET'])
def get_status():
    """獲取Report Engine狀態"""
    try:
        engines_status = check_engines_ready()

        return jsonify({
            'success': True,
            'initialized': report_agent is not None,
            'engines_ready': engines_status['ready'],
            'files_found': engines_status.get('files_found', []),
            'missing_files': engines_status.get('missing_files', []),
            'current_task': current_task.to_dict() if current_task else None
        })
    except Exception as e:
        logger.exception(f"獲取Report Engine狀態失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """開始生成報告"""
    global current_task

    try:
        # 檢查是否有任務在運行
        with task_lock:
            if current_task and current_task.status == "running":
                return jsonify({
                    'success': False,
                    'error': '已有報告生成任務在運行中',
                    'current_task': current_task.to_dict()
                }), 400

            # 如果有已完成的任務，清理它
            if current_task and current_task.status in ["completed", "error"]:
                current_task = None

        # 獲取請求參數
        data = request.get_json() or {}
        query = data.get('query', '智能輿情分析報告')
        custom_template = data.get('custom_template', '')

        # 清空日誌文件
        clear_report_log()

        # 檢查Report Engine是否初始化
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Engine未初始化'
            }), 500

        # 檢查輸入文件是否準備就緒
        engines_status = check_engines_ready()
        if not engines_status['ready']:
            return jsonify({
                'success': False,
                'error': '輸入文件未準備就緒',
                'missing_files': engines_status.get('missing_files', [])
            }), 400

        # 創建新任務
        task_id = f"report_{int(time.time())}"
        task = ReportTask(query, task_id, custom_template)

        with task_lock:
            current_task = task

        # 在後臺線程中運行報告生成
        thread = threading.Thread(
            target=run_report_generation,
            args=(task, query, custom_template),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '報告生成已啓動',
            'task': task.to_dict()
        })

    except Exception as e:
        logger.exception(f"開始生成報告失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id: str):
    """獲取報告生成進度"""
    try:
        if not current_task or current_task.task_id != task_id:
            # 如果任務不存在，可能是已經完成並被清理了
            # 返回一個默認的完成狀態而不是404
            return jsonify({
                'success': True,
                'task': {
                    'task_id': task_id,
                    'status': 'completed',
                    'progress': 100,
                    'error_message': '',
                    'has_result': True,
                    'report_file_ready': False,
                    'report_file_name': '',
                    'report_file_path': ''
                }
            })

        return jsonify({
            'success': True,
            'task': current_task.to_dict()
        })

    except Exception as e:
        logger.exception(f"獲取報告生成進度失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/result/<task_id>', methods=['GET'])
def get_result(task_id: str):
    """獲取報告生成結果"""
    try:
        if not current_task or current_task.task_id != task_id:
            return jsonify({
                'success': False,
                'error': '任務不存在'
            }), 404

        if current_task.status != "completed":
            return jsonify({
                'success': False,
                'error': '報告尚未完成',
                'task': current_task.to_dict()
            }), 400

        return Response(
            current_task.html_content,
            mimetype='text/html'
        )

    except Exception as e:
        logger.exception(f"獲取報告生成結果失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/result/<task_id>/json', methods=['GET'])
def get_result_json(task_id: str):
    """獲取報告生成結果（JSON格式）"""
    try:
        if not current_task or current_task.task_id != task_id:
            return jsonify({
                'success': False,
                'error': '任務不存在'
            }), 404

        if current_task.status != "completed":
            return jsonify({
                'success': False,
                'error': '報告尚未完成',
                'task': current_task.to_dict()
            }), 400

        return jsonify({
            'success': True,
            'task': current_task.to_dict(),
            'html_content': current_task.html_content
        })

    except Exception as e:
        logger.exception(f"獲取報告生成結果失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/download/<task_id>', methods=['GET'])
def download_report(task_id: str):
    """下載已生成的報告HTML文件"""
    try:
        if not current_task or current_task.task_id != task_id:
            return jsonify({
                'success': False,
                'error': '任務不存在'
            }), 404

        if current_task.status != "completed" or not current_task.report_file_path:
            return jsonify({
                'success': False,
                'error': '報告尚未完成或尚未保存'
            }), 400

        if not os.path.exists(current_task.report_file_path):
            return jsonify({
                'success': False,
                'error': '報告文件不存在或已被刪除'
            }), 404

        download_name = current_task.report_file_name or os.path.basename(current_task.report_file_path)
        return send_file(
            current_task.report_file_path,
            mimetype='text/html',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        logger.exception(f"下載報告失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id: str):
    """取消報告生成任務"""
    global current_task

    try:
        with task_lock:
            if current_task and current_task.task_id == task_id:
                if current_task.status == "running":
                    current_task.update_status("cancelled", 0, "用戶取消任務")
                current_task = None

                return jsonify({
                    'success': True,
                    'message': '任務已取消'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '任務不存在或無法取消'
                }), 404

    except Exception as e:
        logger.exception(f"取消報告生成任務失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@report_bp.route('/templates', methods=['GET'])
def get_templates():
    """獲取可用模板列表"""
    try:
        if not report_agent:
            return jsonify({
                'success': False,
                'error': 'Report Engine未初始化'
            }), 500

        template_dir = settings.TEMPLATE_DIR
        templates = []

        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.md'):
                    template_path = os.path.join(template_dir, filename)
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        templates.append({
                            'name': filename.replace('.md', ''),
                            'filename': filename,
                            'description': content.split('\n')[0] if content else '無描述',
                            'size': len(content)
                        })
                    except Exception as e:
                        logger.exception(f"讀取模板失敗 {filename}: {str(e)}")

        return jsonify({
            'success': True,
            'templates': templates,
            'template_dir': template_dir
        })

    except Exception as e:
        logger.exception(f"獲取可用模板列表失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 錯誤處理
@report_bp.errorhandler(404)
def not_found(error):
    logger.exception(f"API端點不存在: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'API端點不存在'
    }), 404


@report_bp.errorhandler(500)
def internal_error(error):
    logger.exception(f"服務器內部錯誤: {str(error)}")
    return jsonify({
        'success': False,
        'error': '服務器內部錯誤'
    }), 500


def clear_report_log():
    """清空report.log文件"""
    try:
        log_file = settings.LOG_FILE
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('')
        logger.info(f"已清空日誌文件: {log_file}")
    except Exception as e:
        logger.exception(f"清空日誌文件失敗: {str(e)}")


@report_bp.route('/log', methods=['GET'])
def get_report_log():
    """獲取report.log內容"""
    try:
        log_file = settings.LOG_FILE
        
        if not os.path.exists(log_file):
            return jsonify({
                'success': True,
                'log_lines': []
            })
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 清理行尾的換行符
        log_lines = [line.rstrip('\n\r') for line in lines if line.strip()]
        
        return jsonify({
            'success': True,
            'log_lines': log_lines
        })
        
    except Exception as e:
        logger.exception(f"讀取日誌失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'讀取日誌失敗: {str(e)}'
        }), 500


@report_bp.route('/log/clear', methods=['POST'])
def clear_log():
    """手動清空日誌"""
    try:
        clear_report_log()
        return jsonify({
            'success': True,
            'message': '日誌已清空'
        })
    except Exception as e:
        logger.exception(f"清空日誌失敗: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'清空日誌失敗: {str(e)}'
        }), 500

"""
簡單的測試運行腳本

可以直接運行此腳本來執行測試
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_monitor import TestLogMonitor


def main():
    """運行所有測試"""
    print("=" * 60)
    print("ForumEngine 日誌解析測試")
    print("=" * 60)
    print()
    
    test_instance = TestLogMonitor()
    test_instance.setup_method()
    
    # 獲取所有測試方法
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for test_method_name in test_methods:
        test_method = getattr(test_instance, test_method_name)
        print(f"運行測試: {test_method_name}...", end=" ")
        
        try:
            test_method()
            print("✓ 通過")
            passed += 1
        except AssertionError as e:
            print(f"✗ 失敗: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ 錯誤: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"測試結果: {passed} 通過, {failed} 失敗")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()


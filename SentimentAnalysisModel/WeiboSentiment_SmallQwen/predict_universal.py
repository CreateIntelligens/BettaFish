#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3微博情感分析統一預測接口
支持0.6B、4B、8B三種規格的Embedding和LoRA模型
"""

import os
import sys
import argparse
import torch
from typing import List, Dict, Tuple, Any

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_config import QWEN3_MODELS, MODEL_PATHS
from qwen3_embedding_universal import Qwen3EmbeddingUniversal
from qwen3_lora_universal import Qwen3LoRAUniversal


class Qwen3UniversalPredictor:
    """Qwen3統一預測器"""
    
    def __init__(self):
        self.models = {}  # 存儲已加載的模型 {model_key: {model: obj, display_name: str}}
        
    def _get_model_key(self, model_type: str, model_size: str) -> str:
        """生成模型鍵值"""
        return f"{model_type}_{model_size}"
    
    def load_model(self, model_type: str, model_size: str) -> None:
        """加載指定的模型"""
        if model_type not in ['embedding', 'lora']:
            raise ValueError(f"不支持的模型類型: {model_type}")
        if model_size not in ['0.6B', '4B', '8B']:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        model_path = MODEL_PATHS[model_type][model_size]
        model_key = self._get_model_key(model_type, model_size)
        
        # 檢查訓練好的模型文件是否存在
        if not os.path.exists(model_path):
            print(f"訓練好的模型文件不存在: {model_path}")
            print(f"請先訓練 {model_type.upper()}-{model_size} 模型，或檢查模型路徑配置")
            return
        
        print(f"加載 {model_type.upper()}-{model_size} 模型...")
        
        try:
            if model_type == 'embedding':
                model = Qwen3EmbeddingUniversal(model_size)
                model.load_model(model_path)
            else:  # lora
                model = Qwen3LoRAUniversal(model_size)
                model.load_model(model_path)
            
            self.models[model_key] = {
                'model': model,
                'display_name': f"Qwen3-{model_type.title()}-{model_size}"
            }
            print(f"{model_type.upper()}-{model_size} 模型加載成功")
            
        except Exception as e:
            print(f"加載 {model_type.upper()}-{model_size} 模型失敗: {e}")
            print(f"這可能是因爲基礎模型下載失敗或訓練好的模型文件損壞")
    
    def load_all_models(self, model_dir: str = './models') -> None:
        """加載所有可用的模型"""
        print("開始加載所有可用的Qwen3模型...")
        
        loaded_count = 0
        for model_type in ['embedding', 'lora']:
            for model_size in ['0.6B', '4B', '8B']:
                try:
                    self.load_model(model_type, model_size)
                    loaded_count += 1
                except Exception as e:
                    print(f"跳過 {model_type}-{model_size}: {e}")
        
        print(f"\n已加載 {loaded_count} 個模型")
        self._print_loaded_models()
    
    def load_specific_models(self, model_configs: List[Tuple[str, str]]) -> None:
        """加載指定的模型配置
        Args:
            model_configs: [(model_type, model_size), ...] 的列表
        """
        print("加載指定的Qwen3模型...")
        
        for model_type, model_size in model_configs:
            try:
                self.load_model(model_type, model_size)
            except Exception as e:
                print(f"跳過 {model_type}-{model_size}: {e}")
        
        print(f"\n已加載 {len(self.models)} 個模型")
        self._print_loaded_models()
    
    def _print_loaded_models(self):
        """打印已加載的模型列表"""
        if self.models:
            print("已加載模型:")
            for model_info in self.models.values():
                print(f"  - {model_info['display_name']}")
        else:
            print("沒有成功加載任何模型")
    
    def predict_single(self, text: str, model_key: str = None) -> Dict[str, Tuple[int, float]]:
        """單文本預測
        Args:
            text: 要預測的文本
            model_key: 指定模型鍵值，None表示使用所有模型
        Returns:
            {model_name: (prediction, confidence), ...}
        """
        results = {}
        
        if model_key and model_key in self.models:
            # 使用指定模型
            model_info = self.models[model_key]
            try:
                prediction, confidence = model_info['model'].predict_single(text)
                results[model_info['display_name']] = (prediction, confidence)
            except Exception as e:
                print(f"模型 {model_info['display_name']} 預測失敗: {e}")
                results[model_info['display_name']] = (0, 0.0)
        else:
            # 使用所有模型
            for model_info in self.models.values():
                try:
                    prediction, confidence = model_info['model'].predict_single(text)
                    results[model_info['display_name']] = (prediction, confidence)
                except Exception as e:
                    print(f"模型 {model_info['display_name']} 預測失敗: {e}")
                    results[model_info['display_name']] = (0, 0.0)
        
        return results
    
    def predict_batch(self, texts: List[str]) -> Dict[str, List[int]]:
        """批量預測"""
        results = {}
        
        for model_info in self.models.values():
            try:
                predictions = model_info['model'].predict(texts)
                results[model_info['display_name']] = predictions
            except Exception as e:
                print(f"模型 {model_info['display_name']} 預測失敗: {e}")
                results[model_info['display_name']] = [0] * len(texts)
        
        return results
    
    def ensemble_predict(self, text: str) -> Tuple[int, float]:
        """集成預測"""
        if len(self.models) < 2:
            raise ValueError("集成預測需要至少2個模型")
        
        results = self.predict_single(text)
        
        # 加權平均（這裏使用簡單平均，可以根據模型性能調整權重）
        total_weight = 0
        weighted_prob = 0
        
        for model_name, (pred, conf) in results.items():
            if conf > 0:  # 只考慮有效預測
                prob = conf if pred == 1 else 1 - conf
                weighted_prob += prob
                total_weight += 1
        
        if total_weight == 0:
            return 0, 0.5
        
        final_prob = weighted_prob / total_weight
        final_pred = int(final_prob > 0.5)
        final_conf = final_prob if final_pred == 1 else 1 - final_prob
        
        return final_pred, final_conf
    
    def _select_and_load_model(self):
        """讓用戶選擇並加載模型"""
        print("Qwen3微博情感分析預測系統")
        print("="*40)
        print("請選擇要使用的模型:")
        print("\n方法選擇:")
        print("  1. Embedding + 分類頭 (推理快速，顯存佔用少)")
        print("  2. LoRA微調 (效果更好，顯存佔用較多)")
        
        method_choice = None
        while method_choice not in ['1', '2']:
            method_choice = input("\n請選擇方法 (1/2): ").strip()
            if method_choice not in ['1', '2']:
                print("無效選擇，請輸入 1 或 2")
        
        method_type = "embedding" if method_choice == '1' else "lora"
        method_name = "Embedding + 分類頭" if method_choice == '1' else "LoRA微調"
        
        print(f"\n已選擇: {method_name}")
        print("\n模型大小選擇:")
        print("  1. 0.6B - 輕量級，推理快速")
        print("  2. 4B  - 中等規模，性能均衡") 
        print("  3. 8B  - 大規模，性能最佳")
        
        size_choice = None
        while size_choice not in ['1', '2', '3']:
            size_choice = input("\n請選擇模型大小 (1/2/3): ").strip()
            if size_choice not in ['1', '2', '3']:
                print("無效選擇，請輸入 1、2 或 3")
        
        size_map = {'1': '0.6B', '2': '4B', '3': '8B'}
        model_size = size_map[size_choice]
        
        print(f"已選擇: Qwen3-{method_name}-{model_size}")
        print("正在加載模型...")
        
        try:
            self.load_model(method_type, model_size)
            print(f"模型加載成功!")
        except Exception as e:
            print(f"模型加載失敗: {e}")
            print("請檢查模型文件是否存在，或先進行訓練")
    
    def interactive_predict(self):
        """交互式預測模式"""
        if len(self.models) == 0:
            # 讓用戶選擇要加載的模型
            self._select_and_load_model()
            if len(self.models) == 0:
                print("沒有加載任何模型，退出預測")
                return
        
        print("\n" + "="*60)
        print("Qwen3微博情感分析預測系統")
        print("="*60)
        print("已加載模型:")
        for model_info in self.models.values():
            print(f"   - {model_info['display_name']}")
        print("\n命令提示:")
        print("   輸入 'q' 退出程序")
        print("   輸入 'switch' 切換模型")  
        print("   輸入 'models' 查看已加載模型")
        print("   輸入 'compare' 比較所有模型性能")
        print("-"*60)
        
        while True:
            try:
                text = input("\n請輸入要分析的微博內容: ").strip()
                
                if text.lower() == 'q':
                    print("感謝使用，再見！")
                    break
                
                if text.lower() == 'models':
                    print("已加載模型:")
                    for model_info in self.models.values():
                        print(f"   - {model_info['display_name']}")
                    continue
                
                if text.lower() == 'switch':
                    print("切換模型...")
                    self.models.clear()  # 清空當前模型
                    self._select_and_load_model()
                    if len(self.models) > 0:
                        print("模型切換成功!")
                        for model_info in self.models.values():
                            print(f"   當前模型: {model_info['display_name']}")
                    continue
                
                if text.lower() == 'compare':
                    test_text = input("請輸入要比較的文本: ")
                    self._compare_models(test_text)
                    continue
                
                if not text:
                    print("請輸入有效內容")
                    continue
                
                # 預測
                results = self.predict_single(text)
                
                print(f"\n原文: {text}")
                print("預測結果:")
                
                # 按模型類型和大小排序顯示
                sorted_results = sorted(results.items())
                for model_name, (pred, conf) in sorted_results:
                    sentiment = "正面" if pred == 1 else "負面"
                    print(f"   {model_name:20}: {sentiment} (置信度: {conf:.4f})")
                
                # 只顯示單個模型的預測結果（不進行集成）
                
            except KeyboardInterrupt:
                print("\n\n程序被中斷，再見！")
                break
            except Exception as e:
                print(f"預測過程中出現錯誤: {e}")
    
    def _compare_models(self, text: str):
        """比較不同模型的性能"""
        print(f"\n模型性能比較 - 文本: {text}")
        print("-" * 60)
        
        results = self.predict_single(text)
        
        embedding_models = []
        lora_models = []
        
        for model_name, (pred, conf) in results.items():
            sentiment = "正面" if pred == 1 else "負面"
            if "Embedding" in model_name:
                embedding_models.append((model_name, sentiment, conf))
            elif "Lora" in model_name:
                lora_models.append((model_name, sentiment, conf))
        
        if embedding_models:
            print("Embedding + 分類頭方法:")
            for name, sentiment, conf in embedding_models:
                print(f"   {name}: {sentiment} ({conf:.4f})")
        
        if lora_models:
            print("LoRA微調方法:")
            for name, sentiment, conf in lora_models:
                print(f"   {name}: {sentiment} ({conf:.4f})")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Qwen3微博情感分析統一預測接口')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='模型文件目錄')
    parser.add_argument('--model_type', type=str, choices=['embedding', 'lora'],
                        help='指定模型類型')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'],
                        help='指定模型大小')
    parser.add_argument('--text', type=str,
                        help='直接預測指定文本')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='交互式預測模式（默認）')
    parser.add_argument('--ensemble', action='store_true',
                        help='使用集成預測')
    parser.add_argument('--load_all', action='store_true',
                        help='加載所有可用模型')
    
    args = parser.parse_args()
    
    # 創建預測器
    predictor = Qwen3UniversalPredictor()
    
    # 加載模型
    if args.load_all:
        # 加載所有模型
        predictor.load_all_models(args.model_dir)
    elif args.model_type and args.model_size:
        # 加載指定模型
        predictor.load_model(args.model_type, args.model_size)
    # 如果沒有指定模型，交互式模式會讓用戶選擇
    
    # 如果指定了文本，直接預測
    if args.text:
        if args.ensemble and len(predictor.models) > 1:
            pred, conf = predictor.ensemble_predict(args.text)
            sentiment = "正面" if pred == 1 else "負面"
            print(f"文本: {args.text}")
            print(f"集成預測: {sentiment} (置信度: {conf:.4f})")
        else:
            results = predictor.predict_single(args.text)
            print(f"文本: {args.text}")
            for model_name, (pred, conf) in results.items():
                sentiment = "正面" if pred == 1 else "負面"
                print(f"{model_name}: {sentiment} (置信度: {conf:.4f})")
    else:
        # 進入交互式模式
        predictor.interactive_predict()


if __name__ == "__main__":
    main()
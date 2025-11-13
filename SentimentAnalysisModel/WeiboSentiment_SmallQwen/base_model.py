# -*- coding: utf-8 -*-
"""
Qwen3模型基礎類，統一接口
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


class BaseQwenModel(ABC):
    """Qwen3情感分析模型基類"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """訓練模型"""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """預測文本情感"""
        pass
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """預測單條文本的情感
        
        Args:
            text: 待預測文本
            
        Returns:
            (predicted_label, confidence)
        """
        predictions = self.predict([text])
        return predictions[0], 0.0  # 默認置信度爲0
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> Dict[str, float]:
        """評估模型性能"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，請先調用train方法")
            
        texts = [item[0] for item in test_data]
        labels = [item[1] for item in test_data]
        
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        print(f"\n{self.model_name} 模型評估結果:")
        print(f"準確率: {accuracy:.4f}")
        print(f"F1分數: {f1:.4f}")
        print("\n詳細報告:")
        print(classification_report(labels, predictions))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(labels, predictions)
        }
    
    @abstractmethod
    def save_model(self, model_path: str = None) -> None:
        """保存模型到文件"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """從文件加載模型"""
        pass
    
    @staticmethod
    def load_data(train_path: str = None, test_path: str = None, csv_path: str = 'dataset/weibo_senti_100k.csv') -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """加載訓練和測試數據
        
        Args:
            train_path: 訓練數據txt文件路徑（可選）
            test_path: 測試數據txt文件路徑（可選）
            csv_path: CSV數據文件路徑（默認使用）
        """
        
        # 優先嚐試使用CSV文件
        if os.path.exists(csv_path):
            print(f"從CSV文件加載數據: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # 檢查數據格式
            if 'review' in df.columns and 'label' in df.columns:
                # 將DataFrame轉換爲元組列表
                data = [(row['review'], row['label']) for _, row in df.iterrows()]
                
                # 分割訓練和測試數據，固定測試集爲5000條
                total_samples = len(data)
                if total_samples > 5000:
                    test_size = 5000
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=test_size, 
                        random_state=42, 
                        stratify=[label for _, label in data]
                    )
                else:
                    # 如果總數據不足5000條，使用20%作爲測試集
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=[label for _, label in data]
                    )
                
                print(f"訓練數據量: {len(train_data)}")
                print(f"測試數據量: {len(test_data)}")
                
                return train_data, test_data
            else:
                print(f"CSV文件格式不正確，缺少'review'或'label'列")
        
        # 如果CSV不存在，嘗試使用txt文件
        elif train_path and test_path and os.path.exists(train_path) and os.path.exists(test_path):
            def load_corpus(path):
                data = []
                with open(path, "r", encoding="utf8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            content = parts[0]
                            sentiment = int(parts[1])
                            data.append((content, sentiment))
                return data
            
            print("從txt文件加載訓練數據...")
            train_data = load_corpus(train_path)
            print(f"訓練數據量: {len(train_data)}")
            
            print("從txt文件加載測試數據...")
            test_data = load_corpus(test_path)
            print(f"測試數據量: {len(test_data)}")
            
            return train_data, test_data
        
        else:
            # 如果都沒有，提供樣例數據創建指導
            print("未找到數據文件!")
            print("請確保以下文件之一存在:")
            print(f"1. CSV文件: {csv_path}")
            print(f"2. txt文件: {train_path} 和 {test_path}")
            print("\n數據格式要求:")
            print("CSV文件: 包含'review'和'label'列")
            print("txt文件: 每行格式爲'文本內容\\t標籤'")
            
            # 創建樣例數據
            sample_data = [
                ("今天天氣真好，心情很棒!", 1),
                ("這部電影太無聊了", 0),
                ("非常喜歡這個產品", 1),
                ("服務態度很差", 0),
                ("質量不錯，值得推薦", 1)
            ]
            
            print("使用樣例數據進行演示...")
            train_data = sample_data * 20  # 擴充樣例數據
            test_data = sample_data * 5
            
            return train_data, test_data
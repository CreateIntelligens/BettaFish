# -*- coding: utf-8 -*-
"""
基礎模型類，爲所有情感分析模型提供統一接口
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_corpus


class BaseModel(ABC):
    """情感分析模型基類"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
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
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型到文件"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，無法保存")
            
        if model_path is None:
            model_path = f"model/{self.model_name}_model.pkl"
            
        # 創建保存目錄
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型數據
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """從文件加載模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.vectorizer = model_data.get('vectorizer')
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        
        print(f"已加載模型: {model_path}")
    
    @staticmethod
    def load_data(train_path: str, test_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """加載訓練和測試數據"""
        print("加載訓練數據...")
        train_data = load_corpus(train_path)
        print(f"訓練數據量: {len(train_data)}")
        
        print("加載測試數據...")
        test_data = load_corpus(test_path)
        print(f"測試數據量: {len(test_data)}")
        
        return train_data, test_data
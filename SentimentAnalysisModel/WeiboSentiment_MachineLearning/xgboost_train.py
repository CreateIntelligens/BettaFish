# -*- coding: utf-8 -*-
"""
XGBoost情感分析模型訓練腳本
"""
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

from base_model import BaseModel
from utils import stopwords


class XGBoostModel(BaseModel):
    """XGBoost情感分析模型"""
    
    def __init__(self):
        super().__init__("XGBoost")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """訓練XGBoost模型
        
        Args:
            train_data: 訓練數據，格式爲[(text, label), ...]
            **kwargs: 其他參數，支持XGBoost的各種參數
        """
        print(f"開始訓練 {self.model_name} 模型...")
        
        # 準備數據
        df_train = pd.DataFrame(train_data, columns=["words", "label"])
        
        # 特徵編碼（詞袋模型，限制特徵數量）
        max_features = kwargs.get('max_features', 2000)
        print(f"構建詞袋模型 (max_features={max_features})...")
        self.vectorizer = CountVectorizer(
            token_pattern=r'\[?\w+\]?', 
            stop_words=stopwords,
            max_features=max_features
        )
        
        X_train = self.vectorizer.fit_transform(df_train["words"])
        y_train = df_train["label"]
        
        print(f"特徵維度: {X_train.shape[1]}")
        
        # XGBoost參數設置
        params = {
            'booster': kwargs.get('booster', 'gbtree'),
            'max_depth': kwargs.get('max_depth', 6),
            'scale_pos_weight': kwargs.get('scale_pos_weight', 0.5),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'eta': kwargs.get('eta', 0.3),
            'nthread': kwargs.get('nthread', 10),
        }
        
        num_boost_round = kwargs.get('num_boost_round', 200)
        
        print(f"訓練XGBoost分類器...")
        print(f"參數: {params}")
        print(f"迭代輪數: {num_boost_round}")
        
        # 創建DMatrix
        dmatrix = xgb.DMatrix(X_train, label=y_train)
        
        # 訓練模型
        self.model = xgb.train(params, dmatrix, num_boost_round=num_boost_round)
        
        self.is_trained = True
        print(f"{self.model_name} 模型訓練完成！")
        
    def predict(self, texts: List[str]) -> List[int]:
        """預測文本情感
        
        Args:
            texts: 待預測文本列表
            
        Returns:
            預測結果列表
        """
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，請先調用train方法")
            
        # 特徵轉換
        X = self.vectorizer.transform(texts)
        
        # 創建DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # 預測概率
        y_prob = self.model.predict(dmatrix)
        
        # 轉換爲類別標籤
        y_pred = (y_prob > 0.5).astype(int)
        
        return y_pred.tolist()
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """預測單條文本的情感
        
        Args:
            text: 待預測文本
            
        Returns:
            (predicted_label, confidence)
        """
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，請先調用train方法")
            
        # 特徵轉換
        X = self.vectorizer.transform([text])
        
        # 創建DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # 預測概率
        prob = self.model.predict(dmatrix)[0]
        
        # 轉換爲類別標籤和置信度
        prediction = int(prob > 0.5)
        confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, float(confidence)
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> dict:
        """評估模型性能，包含AUC指標"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，請先調用train方法")
            
        texts = [item[0] for item in test_data]
        labels = [item[1] for item in test_data]
        
        # 預測類別
        predictions = self.predict(texts)
        
        # 預測概率（用於計算AUC）
        X = self.vectorizer.transform(texts)
        dmatrix = xgb.DMatrix(X)
        probabilities = self.model.predict(dmatrix)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        auc = roc_auc_score(labels, probabilities)
        
        print(f"\n{self.model_name} 模型評估結果:")
        print(f"準確率: {accuracy:.4f}")
        print(f"F1分數: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='XGBoost情感分析模型訓練')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='訓練數據路徑')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='測試數據路徑')
    parser.add_argument('--model_path', type=str, default='./model/xgboost_model.pkl',
                        help='模型保存路徑')
    parser.add_argument('--max_features', type=int, default=2000,
                        help='最大特徵數量')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='XGBoost最大深度')
    parser.add_argument('--eta', type=float, default=0.3,
                        help='XGBoost學習率')
    parser.add_argument('--num_boost_round', type=int, default=200,
                        help='XGBoost迭代輪數')
    parser.add_argument('--eval_only', action='store_true',
                        help='僅評估已有模型，不進行訓練')
    
    args = parser.parse_args()
    
    # 創建模型
    model = XGBoostModel()
    
    if args.eval_only:
        # 僅評估模式
        print("評估模式：加載已有模型進行評估")
        model.load_model(args.model_path)
        
        # 加載測試數據
        _, test_data = BaseModel.load_data(args.train_path, args.test_path)
        
        # 評估模型
        model.evaluate(test_data)
    else:
        # 訓練模式
        # 加載數據
        train_data, test_data = BaseModel.load_data(args.train_path, args.test_path)
        
        # 訓練模型
        model.train(
            train_data,
            max_features=args.max_features,
            max_depth=args.max_depth,
            eta=args.eta,
            num_boost_round=args.num_boost_round
        )
        
        # 評估模型
        model.evaluate(test_data)
        
        # 保存模型
        model.save_model(args.model_path)
        
        # 示例預測
        print("\n示例預測:")
        test_texts = [
            "今天天氣真好，心情很棒",
            "這部電影太無聊了，浪費時間",
            "哈哈哈，太有趣了"
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "正面" if pred == 1 else "負面"
            print(f"文本: {text}")
            print(f"預測: {sentiment} (置信度: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()
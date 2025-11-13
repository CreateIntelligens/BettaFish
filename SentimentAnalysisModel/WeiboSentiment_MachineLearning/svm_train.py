# -*- coding: utf-8 -*-
"""
SVM情感分析模型訓練腳本
"""
import argparse
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

from base_model import BaseModel
from utils import stopwords


class SVMModel(BaseModel):
    """SVM情感分析模型"""
    
    def __init__(self):
        super().__init__("SVM")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """訓練SVM模型
        
        Args:
            train_data: 訓練數據，格式爲[(text, label), ...]
            **kwargs: 其他參數，支持kernel, C等SVM參數
        """
        print(f"開始訓練 {self.model_name} 模型...")
        
        # 準備數據
        df_train = pd.DataFrame(train_data, columns=["words", "label"])
        
        # 特徵編碼（TF-IDF模型）
        print("構建TF-IDF特徵...")
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'\[?\w+\]?', 
            stop_words=stopwords
        )
        
        X_train = self.vectorizer.fit_transform(df_train["words"])
        y_train = df_train["label"]
        
        print(f"特徵維度: {X_train.shape[1]}")
        
        # 獲取SVM參數
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 1.0)
        gamma = kwargs.get('gamma', 'scale')
        
        # 訓練模型
        print(f"訓練SVM分類器 (kernel={kernel}, C={C}, gamma={gamma})...")
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.model.fit(X_train, y_train)
        
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
        
        # 預測
        predictions = self.model.predict(X)
        
        return predictions.tolist()
    
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
        
        # 預測
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return int(prediction), float(confidence)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='SVM情感分析模型訓練')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='訓練數據路徑')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='測試數據路徑')
    parser.add_argument('--model_path', type=str, default='./model/svm_model.pkl',
                        help='模型保存路徑')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help='SVM核函數類型')
    parser.add_argument('--C', type=float, default=1.0,
                        help='SVM正則化參數C')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='SVM核函數參數gamma')
    parser.add_argument('--eval_only', action='store_true',
                        help='僅評估已有模型，不進行訓練')
    
    args = parser.parse_args()
    
    # 創建模型
    model = SVMModel()
    
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
        model.train(train_data, kernel=args.kernel, C=args.C, gamma=args.gamma)
        
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
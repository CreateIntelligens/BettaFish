# -*- coding: utf-8 -*-
"""
LSTM情感分析模型訓練腳本
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gensim import models
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from typing import List, Tuple, Dict, Any
import numpy as np

from base_model import BaseModel


class LSTMDataset(Dataset):
    """LSTM數據集"""
    
    def __init__(self, data: List[Tuple[str, int]], word2vec_model):
        self.data = []
        self.label = []
        
        for text, label in data:
            vectors = []
            for word in text.split(" "):
                if word in word2vec_model.wv.key_to_index:
                    vectors.append(word2vec_model.wv[word])
            
            if len(vectors) > 0:  # 確保有有效的詞向量
                vectors = torch.Tensor(vectors)
                self.data.append(vectors)
                self.label.append(label)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.label)


def collate_fn(data):
    """批處理函數"""
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    data = pad_sequence(x, batch_first=True, padding_value=0)
    return data, torch.tensor(y, dtype=torch.float32), data_length


class LSTMNet(nn.Module):
    """LSTM網絡結構"""
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 雙向LSTM
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths):
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        packed_input = pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        packed_out, (h_n, h_c) = self.lstm(packed_input, (h0, c0))
        
        # 雙向LSTM，拼接最後的隱藏狀態
        lstm_out = torch.cat([h_n[-2], h_n[-1]], 1)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out


class LSTMModel(BaseModel):
    """LSTM情感分析模型"""
    
    def __init__(self):
        super().__init__("LSTM")
        self.word2vec_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _train_word2vec(self, train_data: List[Tuple[str, int]], **kwargs):
        """訓練Word2Vec詞向量"""
        print("訓練Word2Vec詞向量...")
        
        # 準備Word2Vec輸入數據
        wv_input = [text.split(" ") for text, _ in train_data]
        
        vector_size = kwargs.get('vector_size', 64)
        min_count = kwargs.get('min_count', 1)
        epochs = kwargs.get('epochs', 1000)
        
        # 訓練Word2Vec
        self.word2vec_model = models.Word2Vec(
            wv_input,
            vector_size=vector_size,
            min_count=min_count,
            epochs=epochs
        )
        
        print(f"Word2Vec訓練完成，詞向量維度: {vector_size}")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """訓練LSTM模型"""
        print(f"開始訓練 {self.model_name} 模型...")
        
        # 訓練Word2Vec
        self._train_word2vec(train_data, **kwargs)
        
        # 超參數
        learning_rate = kwargs.get('learning_rate', 5e-4)
        num_epochs = kwargs.get('num_epochs', 5)
        batch_size = kwargs.get('batch_size', 100)
        embed_size = kwargs.get('embed_size', 64)
        hidden_size = kwargs.get('hidden_size', 64)
        num_layers = kwargs.get('num_layers', 2)
        
        print(f"LSTM超參數: lr={learning_rate}, epochs={num_epochs}, "
              f"batch_size={batch_size}, hidden_size={hidden_size}")
        
        # 創建數據集
        train_dataset = LSTMDataset(train_data, self.word2vec_model)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 collate_fn=collate_fn, shuffle=True)
        
        # 創建模型
        self.model = LSTMNet(embed_size, hidden_size, num_layers).to(self.device)
        
        # 損失函數和優化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 訓練循環
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i, (x, labels, lengths) in enumerate(train_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                # 前向傳播
                outputs = self.model(x, lengths)
                logits = outputs.view(-1)
                loss = criterion(logits, labels)
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {avg_loss:.4f}")
            
            # 保存每個epoch的模型
            if kwargs.get('save_each_epoch', False):
                epoch_model_path = f"./model/lstm_epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), epoch_model_path)
                print(f"已保存模型: {epoch_model_path}")
        
        self.is_trained = True
        print(f"{self.model_name} 模型訓練完成！")
    
    def predict(self, texts: List[str]) -> List[int]:
        """預測文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，請先調用train方法")
        
        # 創建數據集
        test_data = [(text, 0) for text in texts]  # 標籤無關緊要
        test_dataset = LSTMDataset(test_data, self.word2vec_model)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for x, _, lengths in test_loader:
                x = x.to(self.device)
                outputs = self.model(x, lengths)
                outputs = outputs.view(-1)
                
                # 轉換爲類別標籤
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """預測單條文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，請先調用train方法")
        
        # 轉換爲詞向量
        vectors = []
        for word in text.split(" "):
            if word in self.word2vec_model.wv.key_to_index:
                vectors.append(self.word2vec_model.wv[word])
        
        if len(vectors) == 0:
            return 0, 0.5  # 如果沒有有效詞向量，返回默認值
        
        # 轉換爲tensor
        x = torch.Tensor(vectors).unsqueeze(0).to(self.device)  # 添加batch維度
        lengths = [len(vectors)]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, lengths)
            prob = output.item()
            prediction = int(prob > 0.5)
            confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練，無法保存")
        
        if model_path is None:
            model_path = f"./model/{self.model_name.lower()}_model.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型狀態和Word2Vec
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'word2vec_model': self.word2vec_model,
            'model_config': {
                'embed_size': 64,
                'hidden_size': 64,
                'num_layers': 2
            },
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加載模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device)
        
        # 加載Word2Vec
        self.word2vec_model = model_data['word2vec_model']
        
        # 重建LSTM網絡
        config = model_data['model_config']
        self.model = LSTMNet(
            config['embed_size'],
            config['hidden_size'],
            config['num_layers']
        ).to(self.device)
        
        # 加載模型權重
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.is_trained = True
        print(f"已加載模型: {model_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='LSTM情感分析模型訓練')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='訓練數據路徑')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='測試數據路徑')
    parser.add_argument('--model_path', type=str, default='./model/lstm_model.pth',
                        help='模型保存路徑')
    parser.add_argument('--epochs', type=int, default=5,
                        help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批大小')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM隱藏層大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='學習率')
    parser.add_argument('--eval_only', action='store_true',
                        help='僅評估已有模型，不進行訓練')
    
    args = parser.parse_args()
    
    # 創建模型
    model = LSTMModel()
    
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
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate
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
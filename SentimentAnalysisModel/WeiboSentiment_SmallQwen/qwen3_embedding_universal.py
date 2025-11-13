# -*- coding: utf-8 -*-
"""
Qwen3-Embedding通用訓練腳本
支持0.6B、4B、8B三種規模的模型
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import warnings
from tqdm import tqdm

from base_model import BaseQwenModel
from models_config import QWEN3_MODELS, MODEL_PATHS

warnings.filterwarnings("ignore")


class SentimentDataset(Dataset):
    """情感分析數據集"""
    
    def __init__(self, data: List[Tuple[str, int]], tokenizer, max_length=512):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SentimentClassifier(nn.Module):
    """情感分類器"""
    
    def __init__(self, embedding_model, embedding_dim, hidden_dim=256):
        super(SentimentClassifier, self).__init__()
        self.embedding_model = embedding_model
        
        # 凍結embedding模型參數
        for param in self.embedding_model.parameters():
            param.requires_grad = False
            
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        # 獲取embedding
        with torch.no_grad():
            outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 通過分類頭
        logits = self.classifier(embeddings)
        return logits.squeeze()


class Qwen3EmbeddingUniversal(BaseQwenModel):
    """通用Qwen3-Embedding模型"""
    
    def __init__(self, model_size: str = "0.6B"):
        if model_size not in QWEN3_MODELS:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        super().__init__(f"Qwen3-Embedding-{model_size}")
        self.model_size = model_size
        self.config = QWEN3_MODELS[model_size]
        self.model_name_hf = self.config["embedding_model"]
        self.embedding_dim = self.config["embedding_dim"]
        
        self.tokenizer = None
        self.embedding_model = None
        self.classifier_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_embedding_model(self):
        """加載Qwen3 Embedding模型"""
        print(f"加載{self.model_size}模型: {self.model_name_hf}")
        
        # 第一步：檢查當前文件夾的models目錄
        local_model_dir = f"./models/qwen3-embedding-{self.model_size.lower()}"
        if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            try:
                print(f"發現本地模型，從本地加載: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.embedding_model = AutoModel.from_pretrained(local_model_dir).to(self.device)
                print(f"從本地模型加載{self.model_size}模型成功")
                return
                
            except Exception as e:
                print(f"本地模型加載失敗: {e}")
        
        # 第二步：檢查HuggingFace緩存
        try:
            from transformers.utils import default_cache_path
            cache_path = default_cache_path
            print(f"檢查HuggingFace緩存: {cache_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.embedding_model = AutoModel.from_pretrained(self.model_name_hf).to(self.device)
            print(f"從HuggingFace緩存加載{self.model_size}模型成功")
            
            # 保存到本地models目錄
            print(f"保存模型到本地: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_model_dir)
            self.embedding_model.save_pretrained(local_model_dir)
            print(f"模型已保存到: {local_model_dir}")
            
        except Exception as e:
            print(f"從HuggingFace緩存加載失敗: {e}")
            
            # 第三步：從HuggingFace下載
            try:
                print(f"正在從HuggingFace下載{self.model_size}模型...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                )
                self.embedding_model = AutoModel.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                ).to(self.device)
                
                # 保存到本地models目錄
                os.makedirs(local_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_dir)
                self.embedding_model.save_pretrained(local_model_dir)
                print(f"{self.model_size}模型下載並保存到: {local_model_dir}")
                
            except Exception as e2:
                print(f"從HuggingFace下載也失敗: {e2}")
                raise RuntimeError(f"無法加載{self.model_size}模型，所有方法都失敗了")
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """訓練模型"""
        print(f"開始訓練 Qwen3-Embedding-{self.model_size} 模型...")
        
        # 加載embedding模型
        self._load_embedding_model()
        
        # 超參數（使用配置文件的推薦值或用戶指定值）
        batch_size = kwargs.get('batch_size', self.config['recommended_batch_size'])
        learning_rate = kwargs.get('learning_rate', self.config['recommended_lr'])
        num_epochs = kwargs.get('num_epochs', 5)
        max_length = kwargs.get('max_length', 512)
        
        print(f"超參數: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
        print(f"嵌入維度: {self.embedding_dim}")
        
        # 創建數據集
        train_dataset = SentimentDataset(train_data, self.tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 創建分類器
        self.classifier_model = SentimentClassifier(
            self.embedding_model, 
            self.embedding_dim
        ).to(self.device)
        
        # 損失函數和優化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.classifier_model.classifier.parameters(), lr=learning_rate)
        
        # 訓練循環
        self.classifier_model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向傳播
                outputs = self.classifier_model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': total_loss / num_batches})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        self.model = self.classifier_model
        self.is_trained = True
        print(f"Qwen3-Embedding-{self.model_size} 模型訓練完成！")
    
    def predict(self, texts: List[str]) -> List[int]:
        """預測文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")
        
        predictions = []
        batch_size = 32
        
        self.classifier_model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.classifier_model(input_ids, attention_mask)
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """預測單條文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")
        
        self.classifier_model.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            output = self.classifier_model(input_ids, attention_mask)
            prob = output.item()
            prediction = int(prob > 0.5)
            confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")
        
        if model_path is None:
            model_path = MODEL_PATHS["embedding"][self.model_size]
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'classifier_state_dict': self.classifier_model.classifier.state_dict(),
            'model_size': self.model_size,
            'model_name_hf': self.model_name_hf,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加載模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加載模型數據
        model_data = torch.load(model_path, map_location=self.device)
        
        # 驗證模型大小匹配
        if model_data['model_size'] != self.model_size:
            raise ValueError(f"模型大小不匹配: 期望{self.model_size}, 實際{model_data['model_size']}")
        
        # 加載embedding模型
        self._load_embedding_model()
        
        # 重建分類器
        self.classifier_model = SentimentClassifier(
            self.embedding_model, 
            model_data['embedding_dim']
        ).to(self.device)
        self.classifier_model.classifier.load_state_dict(model_data['classifier_state_dict'])
        
        self.model = self.classifier_model
        self.is_trained = True
        print(f"已加載Qwen3-Embedding-{self.model_size}模型: {model_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Qwen3-Embedding通用訓練腳本')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'], 
                        help='模型大小')
    parser.add_argument('--train_path', type=str, default='./dataset/train.txt',
                        help='訓練數據路徑')
    parser.add_argument('--test_path', type=str, default='./dataset/test.txt',
                        help='測試數據路徑')
    parser.add_argument('--model_path', type=str, help='模型保存路徑（可選）')
    parser.add_argument('--epochs', type=int, default=5, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, help='批大小（可選，使用推薦值）')
    parser.add_argument('--learning_rate', type=float, help='學習率（可選，使用推薦值）')
    parser.add_argument('--eval_only', action='store_true', help='僅評估模式')
    
    args = parser.parse_args()
    
    # 如果沒有指定模型大小，則詢問用戶
    if not args.model_size:
        print("Qwen3-Embedding模型訓練")
        print("="*40)
        print("可用模型大小:")
        print("  1. 0.6B - 輕量級，訓練快速，顯存需求約4GB")
        print("  2. 4B  - 中等規模，性能均衡，顯存需求約16GB") 
        print("  3. 8B  - 大規模，性能最佳，顯存需求約32GB")
        
        while True:
            choice = input("\n請選擇模型大小 (1/2/3): ").strip()
            if choice == '1':
                args.model_size = '0.6B'
                break
            elif choice == '2':
                args.model_size = '4B'
                break
            elif choice == '3':
                args.model_size = '8B'
                break
            else:
                print("無效選擇，請輸入 1、2 或 3")
        
        print(f"已選擇: Qwen3-Embedding-{args.model_size}")
        print()
    
    # 確保models目錄存在
    os.makedirs('./models', exist_ok=True)
    
    # 創建模型
    model = Qwen3EmbeddingUniversal(args.model_size)
    
    # 確定模型保存路徑
    model_path = args.model_path or MODEL_PATHS["embedding"][args.model_size]
    
    if args.eval_only:
        # 僅評估模式
        print(f"評估模式：加載Qwen3-Embedding-{args.model_size}模型")
        model.load_model(model_path)
        
        _, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        model.evaluate(test_data)
    else:
        # 訓練模式
        train_data, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        
        # 準備訓練參數
        train_kwargs = {'num_epochs': args.epochs}
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        
        # 訓練模型
        model.train(train_data, **train_kwargs)
        
        # 評估模型
        model.evaluate(test_data)
        
        # 保存模型
        model.save_model(model_path)
        
        # 示例預測
        print(f"\nQwen3-Embedding-{args.model_size} 示例預測:")
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
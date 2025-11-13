import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2ForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from adapter import AdapterLayer
from gpt2_adapter import GPT2BlockWithAdapter

# 設置隨機種子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 定義微博情感分析數據集
class WeiboSentimentDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            review,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定義GPT2分類模型，帶Adapter
class GPT2ClassifierWithAdapter(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=2):
        super(GPT2ClassifierWithAdapter, self).__init__()
        # 加載預訓練模型
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels
        )
        
        # 確保模型配置中設置了pad_token_id
        self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        
        # 替換原始的GPT2Block爲帶Adapter的版本
        config = self.gpt2.config
        for i in range(len(self.gpt2.transformer.h)):
            # 保存原始權重
            old_block = self.gpt2.transformer.h[i]
            # 創建帶Adapter的新Block
            new_block = GPT2BlockWithAdapter(config)
            # 複製原始權重
            new_block.load_state_dict(old_block.state_dict(), strict=False)
            # 替換
            self.gpt2.transformer.h[i] = new_block
            
        # 凍結原始GPT2參數
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        # 解凍分類器層和Adapter層參數
        for param in self.gpt2.score.parameters():
            param.requires_grad = True
            
        # 解凍所有Adapter層
        for i in range(len(self.gpt2.transformer.h)):
            for param in self.gpt2.transformer.h[i].adapter.parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# 訓練函數
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=3):
    best_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"======== Epoch {epoch+1} / {epochs} ========")
        model.train()
        total_loss = 0
        
        # 訓練循環
        progress_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for batch in progress_bar:
            # 將數據移到GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 參數更新
            optimizer.step()
            scheduler.step()
            
            # 更新進度條
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 計算平均訓練損失
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 評估模型
        val_metrics = evaluate_model(model, val_dataloader, device)
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation F1 Score: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "best_weibo_sentiment_model.pth")
            print("Saved best model!")

# 評估函數
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 獲取預測結果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 計算評估指標
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    # 設置模型本地保存路徑
    model_name = 'uer/gpt2-chinese-cluecorpussmall'
    local_model_path = './models/gpt2-chinese'
    
    # 確保目錄存在
    os.makedirs(local_model_path, exist_ok=True)
    
    # 加載數據集
    print("加載微博情感數據集...")
    df = pd.read_csv('dataset/weibo_senti_100k.csv')
    
    # 分割數據集
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # 加載tokenizer和模型
    print("加載預訓練模型和tokenizer...")
    
    # 檢查本地是否已有模型
    if os.path.exists(os.path.join(local_model_path, 'config.json')):
        print(f"從本地路徑加載模型: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
    else:
        print(f"從Hugging Face下載模型到: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=local_model_path)
        # 保存tokenizer到本地
        tokenizer.save_pretrained(local_model_path)
    
    # 設置padding token (BertTokenizer通常已有[PAD]作爲padding token)
    if tokenizer.pad_token is None:
        # 如果沒有，顯式設置爲[PAD]
        tokenizer.pad_token = '[PAD]'
    
    # 記錄pad_token的ID，確保模型和tokenizer使用相同的pad_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # 創建數據集
    train_dataset = WeiboSentimentDataset(
        train_df['review'].values,
        train_df['label'].values,
        tokenizer
    )
    
    val_dataset = WeiboSentimentDataset(
        val_df['review'].values,
        val_df['label'].values,
        tokenizer
    )
    
    # 創建數據加載器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 初始化模型
    if (os.path.exists(os.path.join(local_model_path, 'pytorch_model.bin')) or 
        os.path.exists(os.path.join(local_model_path, 'model.safetensors'))):
        print(f"從本地路徑加載模型權重: {local_model_path}")
        model = GPT2ClassifierWithAdapter(local_model_path)
    else:
        print(f"從Hugging Face下載模型權重到: {local_model_path}")
        # 直接從Hugging Face下載並保存完整模型
        temp_model = GPT2ForSequenceClassification.from_pretrained(model_name)
        temp_model.save_pretrained(local_model_path)
        # 然後用保存的模型創建GPT2ClassifierWithAdapter
        model = GPT2ClassifierWithAdapter(local_model_path)
    
    # 確保模型使用與tokenizer相同的pad_token_id
    model.gpt2.config.pad_token_id = pad_token_id
    model.to(device)
    
    # 統計需要訓練的參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型總參數量: {total_params}")
    print(f"需要訓練的參數量: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    
    # 設置優化器和學習率調度器
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        eps=1e-8
    )
    
    # 設置總訓練步數和warmup步數
    total_steps = len(train_dataloader) * 2  # 2個epoch
    warmup_steps = int(total_steps * 0.1)  # 10%的warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 訓練模型
    print("開始訓練...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=2
    )
    
    print("訓練完成!")

if __name__ == "__main__":
    main() 
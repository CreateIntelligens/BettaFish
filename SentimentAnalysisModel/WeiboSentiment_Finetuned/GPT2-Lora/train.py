import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2ForSequenceClassification, 
    BertTokenizer, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 導入PEFT庫中的LoRA相關組件
from peft import LoraConfig, TaskType, get_peft_model

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
            # 保存LoRA權重
            model.save_pretrained("./best_weibo_sentiment_lora")
            print("Saved best LoRA model!")

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
    os.makedirs('./best_weibo_sentiment_lora', exist_ok=True)
    
    # 加載數據集
    print("加載微博情感數據集...")
    df = pd.read_csv('dataset/weibo_senti_100k.csv')
    
    # 分割數據集
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # 加載tokenizer
    print("加載預訓練模型和tokenizer...")
    
    # 檢查本地是否已有模型
    if os.path.exists(os.path.join(local_model_path, 'config.json')):
        print(f"從本地路徑加載tokenizer: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(local_model_path)
    else:
        print(f"從Hugging Face下載tokenizer到: {local_model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=local_model_path)
        # 保存tokenizer到本地
        tokenizer.save_pretrained(local_model_path)
    
    # 設置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
    
    # 記錄pad_token的ID
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
    
    # 加載預訓練的GPT2模型
    print("加載GPT2模型...")
    if (os.path.exists(os.path.join(local_model_path, 'pytorch_model.bin')) or 
        os.path.exists(os.path.join(local_model_path, 'model.safetensors'))):
        print(f"從本地路徑加載模型權重: {local_model_path}")
        model = GPT2ForSequenceClassification.from_pretrained(local_model_path, num_labels=2)
    else:
        print(f"從Hugging Face下載模型權重到: {local_model_path}")
        # 直接從Hugging Face下載並保存完整模型
        model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.save_pretrained(local_model_path)
    
    # 確保模型使用與tokenizer相同的pad_token_id
    model.config.pad_token_id = pad_token_id
    
    # 配置LoRA參數
    print("配置LoRA參數...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 序列分類任務
        target_modules=["c_attn", "c_proj"],  # GPT2的注意力投影層
        inference_mode=False,  # 訓練模式
        r=8,  # LoRA秩，控制可訓練參數數量
        lora_alpha=32,  # LoRA alpha參數，縮放因子
        lora_dropout=0.1,  # LoRA Dropout
    )
    
    # 將模型轉換爲PEFT格式的LoRA模型
    print("創建LoRA模型...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可訓練參數數量和佔比
    
    model.to(device)
    
    # 設置優化器和學習率調度器
    print("設置優化器...")
    optimizer = AdamW(
        model.parameters(),  # PEFT會自動處理參數篩選
        lr=5e-4,  # LoRA通常使用較高的學習率
        eps=1e-8
    )
    
    # 設置總訓練步數和warmup步數
    total_steps = len(train_dataloader) * 3  # 3個epoch
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
        epochs=3
    )
    
    print("訓練完成!")
    print("LoRA權重已保存到: ./best_weibo_sentiment_lora/")

if __name__ == "__main__":
    main()
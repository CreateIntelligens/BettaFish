import torch
from transformers import BertTokenizer
from train import GPT2ClassifierWithAdapter
import re

def preprocess_text(text):
    """簡單的文本預處理"""
    return text

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 使用本地模型路徑而不是在線模型名稱
    local_model_path = './models/gpt2-chinese'
    model_path = 'best_weibo_sentiment_model.pth'
    
    print(f"加載模型: {model_path}")
    # 從本地加載tokenizer
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
    
    # 加載模型，使用本地模型路徑
    model = GPT2ClassifierWithAdapter(local_model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n============= 微博情感分析 =============")
    print("輸入微博內容進行分析 (輸入 'q' 退出):")
    
    while True:
        text = input("\n請輸入微博內容: ")
        if text.lower() == 'q':
            break
        
        # 預處理文本
        processed_text = preprocess_text(text)
        
        # 對文本進行編碼
        encoding = tokenizer(
            processed_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 轉移到設備
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 預測
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # 輸出結果
        confidence = probabilities[0][prediction].item()
        label = "正面情感" if prediction == 1 else "負面情感"
        
        print(f"預測結果: {label} (置信度: {confidence:.4f})")

if __name__ == "__main__":
    main() 
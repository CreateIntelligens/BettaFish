import torch
from transformers import GPT2ForSequenceClassification, BertTokenizer
from peft import PeftModel
import os
import re

def preprocess_text(text):
    return text

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 模型和權重路徑
    base_model_path = './models/gpt2-chinese'
    lora_model_path = './best_weibo_sentiment_lora'
    
    print("加載模型和tokenizer...")
    
    # 檢查LoRA模型是否存在
    if not os.path.exists(lora_model_path):
        print(f"錯誤: 找不到LoRA模型路徑 {lora_model_path}")
        print("請先運行 train.py 進行訓練")
        return
    
    # 加載tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '[PAD]'
    except Exception as e:
        print(f"加載tokenizer失敗: {e}")
        print("請確保models/gpt2-chinese目錄包含tokenizer文件")
        return
    
    # 加載基礎模型
    try:
        base_model = GPT2ForSequenceClassification.from_pretrained(
            base_model_path, 
            num_labels=2
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        print(f"加載基礎模型失敗: {e}")
        print("請確保models/gpt2-chinese目錄包含模型文件")
        return
    
    # 加載LoRA權重
    try:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.to(device)
        model.eval()
        print("LoRA模型加載成功!")
    except Exception as e:
        print(f"加載LoRA權重失敗: {e}")
        print("請確保LoRA權重文件存在且格式正確")
        return
    
    print("\n============= 微博情感分析 (LoRA版) =============")
    print("輸入微博內容進行分析 (輸入 'q' 退出):")
    
    while True:
        text = input("\n請輸入微博內容: ")
        if text.lower() == 'q':
            break
        
        if not text.strip():
            print("輸入不能爲空，請重新輸入")
            continue
        
        try:
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
            
        except Exception as e:
            print(f"預測時發生錯誤: {e}")
            continue

if __name__ == "__main__":
    main()
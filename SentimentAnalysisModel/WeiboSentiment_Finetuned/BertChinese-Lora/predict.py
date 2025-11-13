import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

def preprocess_text(text):
    return text

def main():
    print("正在加載微博情感分析模型...")
    
    # 使用HuggingFace預訓練模型
    model_name = "wsqstar/GISchat-weibo-100k-fine-tuned-bert"
    local_model_path = "./model"
    
    try:
        # 檢查本地是否已有模型
        import os
        if os.path.exists(local_model_path):
            print("從本地加載模型...")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            print("首次使用，正在下載模型到本地...")
            # 下載並保存到本地
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # 保存到本地
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print(f"模型已保存到: {local_model_path}")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"模型加載成功! 使用設備: {device}")
        
    except Exception as e:
        print(f"模型加載失敗: {e}")
        print("請檢查網絡連接或使用pipeline方式")
        return
    
    print("\n============= 微博情感分析 =============")
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
            
            # 分詞編碼
            inputs = tokenizer(
                processed_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # 轉移到設備
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 預測
            with torch.no_grad():
                outputs = model(**inputs)
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
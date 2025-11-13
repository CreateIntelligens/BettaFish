from transformers import pipeline
import re

def preprocess_text(text):
    """簡單的文本預處理"""
    text = re.sub(r"\{%.+?%\}", " ", text)           # 去除 {%xxx%}
    text = re.sub(r"@.+?( |$)", " ", text)           # 去除 @xxx
    text = re.sub(r"【.+?】", " ", text)              # 去除 【xx】
    text = re.sub(r"\u200b", " ", text)              # 去除特殊字符
    # 刪除表情符號
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f018-\U0001f270\U0000231a-\U0000231b\U0000238d-\U0000238d\U000024c2-\U0001f251]+', '', text)
    text = re.sub(r"\s+", " ", text)                 # 多個空格合併
    return text.strip()

def main():
    print("正在加載微博情感分析模型...")
    
    # 使用pipeline方式 - 更簡單
    model_name = "wsqstar/GISchat-weibo-100k-fine-tuned-bert"
    local_model_path = "./model"
    
    try:
        # 檢查本地是否已有模型
        import os
        if os.path.exists(local_model_path):
            print("從本地加載模型...")
            classifier = pipeline(
                "text-classification", 
                model=local_model_path,
                return_all_scores=True
            )
        else:
            print("首次使用，正在下載模型到本地...")
            # 先下載模型
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # 保存到本地
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print(f"模型已保存到: {local_model_path}")
            
            # 使用本地模型創建pipeline
            classifier = pipeline(
                "text-classification", 
                model=local_model_path,
                return_all_scores=True
            )
        print("模型加載成功!")
        
    except Exception as e:
        print(f"模型加載失敗: {e}")
        print("請檢查網絡連接")
        return
    
    print("\n============= 微博情感分析 (Pipeline版) =============")
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
            
            # 預測
            outputs = classifier(processed_text)
            
            # 解析結果
            positive_score = None
            negative_score = None
            
            for output in outputs[0]:
                if output['label'] == 'LABEL_1':  # 正面
                    positive_score = output['score']
                elif output['label'] == 'LABEL_0':  # 負面
                    negative_score = output['score']
            
            # 確定預測結果
            if positive_score > negative_score:
                label = "正面情感"
                confidence = positive_score
            else:
                label = "負面情感"
                confidence = negative_score
            
            print(f"預測結果: {label} (置信度: {confidence:.4f})")
            
        except Exception as e:
            print(f"預測時發生錯誤: {e}")
            continue

if __name__ == "__main__":
    main()
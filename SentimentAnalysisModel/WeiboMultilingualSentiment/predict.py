import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

def preprocess_text(text):
    """簡單的文本預處理，適用於多語言文本"""
    return text

def main():
    print("正在加載多語言情感分析模型...")
    
    # 使用多語言情感分析模型
    model_name = "tabularisai/multilingual-sentiment-analysis"
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
        
        # 情感標籤映射（5級分類）
        sentiment_map = {
            0: "非常負面", 1: "負面", 2: "中性", 3: "正面", 4: "非常正面"
        }
        
    except Exception as e:
        print(f"模型加載失敗: {e}")
        print("請檢查網絡連接")
        return
    
    print("\n============= 多語言情感分析 =============")
    print("支持語言: 中文、英文、西班牙文、阿拉伯文、日文、韓文等22種語言")
    print("情感等級: 非常負面、負面、中性、正面、非常正面")
    print("輸入文本進行分析 (輸入 'q' 退出):")
    print("輸入 'demo' 查看多語言示例")
    
    while True:
        text = input("\n請輸入文本: ")
        if text.lower() == 'q':
            break
        
        if text.lower() == 'demo':
            show_multilingual_demo(tokenizer, model, device, sentiment_map)
            continue
        
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
            label = sentiment_map[prediction]
            
            print(f"預測結果: {label} (置信度: {confidence:.4f})")
            
            # 顯示所有類別的概率
            print("詳細概率分佈:")
            for i, (label_name, prob) in enumerate(zip(sentiment_map.values(), probabilities[0])):
                print(f"  {label_name}: {prob:.4f}")
            
        except Exception as e:
            print(f"預測時發生錯誤: {e}")
            continue

def show_multilingual_demo(tokenizer, model, device, sentiment_map):
    """展示多語言情感分析示例"""
    print("\n=== 多語言情感分析示例 ===")
    
    demo_texts = [
        # 中文
        ("今天天氣真好，心情特別棒！", "中文"),
        ("這家餐廳的菜味道非常棒！", "中文"),
        ("服務態度太差了，很失望", "中文"),
        
        # 英文
        ("I absolutely love this product!", "英文"),
        ("The customer service was disappointing.", "英文"),
        ("The weather is fine, nothing special.", "英文"),
        
        # 日文
        ("このレストランの料理は本當に美味しいです！", "日文"),
        ("このホテルのサービスはがっかりしました。", "日文"),
        
        # 韓文
        ("이 가게의 케이크는 정말 맛있어요！", "韓文"),
        ("서비스가 너무 별로였어요。", "韓文"),
        
        # 西班牙文
        ("¡Me encanta cómo quedó la decoración!", "西班牙文"),
        ("El servicio fue terrible y muy lento.", "西班牙文"),
    ]
    
    for text, language in demo_texts:
        try:
            inputs = tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
            
            confidence = probabilities[0][prediction].item()
            label = sentiment_map[prediction]
            
            print(f"\n{language}: {text}")
            print(f"結果: {label} (置信度: {confidence:.4f})")
            
        except Exception as e:
            print(f"處理 {text} 時出錯: {e}")
    
    print("\n=== 示例結束 ===")
    
    '''
    正在加載多語言情感分析模型...
從本地加載模型...
模型加載成功! 使用設備: cuda

============= 多語言情感分析 =============
支持語言: 中文、英文、西班牙文、阿拉伯文、日文、韓文等22種語言
情感等級: 非常負面、負面、中性、正面、非常正面
輸入文本進行分析 (輸入 'q' 退出):
輸入 'demo' 查看多語言示例

請輸入文本: 我喜歡你
C:\Users\67093\.conda\envs\pytorch_python11\Lib\site-packages\transformers\models\distilbert\modeling_distilbert.py:401: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
預測結果: 正面 (置信度: 0.5204)
詳細概率分佈:
  非常負面: 0.0329
  負面: 0.0263
  中性: 0.1987
  正面: 0.5204
  非常正面: 0.2216

請輸入文本:
    '''

if __name__ == "__main__":
    main()
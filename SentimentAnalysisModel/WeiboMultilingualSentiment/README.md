# 多語言情感分析 - Multilingual Sentiment Analysis

本模塊使用HuggingFace上的多語言情感分析模型進行情感分析，支持22種語言。

## 模型信息

- **模型名稱**: tabularisai/multilingual-sentiment-analysis  
- **基礎模型**: distilbert-base-multilingual-cased
- **支持語言**: 22種語言，包括：
  - 中文 (中文)
  - English (英語)
  - Español (西班牙語)
  - 日本語 (日語)
  - 한국어 (韓語)
  - Français (法語)
  - Deutsch (德語)
  - Русский (俄語)
  - العربية (阿拉伯語)
  - हिन्दी (印地語)
  - Português (葡萄牙語)
  - Italiano (意大利語)
  - 等等...

- **輸出類別**: 5級情感分類
  - 非常負面 (Very Negative)
  - 負面 (Negative)
  - 中性 (Neutral)
  - 正面 (Positive)
  - 非常正面 (Very Positive)

## 快速開始

1. 確保已安裝依賴：
```bash
pip install transformers torch
```

2. 運行預測程序：
```bash
python predict.py
```

3. 輸入任意語言的文本進行分析：
```
請輸入文本: I love this product!
預測結果: 非常正面 (置信度: 0.9456)
```

4. 查看多語言示例：
```
請輸入文本: demo
```

## 代碼示例

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加載模型
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 預測
texts = [
    "今天心情很好",  # 中文
    "I love this!",  # 英文
    "¡Me encanta!"   # 西班牙文
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = {0: "非常負面", 1: "負面", 2: "中性", 3: "正面", 4: "非常正面"}
    print(f"{text} -> {sentiment_map[prediction]}")
```

## 特色功能

- **多語言支持**: 無需指定語言，自動識別22種語言
- **5級精細分類**: 比傳統二分類更細緻的情感分析
- **高精度**: 基於DistilBERT的先進架構
- **本地緩存**: 首次下載後保存到本地，加快後續使用

## 應用場景

- 國際社交媒體監控
- 多語言客戶反饋分析
- 全球產品評論情感分類
- 跨語言品牌情感追蹤
- 多語言客服優化
- 國際市場研究

## 模型存儲

- 首次運行時會自動下載模型到當前目錄的 `model` 文件夾
- 後續運行會直接從本地加載，無需重複下載
- 模型大小約135MB，首次下載需要網絡連接

## 文件說明

- `predict.py`: 主預測程序，使用直接模型調用
- `README.md`: 使用說明

## 注意事項

- 首次運行時會自動下載模型，需要網絡連接
- 模型會保存到當前目錄，方便後續使用
- 支持GPU加速，會自動檢測可用設備
- 如需清理模型文件，刪除 `model` 文件夾即可
- 該模型基於合成數據訓練，在實際應用中建議進行驗證
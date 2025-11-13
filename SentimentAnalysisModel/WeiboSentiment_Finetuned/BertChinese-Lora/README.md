# 微博情感分析 - 基於BertChinese的微調模型

本模塊使用HuggingFace上的預訓練微博情感分析模型進行情感分析。

## 模型信息

- **模型名稱**: wsqstar/GISchat-weibo-100k-fine-tuned-bert  
- **模型類型**: BERT中文情感分類模型
- **訓練數據**: 10萬條微博數據
- **輸出**: 二分類（正面/負面情感）

## 使用方法

### 方法1: 直接模型調用 (推薦)
```bash
python predict.py
```

### 方法2: Pipeline方式
```bash
python predict_pipeline.py
```

## 快速開始

1. 確保已安裝依賴：
```bash
pip install transformers torch
```

2. 運行預測程序：
```bash
python predict.py
```

3. 輸入微博文本進行分析：
```
請輸入微博內容: 今天天氣真好，心情特別棒！
預測結果: 正面情感 (置信度: 0.9234)
```

## 代碼示例

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加載模型
model_name = "wsqstar/GISchat-weibo-100k-fine-tuned-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 預測
text = "今天心情很好"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
print("正面情感" if prediction == 1 else "負面情感")
```

## 文件說明

- `predict.py`: 主預測程序，使用直接模型調用
- `predict_pipeline.py`: 使用pipeline方式的預測程序  
- `README.md`: 使用說明

## 模型存儲

- 首次運行時會自動下載模型到當前目錄的 `model` 文件夾
- 後續運行會直接從本地加載，無需重複下載
- 模型大小約400MB，首次下載需要網絡連接

## 注意事項

- 首次運行時會自動下載模型，需要網絡連接
- 模型會保存到當前目錄，方便後續使用
- 支持GPU加速，會自動檢測可用設備
- 如需清理模型文件，刪除 `model` 文件夾即可
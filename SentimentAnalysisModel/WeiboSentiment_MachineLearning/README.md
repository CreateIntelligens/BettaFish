# 微博情感分析 - 傳統機器學習方法

## 項目介紹

本項目使用5種傳統機器學習方法對中文微博進行情感二分類（正面/負面）：

- **樸素貝葉斯**: 基於詞袋模型的概率分類
- **SVM**: 基於TF-IDF特徵的支持向量機  
- **XGBoost**: 梯度提升決策樹
- **LSTM**: 循環神經網絡 + Word2Vec詞向量
- **BERT+分類頭**: 預訓練語言模型接分類器（我認爲也屬於傳統ML範疇）

## 模型性能

在微博情感數據集上的表現（訓練集10000條，測試集500條）：

| 模型 | 準確率 | AUC | 特點 |
|------|--------|-----|------|
| 樸素貝葉斯 | 85.6% | - | 速度快，內存佔用小 |
| SVM | 85.6% | - | 泛化能力好 |
| XGBoost | 86.0% | 90.4% | 性能穩定，支持特徵重要性 |
| LSTM | 87.0% | 93.1% | 理解序列信息和上下文 |
| BERT+分類頭 | 87.0% | 92.9% | 強大的語義理解能力 |

## 環境配置

```bash
pip install -r requirements.txt
```

數據文件結構：
```
data/
├── weibo2018/
│   ├── train.txt
│   └── test.txt
└── stopwords.txt
```

## 訓練模型（後面可以不接參數直接運行）

### 樸素貝葉斯
```bash
python bayes_train.py
```

### SVM
```bash
python svm_train.py --kernel rbf --C 1.0
```

### XGBoost
```bash
python xgboost_train.py --max_depth 6 --eta 0.3 --num_boost_round 200
```

### LSTM
```bash
python lstm_train.py --epochs 5 --batch_size 100 --hidden_size 64
```

### BERT
```bash
python bert_train.py --epochs 10 --batch_size 100 --learning_rate 1e-3
```

注：BERT模型會自動下載中文預訓練模型（bert-base-chinese）

## 使用預測

### 交互式預測（推薦）
```bash
python predict.py
```

### 命令行預測
```bash
# 單模型預測
python predict.py --model_type bert --text "今天天氣真好，心情很棒"

# 多模型集成預測
python predict.py --ensemble --text "這部電影太無聊了"
```

## 文件結構

```
WeiboSentiment_MachineLearning/
├── bayes_train.py           # 樸素貝葉斯訓練
├── svm_train.py             # SVM訓練
├── xgboost_train.py         # XGBoost訓練
├── lstm_train.py            # LSTM訓練
├── bert_train.py            # BERT訓練
├── predict.py               # 統一預測程序
├── base_model.py            # 基礎模型類
├── utils.py                 # 工具函數
├── requirements.txt         # 依賴包
├── model/                   # 模型保存目錄
└── data/                    # 數據目錄
```

## 注意事項

1. **BERT模型**首次運行會自動下載預訓練模型（約400MB）
2. **LSTM模型**訓練時間較長，建議使用GPU
3. **模型保存**在 `model/` 目錄下，確保有足夠磁盤空間
4. **內存需求**BERT > LSTM > XGBoost > SVM > 樸素貝葉斯

# -*- coding: utf-8 -*-
import jieba
import re
import os
import pickle
from typing import List, Tuple, Any


# 加載停用詞
stopwords = []
stopwords_path = "data/stopwords.txt"
if os.path.exists(stopwords_path):
    with open(stopwords_path, "r", encoding="utf8") as f:
        for w in f:
            stopwords.append(w.strip())
else:
    print(f"警告: 停用詞文件 {stopwords_path} 不存在，將使用空停用詞列表")


def load_corpus(path):
    """
    加載語料庫
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing(content)
            data.append((content, int(seniment)))
    return data


def load_corpus_bert(path):
    """
    加載語料庫
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing_bert(content)
            data.append((content, int(seniment)))
    return data


def processing(text):
    """
    數據預處理, 可以根據自己的需求進行重載
    """
    # 數據清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博話題等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用戶名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (裏面的內容通常都不是用戶自己寫的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是這個數據集中的一個bad case, 不用特別在意
    # 分詞
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    # 對否定詞`不`做特殊處理: 與其後面的詞進行拼接
    while "不" in words:
        index = words.index("不")
        if index == len(words) - 1:
            break
        words[index: index+2] = ["".join(words[index: index+2])]  # 列表切片賦值的酷炫寫法
    # 用空格拼接成字符串
    result = " ".join(words)
    return result


def processing_bert(text):
    """
    數據預處理, 可以根據自己的需求進行重載
    """
    # 數據清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博話題等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用戶名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (裏面的內容通常都不是用戶自己寫的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是這個數據集中的一個bad case, 不用特別在意
    return text


def save_model(model: Any, model_path: str) -> None:
    """
    保存模型到文件
    
    Args:
        model: 要保存的模型對象
        model_path: 保存路徑
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {model_path}")


def load_model(model_path: str) -> Any:
    """
    從文件加載模型
    
    Args:
        model_path: 模型文件路徑
        
    Returns:
        加載的模型對象
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"已加載模型: {model_path}")
    return model


def preprocess_text_simple(text: str) -> str:
    """
    簡單的文本預處理函數，用於預測時的文本清洗
    
    Args:
        text: 原始文本
        
    Returns:
        清洗後的文本
    """
    # 數據清洗
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%}
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】
    text = re.sub("\u200b", " ", text)              # 去除特殊字符
    
    # 刪除表情符號
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f018-\U0001f270\U0000231a-\U0000231b\U0000238d-\U0000238d\U000024c2-\U0001f251]+', '', text)
    
    # 多個空格合併爲一個
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()
# -*- coding: utf-8 -*-
"""
Qwen3-LoRA通用訓練腳本
支持0.6B、4B、8B三種規模的模型
"""
import argparse
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from typing import List, Tuple
import warnings
from tqdm import tqdm

from base_model import BaseQwenModel
from models_config import QWEN3_MODELS, MODEL_PATHS

warnings.filterwarnings("ignore")


class Qwen3LoRAUniversal(BaseQwenModel):
    """通用Qwen3-LoRA模型"""
    
    def __init__(self, model_size: str = "0.6B"):
        if model_size not in QWEN3_MODELS:
            raise ValueError(f"不支持的模型大小: {model_size}")
            
        super().__init__(f"Qwen3-{model_size}-LoRA")
        self.model_size = model_size
        self.config = QWEN3_MODELS[model_size]
        self.model_name_hf = self.config["base_model"]
        
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_base_model(self):
        """加載Qwen3基礎模型"""
        print(f"加載{self.model_size}基礎模型: {self.model_name_hf}")
        
        # 第一步：檢查當前文件夾的models目錄
        local_model_dir = f"./models/qwen3-{self.model_size.lower()}"
        if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            try:
                print(f"發現本地模型，從本地加載: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # 設置pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                print(f"從本地模型加載{self.model_size}基礎模型成功")
                return
                
            except Exception as e:
                print(f"本地模型加載失敗: {e}")
        
        # 第二步：檢查HuggingFace緩存
        try:
            from transformers.utils import default_cache_path
            cache_path = default_cache_path
            print(f"檢查HuggingFace緩存: {cache_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_hf,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 設置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print(f"從HuggingFace緩存加載{self.model_size}基礎模型成功")
            
            # 保存到本地models目錄
            print(f"保存模型到本地: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_model_dir)
            self.base_model.save_pretrained(local_model_dir)
            print(f"模型已保存到: {local_model_dir}")
            
        except Exception as e:
            print(f"從HuggingFace緩存加載失敗: {e}")
            
            # 第三步：從HuggingFace下載
            try:
                print(f"正在從HuggingFace下載{self.model_size}模型...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                )
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_hf,
                    force_download=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                # 保存到本地models目錄
                os.makedirs(local_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_dir)
                self.base_model.save_pretrained(local_model_dir)
                print(f"{self.model_size}模型下載並保存到: {local_model_dir}")
                
            except Exception as e2:
                print(f"從HuggingFace下載也失敗: {e2}")
                raise RuntimeError(f"無法加載{self.model_size}模型，所有方法都失敗了")
    
    def _create_instruction_data(self, data: List[Tuple[str, int]]) -> Dataset:
        """創建指令格式的訓練數據"""
        instructions = []
        
        for text, label in data:
            sentiment = "正面" if label == 1 else "負面"
            
            # 構建指令格式
            instruction = f"請分析以下微博文本的情感傾向，回答'正面'或'負面'。\n\n文本：{text}\n\n情感："
            response = sentiment
            
            
            # 組合成完整的訓練文本
            full_text = f"{instruction}{response}{self.tokenizer.eos_token}"
            
            instructions.append({
                "instruction": instruction,
                "response": response,
                "text": full_text
            })
        
        return Dataset.from_list(instructions)
    
    def _tokenize_function(self, examples):
        """分詞函數"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def _setup_lora(self, **kwargs):
        """設置LoRA配置"""
        lora_r = kwargs.get('lora_r', self.config['lora_r'])
        lora_alpha = kwargs.get('lora_alpha', self.config['lora_alpha'])
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        self.lora_model = get_peft_model(self.base_model, lora_config)
        
        # 統計參數
        total_params = sum(p.numel() for p in self.lora_model.parameters())
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        
        print(f"LoRA配置完成 (r={lora_r}, alpha={lora_alpha})")
        print(f"總參數: {total_params:,}")
        print(f"可訓練參數: {trainable_params:,}")
        print(f"可訓練參數比例: {trainable_params / total_params * 100:.2f}%")
        self.lora_model.print_trainable_parameters()  # PEFT庫自帶的參數統計
        
        return lora_config
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """訓練模型"""
        print(f"開始訓練 Qwen3-{self.model_size}-LoRA 模型...")
        
        # 加載基礎模型
        self._load_base_model()
        
        # 設置LoRA
        self._setup_lora(**kwargs)
        
        # 超參數（使用配置文件的推薦值或用戶指定值）
        num_epochs = kwargs.get('num_epochs', 3)
        batch_size = kwargs.get('batch_size', self.config['recommended_batch_size'] // 2)  # LoRA需要更少批大小
        learning_rate = kwargs.get('learning_rate', self.config['recommended_lr'] / 2)  # LoRA使用更小學習率
        output_dir = kwargs.get('output_dir', f'./models/qwen3_lora_{self.model_size.lower()}_checkpoints')
        
        print(f"超參數: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # 創建指令格式數據
        train_dataset = self._create_instruction_data(train_data)
        
        # 分詞
        tokenized_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # 訓練參數
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            report_to=None,
        )
        
        # 數據整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 創建訓練器
        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 開始訓練
        print(f"開始LoRA微調...")
        trainer.train()
        
        # 保存模型
        self.lora_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.model = self.lora_model
        self.is_trained = True
        print(f"Qwen3-{self.model_size}-LoRA 模型訓練完成！")
    
    def _extract_sentiment(self, generated_text: str, instruction: str) -> int:
        """從生成的文本中提取情感標籤"""
        response = generated_text[len(instruction):].strip()
        
        if "正面" in response:
            return 1
        elif "負面" in response:
            return 0
        else:
            return 0
    
    def predict(self, texts: List[str]) -> List[int]:
        """預測文本情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")
        
        predictions = []
        
        self.lora_model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc=f"Qwen3-{self.model_size}預測中"):
                pred, _ = self.predict_single(text)
                predictions.append(pred)
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """預測單條文本的情感"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")
        
        # 構建指令
        instruction = f"請分析以下微博文本的情感傾向，回答'正面'或'負面'。\n\n文本：{text}\n\n情感："
        
        # 分詞
        inputs = self.tokenizer(instruction, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回答
        self.lora_model.eval()
        with torch.no_grad():
            outputs = self.lora_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解碼生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取情感標籤
        prediction = self._extract_sentiment(generated_text, instruction)
        confidence = 0.8  # 生成式模型的置信度計算較複雜，這裏給個固定值
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError(f"模型 {self.model_name} 尚未訓練")
        
        if model_path is None:
            model_path = MODEL_PATHS["lora"][self.model_size]
        
        os.makedirs(model_path, exist_ok=True)
        
        # 保存LoRA權重
        self.lora_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"LoRA模型已保存到: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """加載模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加載基礎模型
        self._load_base_model()
        
        # 加載LoRA權重
        self.lora_model = PeftModel.from_pretrained(self.base_model, model_path)
        
        self.model = self.lora_model
        self.is_trained = True
        print(f"已加載Qwen3-{self.model_size}-LoRA模型: {model_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Qwen3-LoRA通用訓練腳本')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'], 
                        help='模型大小')
    parser.add_argument('--train_path', type=str, default='./dataset/train.txt',
                        help='訓練數據路徑')
    parser.add_argument('--test_path', type=str, default='./dataset/test.txt',
                        help='測試數據路徑')
    parser.add_argument('--model_path', type=str, help='模型保存路徑（可選）')
    parser.add_argument('--epochs', type=int, default=3, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, help='批大小（可選，使用推薦值）')
    parser.add_argument('--learning_rate', type=float, help='學習率（可選，使用推薦值）')
    parser.add_argument('--lora_r', type=int, help='LoRA秩（可選，使用推薦值）')
    parser.add_argument('--max_samples', type=int, default=0, help='最大訓練樣本數（0表示使用全部數據）')
    parser.add_argument('--eval_only', action='store_true', help='僅評估模式')
    
    args = parser.parse_args()
    
    # 如果沒有指定模型大小，則詢問用戶
    if not args.model_size:
        print("Qwen3-LoRA模型訓練")
        print("="*40)
        print("可用模型大小:")
        print("  1. 0.6B - 輕量級，訓練快速，顯存需求約8GB")
        print("  2. 4B  - 中等規模，性能均衡，顯存需求約32GB") 
        print("  3. 8B  - 大規模，性能最佳，顯存需求約64GB")
        print("\n注意: LoRA微調比Embedding方法需要更多顯存")
        
        while True:
            choice = input("\n請選擇模型大小 (1/2/3): ").strip()
            if choice == '1':
                args.model_size = '0.6B'
                break
            elif choice == '2':
                args.model_size = '4B'
                break
            elif choice == '3':
                args.model_size = '8B'
                break
            else:
                print("無效選擇，請輸入 1、2 或 3")
        
        print(f"已選擇: Qwen3-{args.model_size} + LoRA")
        print()
    
    # 確保models目錄存在
    os.makedirs('./models', exist_ok=True)
    
    # 創建模型
    model = Qwen3LoRAUniversal(args.model_size)
    
    # 確定模型保存路徑
    model_path = args.model_path or MODEL_PATHS["lora"][args.model_size]
    
    if args.eval_only:
        # 僅評估模式
        print(f"評估模式：加載Qwen3-{args.model_size}-LoRA模型")
        model.load_model(model_path)
        
        _, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        # LoRA評估使用少量數據
        test_subset = test_data[:50]
        model.evaluate(test_subset)
    else:
        # 訓練模式
        train_data, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        
        # 訓練數據處理
        if args.max_samples > 0:
            train_subset = train_data[:args.max_samples]
            print(f"使用 {len(train_subset)} 條數據進行LoRA訓練")
        else:
            train_subset = train_data
            print(f"使用全部 {len(train_subset)} 條數據進行LoRA訓練")
        
        # 準備訓練參數
        train_kwargs = {'num_epochs': args.epochs}
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        if args.lora_r:
            train_kwargs['lora_r'] = args.lora_r
        
        # 訓練模型
        model.train(train_subset, **train_kwargs)
        
        # 評估模型（使用少量測試數據）
        test_subset = test_data[:50]
        model.evaluate(test_subset)
        
        # 保存模型
        model.save_model(model_path)
        
        # 示例預測
        print(f"\nQwen3-{args.model_size}-LoRA 示例預測:")
        test_texts = [
            "今天天氣真好，心情很棒",
            "這部電影太無聊了，浪費時間",
            "哈哈哈，太有趣了"
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "正面" if pred == 1 else "負面"
            print(f"文本: {text}")
            print(f"預測: {sentiment} (置信度: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()
import os
import sys
import json
import re
import argparse
import math
import inspect
from typing import Dict, List, Optional, Tuple

# ========== 單卡鎖定（在導入 torch/transformers 前執行） ==========
def _extract_gpu_arg(argv: List[str], default: str = "0") -> str:
    for i, arg in enumerate(argv):
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
        if arg == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
    return default

env_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
try:
    gpu_to_use = _extract_gpu_arg(sys.argv, default="0")
except Exception:
    gpu_to_use = "0"
# 若未設置或暴露了多卡，則強制只暴露單卡（默認0）以確保直接運行穩定
if (not env_vis) or ("," in env_vis):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# 清理可能由外部啓動器注入的分佈式環境變量，避免誤觸多卡/分佈式
for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
    os.environ.pop(_k, None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
try:
    from transformers import EarlyStoppingCallback  # type: ignore
except Exception:  # pragma: no cover
    EarlyStoppingCallback = None  # type: ignore

# 預置可選中文基座模型（可擴展）
BACKBONE_CANDIDATES: List[Tuple[str, str]] = [
    ("1) google-bert/bert-base-chinese", "google-bert/bert-base-chinese"),
    ("2) hfl/chinese-roberta-wwm-ext-large", "hfl/chinese-roberta-wwm-ext-large"),
    ("3) hfl/chinese-macbert-large", "hfl/chinese-macbert-large"),
    ("4) IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese"),
    ("5) IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese", "IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese"),
    ("6) Langboat/mengzi-bert-base", "Langboat/mengzi-bert-base"),
    ("7) BAAI/bge-base-zh", "BAAI/bge-base-zh"),
    ("8) nghuyong/ernie-3.0-base-zh", "nghuyong/ernie-3.0-base-zh"),
]


def prompt_backbone_interactive(current_id: str) -> str:
    """交互式選擇基座模型。

    - 當處於非交互環境（stdin 非 TTY）或設置了環境變量 NON_INTERACTIVE=1 時，直接返回 current_id。
    - 用戶可輸入序號選擇預置項，或直接輸入任意 Hugging Face 模型 ID。
    - 空回車使用當前默認。
    """
    if os.environ.get("NON_INTERACTIVE", "0") == "1":
        return current_id
    try:
        if not sys.stdin.isatty():
            return current_id
    except Exception:
        return current_id

    print("\n可選中文基座模型（直接回車使用默認）:")
    for label, hf_id in BACKBONE_CANDIDATES:
        print(f"  {label}")
    print(f"當前默認: {current_id}")
    choice = input("請輸入序號或直接粘貼模型ID（回車沿用默認）: ").strip()
    if not choice:
        return current_id
    # 數字選項
    if choice.isdigit():
        idx = int(choice)
        for label, hf_id in BACKBONE_CANDIDATES:
            if label.startswith(f"{idx})"):
                return hf_id
        print("未找到該序號，沿用默認。")
        return current_id
    # 自定義 HF 模型 ID
    return choice


def preprocess_text(text: str) -> str:
    return from django.utils.translation import ungettext


def ensure_base_model_local(model_name_or_path: str, local_model_root: str) -> Tuple[str, AutoTokenizer]:
    os.makedirs(local_model_root, exist_ok=True)
    base_dir = os.path.join(local_model_root, "bert-base-chinese")

    def is_ready(path: str) -> bool:
        return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

    # 1) 本地現成
    if is_ready(base_dir):
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        return base_dir, tokenizer

    # 2) 本機緩存
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        base = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
        os.makedirs(base_dir, exist_ok=True)
        tokenizer.save_pretrained(base_dir)
        base.save_pretrained(base_dir)
        return base_dir, tokenizer
    except Exception:
        pass

    # 3) 遠程下載
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base = AutoModel.from_pretrained(model_name_or_path)
    os.makedirs(base_dir, exist_ok=True)
    tokenizer.save_pretrained(base_dir)
    base.save_pretrained(base_dir)
    return base_dir, tokenizer


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_column: str,
        label_column: str,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        text = preprocess_text(row[self.text_column])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.label_column in row and pd.notna(row[self.label_column]):
            label_str = str(row[self.label_column])
            item["labels"] = torch.tensor(self.label2id[label_str], dtype=torch.long)
        return item


def build_label_mappings(train_df: pd.DataFrame, label_column: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels: List[str] = [str(x) for x in train_df[label_column].dropna().astype(str).tolist()]
    unique_sorted = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_sorted)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def autodetect_columns(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[str, str]:
    if text_col != "auto" and label_col != "auto":
        return text_col, label_col
    candidates_text = ["text", "content", "sentence", "title", "desc", "question"]
    candidates_label = ["label", "labels", "category", "topic", "class"]
    t = text_col
    l = label_col
    if text_col == "auto":
        for name in candidates_text:
            if name in df.columns:
                t = name
                break
    if label_col == "auto":
        for name in candidates_label:
            if name in df.columns:
                l = name
                break
    if t == "auto" or l == "auto":
        raise ValueError(
            f"無法自動識別列名，請顯式傳入 --text_col 與 --label_col。現有列: {list(df.columns)}"
        )
    return t, l


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 google-bert/bert-base-chinese 在本目錄數據集上進行文本分類微調")
    parser.add_argument("--train_file", type=str, default="./dataset/web_text_zh_train.csv")
    parser.add_argument("--valid_file", type=str, default="./dataset/web_text_zh_valid.csv")
    parser.add_argument("--text_col", type=str, default="auto", help="文本列名，默認自動識別")
    parser.add_argument("--label_col", type=str, default="auto", help="標籤列名，默認自動識別")
    parser.add_argument("--model_root", type=str, default="./model", help="本地模型根目錄")
    parser.add_argument("--pretrained_name", type=str, default="google-bert/bert-base-chinese", help="Hugging Face 模型ID；留空則進入交互選擇")
    parser.add_argument("--save_subdir", type=str, default="bert-chinese-classifier")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="指定單卡 GPU，如 0 或 1")
    parser.add_argument("--eval_fraction", type=float, default=0.25, help="每多少個 epoch 做一次評估與保存，例如 0.25 表示每四分之一個 epoch")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="早停耐心（以評估輪次計）")
    parser.add_argument("--early_stop_threshold", type=float, default=0.0, help="早停最小改善閾值（與 metric_for_best_model 同單位）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = args.model_root if os.path.isabs(args.model_root) else os.path.join(script_dir, args.model_root)
    os.makedirs(model_root, exist_ok=True)

    # 交互式選擇基座模型（若允許交互且未通過環境禁用）
    selected_model_id = prompt_backbone_interactive(args.pretrained_name)
    # 確保基礎模型就緒
    base_dir, tokenizer = ensure_base_model_local(selected_model_id, model_root)
    print(f"[Info] 使用基礎模型目錄: {base_dir}")

    # 讀取數據
    train_path = args.train_file if os.path.isabs(args.train_file) else os.path.join(script_dir, args.train_file)
    valid_path = args.valid_file if os.path.isabs(args.valid_file) else os.path.join(script_dir, args.valid_file)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"訓練集不存在: {train_path}")
    train_df = pd.read_csv(train_path)
    if not os.path.isfile(valid_path):
        # 若未提供或不存在驗證集，自動切分
        shuffled = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * 0.9)
        valid_df = shuffled.iloc[split_idx:].reset_index(drop=True)
        train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    else:
        valid_df = pd.read_csv(valid_path)
    print(f"[Info] 訓練集: {train_path} | 樣本數: {len(train_df)}")
    print(f"[Info] 驗證集: {valid_path if os.path.isfile(valid_path) else '(從訓練集切分)'} | 樣本數: {len(valid_df)}")

    # 自動識別列名
    text_col, label_col = autodetect_columns(train_df, args.text_col, args.label_col)
    print(f"[Info] 文本列: {text_col} | 標籤列: {label_col}")

    # 標籤映射（使用 訓練集∪驗證集 的並集，避免驗證集中出現新標籤導致報錯）
    combined_labels_df = pd.concat([train_df[[label_col]], valid_df[[label_col]]], ignore_index=True)
    label2id, id2label = build_label_mappings(combined_labels_df, label_col)
    if len(label2id) < 2:
        raise ValueError("標籤類別數少於 2，無法訓練分類模型。")
    print(f"[Info] 標籤類別數: {len(label2id)}")
    # 提示驗證集中未出現在訓練集的標籤數量
    try:
        train_label_set = set(str(x) for x in train_df[label_col].dropna().astype(str).tolist())
        valid_label_set = set(str(x) for x in valid_df[label_col].dropna().astype(str).tolist())
        unseen_in_train = sorted(valid_label_set - train_label_set)
        if unseen_in_train:
            preview = ", ".join(unseen_in_train[:10])
            print(f"[Warn] 驗證集中存在 {len(unseen_in_train)} 個訓練未出現的標籤（已納入映射以避免報錯）。示例: {preview} ...")
    except Exception:
        pass

    # 數據集
    train_dataset = TextClassificationDataset(train_df, tokenizer, text_col, label_col, label2id, args.max_length)
    eval_dataset = TextClassificationDataset(valid_df, tokenizer, text_col, label_col, label2id, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 模型
    config = AutoConfig.from_pretrained(
        base_dir,
        num_labels=len(label2id),
        id2label={int(i): str(l) for i, l in id2label.items()},
        label2id={str(l): int(i) for l, i in label2id.items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_dir,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # 訓練參數
    output_dir = os.path.join(model_root, args.save_subdir)
    os.makedirs(output_dir, exist_ok=True)
    # 訓練參數（兼容不同 transformers 版本）
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_epochs,
        "logging_steps": 100,
        "fp16": args.fp16,
        "seed": args.seed,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    # 可選參數（僅在支持時添加，儘量簡化與參考實現一致以提升兼容性）
    if "warmup_ratio" in allowed:
        args_dict["warmup_ratio"] = args.warmup_ratio
    if "report_to" in allowed:
        args_dict["report_to"] = []
    # 評估/保存步進：按 eval_fraction 折算每個 epoch 的步數
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(1, args.batch_size)))
    eval_every_steps = max(1, math.ceil(steps_per_epoch * max(0.01, min(1.0, args.eval_fraction))))
    # 策略式（新/舊版本字段名兼容）
    key_eval = "evaluation_strategy" if "evaluation_strategy" in allowed else ("eval_strategy" if "eval_strategy" in allowed else None)
    if key_eval:
        args_dict[key_eval] = "steps"
    if "save_strategy" in allowed:
        args_dict["save_strategy"] = "steps"
    if "eval_steps" in allowed:
        args_dict["eval_steps"] = eval_every_steps
    if "save_steps" in allowed:
        args_dict["save_steps"] = eval_every_steps
    if "save_total_limit" in allowed:
        args_dict["save_total_limit"] = 5
    # 將日誌步長與評估/保存步長對齊，減少刷屏
    if "logging_steps" in allowed:
        args_dict["logging_steps"] = eval_every_steps
    # 最優模型回滾（僅當評估與保存策略一致時開啓）
    if "metric_for_best_model" in allowed:
        args_dict["metric_for_best_model"] = "f1"
    if "greater_is_better" in allowed:
        args_dict["greater_is_better"] = True
    if "load_best_model_at_end" in allowed:
        eval_strat = args_dict.get("evaluation_strategy", args_dict.get("eval_strategy"))
        save_strat = args_dict.get("save_strategy")
        if eval_strat == save_strat and eval_strat in ("steps", "epoch"):
            args_dict["load_best_model_at_end"] = True

    # 兼容無 warmup_ratio 的版本：若支持 warmup_steps 則忽略比例
    if "warmup_ratio" not in allowed and "warmup_steps" in allowed:
        # 不計算總步數，默認 0
        args_dict["warmup_steps"] = 0

    # 若不支持策略式參數：退化爲每 eval_every_steps 步保存/評估
    if "save_strategy" not in allowed and "save_steps" in allowed:
        args_dict["save_steps"] = eval_every_steps
    if ("evaluation_strategy" not in allowed and "eval_strategy" not in allowed) and "eval_steps" in allowed:
        args_dict["eval_steps"] = eval_every_steps

    # 如果支持 load_best_model_at_end，但無法同時設置評估/保存策略，則關閉它以避免報錯
    if "load_best_model_at_end" in allowed:
        want_load_best = args_dict.get("load_best_model_at_end", False)
        eval_set = args_dict.get("evaluation_strategy", None)
        save_set = args_dict.get("save_strategy", None)
        if want_load_best and (eval_set is None or save_set is None or eval_set != save_set):
            args_dict["load_best_model_at_end"] = False

    training_args = TrainingArguments(**args_dict)
    print("[Info] 訓練參數要點:")
    print(f"       epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}, weight_decay={args.weight_decay}")
    print(f"       max_length={args.max_length}, seed={args.seed}, fp16={args.fp16}")
    if "warmup_ratio" in allowed and "warmup_ratio" in args_dict:
        print(f"       warmup_ratio={args_dict['warmup_ratio']}")
    elif "warmup_steps" in allowed and "warmup_steps" in args_dict:
        print(f"       warmup_steps={args_dict['warmup_steps']}")
    print(f"       steps_per_epoch={steps_per_epoch}, eval_every_steps={eval_every_steps}")
    print(f"       eval_strategy={args_dict.get('evaluation_strategy', args_dict.get('eval_strategy'))}, save_strategy={args_dict.get('save_strategy')}, logging_steps={args_dict.get('logging_steps')}")
    print(f"       save_total_limit={args_dict.get('save_total_limit', 'n/a')}, load_best_model_at_end={args_dict.get('load_best_model_at_end', False)}")

    callbacks = []
    if EarlyStoppingCallback is not None and (args_dict.get("evaluation_strategy") in ("steps", "epoch") or "eval_steps" in allowed):
        try:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stop_patience,
                    early_stopping_threshold=args.early_stop_threshold,
                )
            )
        except Exception:
            pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    # 設備與 GPU 信息
    try:
        device_cnt = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0) if device_cnt > 0 else "cpu"
        print(f"[Info] CUDA 可見設備數: {device_cnt}, 當前設備: {dev_name}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    except Exception:
        pass

    print("[Info] 開始訓練 ...")

    trainer.train()

    # 保存
    tokenizer.save_pretrained(output_dir)
    trainer.model.config.id2label = {int(i): str(l) for i, l in id2label.items()}
    trainer.model.config.label2id = {str(l): int(i) for l, i in label2id.items()}
    trainer.save_model(output_dir)
    try:
        best_metric = getattr(trainer.state, "best_metric", None)
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_metric is not None and best_ckpt is not None:
            print(f"[Info] 最優模型: metric={best_metric:.6f} | checkpoint={best_ckpt}")
    except Exception:
        pass

    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": trainer.model.config.label2id, "id2label": trainer.model.config.id2label},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 訓練曲線：可選保存訓練與評估 loss
    try:
        import matplotlib.pyplot as plt  # type: ignore
        logs = trainer.state.log_history
        t_steps, t_losses, e_steps, e_losses = [], [], [], []
        step_counter = 0
        for rec in logs:
            if "loss" in rec and "epoch" in rec:
                step_counter += 1
                t_steps.append(step_counter)
                t_losses.append(rec["loss"])
            if "eval_loss" in rec:
                e_steps.append(step_counter)
                e_losses.append(rec["eval_loss"])
        if t_losses or e_losses:
            plt.figure(figsize=(8,4))
            if t_losses:
                plt.plot(t_steps, t_losses, label="train_loss")
            if e_losses:
                plt.plot(e_steps, e_losses, label="eval_loss")
            plt.xlabel("training step (logged)")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_curve.png"))
    except Exception:
        pass

    print(f"微調完成，模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()



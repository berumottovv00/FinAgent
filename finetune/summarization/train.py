"""
文本摘要 LoRA 微调训练脚本
底座：Qwen-7B-Instruct
数据：data_preprocessing/processed/summarization/train.jsonl  /  val.jsonl
输出：outputs/summarization/  (LoRA adapter + tokenizer)

用法：
    python train.py                          # 使用默认 lora_config.yaml
    python train.py --config lora_config.yaml
    torchrun --nproc_per_node=2 train.py    # 多卡 DDP
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── 路径锚点 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    """相对于脚本目录解析路径。"""
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ── 数据格式化 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "你是一名专业的金融研报分析师，擅长将研报提炼为简明摘要。"


def format_sample(record: dict, tokenizer) -> dict:
    """
    将 Alpaca 格式样本转换为 {"prompt": ..., "completion": ...} 格式。
    trl 1.0 的 SFTTrainer 检测到这两个字段后会自动只在 completion 部分计算 loss。
    """
    instruction = record["instruction"]
    input_text  = record["input"].strip()
    output_text = record["output"].strip()

    user_content = f"{instruction}\n\n【研报原文】\n{input_text}"

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # 追加 <|im_start|>assistant\n
    )
    # completion 只包含助手回复内容 + eos
    completion = output_text + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


def build_dataset(
    records: list[dict],
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    """格式化并过滤超长样本。"""
    formatted = []
    skipped = 0
    for rec in records:
        sample = format_sample(rec, tokenizer)
        total_len = len(tokenizer.encode(sample["prompt"] + sample["completion"]))
        if total_len > max_seq_length:
            skipped += 1
            continue
        formatted.append(sample)

    if skipped:
        print(f"[warn] 跳过 {skipped} 条超长样本（>{max_seq_length} tokens）")
    print(f"[data] 有效样本：{len(formatted)} 条")
    return Dataset.from_list(formatted)


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(SCRIPT_DIR / args.config)

    # ── 路径解析 ────────────────────────────────────────────────────────────
    model_path  = cfg["model_name_or_path"]          # 可以是 HF hub id 或本地路径
    train_file  = resolve_path(SCRIPT_DIR, cfg["train_file"])
    val_file    = resolve_path(SCRIPT_DIR, cfg["val_file"])
    output_dir  = resolve_path(SCRIPT_DIR, cfg["output_dir"])
    max_seq_len = cfg.get("max_seq_length", 2048)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print(f"[init] 加载 tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",   # SFT 训练时右填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 数据集 ──────────────────────────────────────────────────────────────
    print("[data] 加载训练集 ...")
    train_records = load_jsonl(train_file)
    val_records   = load_jsonl(val_file)
    train_dataset = build_dataset(train_records, tokenizer, max_seq_len)
    val_dataset   = build_dataset(val_records,   tokenizer, max_seq_len)

    # ── 模型 ────────────────────────────────────────────────────────────────
    use_bf16 = cfg.get("bf16", True) and torch.cuda.is_bf16_supported()
    use_fp16 = cfg.get("fp16", False) and not use_bf16

    # 可选：4-bit 量化加载（节省显存，适合单卡 24G）
    # bnb_cfg = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    print(f"[init] 加载模型：{model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32),
        device_map="auto",
        # quantization_config=bnb_cfg,   # 取消注释以启用 4-bit
    )
    model.config.use_cache = False     # 训练时关闭 KV-cache

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── SFTConfig（含 TrainingArguments 所有参数）───────────────────────────
    # trl 1.0 用 SFTConfig 替代 TrainingArguments + DataCollatorForCompletionOnlyLM
    # completion_only_loss=True：自动检测 prompt/completion 字段，只在 completion 计算 loss
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 10),
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        report_to=cfg.get("report_to", "tensorboard"),
        max_length=max_seq_len,
        completion_only_loss=True,
        ddp_find_unused_parameters=False,
    )

    # ── SFTTrainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # ── 训练 ────────────────────────────────────────────────────────────────
    print("\n[train] 开始训练 ...")
    trainer.train()

    # ── 保存 LoRA adapter + tokenizer ───────────────────────────────────────
    best_dir = output_dir / "best_adapter"
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\n[done] LoRA adapter 已保存至：{best_dir}")


if __name__ == "__main__":
    main()
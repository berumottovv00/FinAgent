"""
微调前 vs 微调后 情感分类指标对比评估脚本
底座：Qwen2-1.5B-Instruct
数据：data_preprocessing/processed/classification/val.jsonl
适配器：outputs/classification/best_adapter

指标：
  - 标签准确率（Accuracy）
  - 各类别精确率 / 召回率 / F1（正面 / 中性 / 负面）

用法：
    python eval_classification.py
    python eval_classification.py --max_samples 20
    python eval_classification.py --no_base
    python eval_classification.py --output_file outputs/classification/eval_result.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent

SYSTEM_PROMPT = (
    "你是一名专业的金融研报分析师，擅长判断研报的情感倾向。"
    "只输出情感标签（正面/中性/负面）和理由，格式为：\n正面\n理由：..."
)

LABELS = ["正面", "中性", "负面"]


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_prompt(record: dict, tokenizer) -> str:
    instruction = record["instruction"]
    input_text  = record["input"].strip()
    user_content = f"{instruction}\n\n【研报原文】\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def generate_output(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def extract_label(text: str) -> str | None:
    """从模型输出中提取标签。
    优先匹配首行；若首行不是合法标签，则逐行扫描全文（兼容模型输出格式不规范的情况）。
    """
    if not text.strip():
        return None
    lines = [l.strip() for l in text.strip().splitlines()]
    # 优先：第一行直接是标签
    if lines[0] in LABELS:
        return lines[0]
    # 兜底：逐行查找（含"理由：正面"等格式）
    for line in lines:
        for label in LABELS:
            if label in line:
                return label
    return None


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """计算准确率和各类别 Precision / Recall / F1。"""
    correct = 0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct += 1
            tp[ref] += 1
        else:
            if pred is not None:
                fp[pred] += 1
            fn[ref] += 1

    n = len(references)
    accuracy = correct / n

    per_class = {}
    for label in LABELS:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {"precision": p, "recall": r, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)

    return {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "per_class": per_class,
    }


def evaluate_model(model, tokenizer, records: list[dict], label: str, max_new_tokens: int):
    raw_outputs  = []
    pred_labels  = []
    ref_labels   = [r["output"].strip().splitlines()[0].strip() for r in records]

    print(f"\n[eval] 正在评估：{label}")
    for rec in tqdm(records, desc=label, unit="样本"):
        prompt = build_prompt(rec, tokenizer)
        output = generate_output(model, tokenizer, prompt, max_new_tokens)
        raw_outputs.append(output)
        pred_labels.append(extract_label(output))

    invalid = sum(1 for p in pred_labels if p is None)
    if invalid:
        print(f"[warn] {invalid} 条输出未能识别标签，计为错误")
    # None 视为错误标签
    safe_preds = [p if p is not None else "__invalid__" for p in pred_labels]
    metrics = compute_metrics(safe_preds, ref_labels)
    return raw_outputs, pred_labels, metrics


def print_comparison(base_metrics: dict | None, ft_metrics: dict) -> None:
    print("\n" + "=" * 55)
    print("情感分类指标对比")
    print("=" * 55)

    for key, name in [("accuracy", "准确率"), ("macro_f1", "宏平均F1")]:
        fv = ft_metrics[key]
        if base_metrics:
            bv = base_metrics[key]
            d  = fv - bv
            sign = "+" if d >= 0 else ""
            print(f"{name:<10} 微调前: {bv:.4f}  微调后: {fv:.4f}  ({sign}{d:.4f})")
        else:
            print(f"{name:<10} {fv:.4f}")

    print(f"\n── 各类别 F1 ──")
    if base_metrics:
        print(f"  {'类别':<6} {'微调前':>8} {'微调后':>8} {'提升':>8}")
        print(f"  {'-'*36}")
        for label in LABELS:
            bv = base_metrics["per_class"][label]["f1"]
            fv = ft_metrics["per_class"][label]["f1"]
            d  = fv - bv
            sign = "+" if d >= 0 else ""
            print(f"  {label:<6} {bv:>8.4f} {fv:>8.4f} {sign}{d:>7.4f}")
    else:
        for label in LABELS:
            fv = ft_metrics["per_class"][label]["f1"]
            print(f"  {label}: F1={fv:.4f}")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="情感分类微调前后指标对比")
    parser.add_argument("--config",         default="lora_config.yaml")
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--no_base",        action="store_true")
    parser.add_argument("--output_file",    type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(SCRIPT_DIR / args.config)
    model_path  = cfg["model_name_or_path"]
    val_file    = resolve_path(SCRIPT_DIR, cfg["val_file"])
    adapter_dir = resolve_path(SCRIPT_DIR, cfg["output_dir"]) / "best_adapter"

    records = load_jsonl(val_file)
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"[data] 共 {len(records)} 条样本")

    print(f"\n[init] 加载 tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ── 评估 Base Model ──────────────────────────────────────────────────────
    base_metrics     = None
    base_pred_labels = None

    if not args.no_base:
        print(f"\n[init] 加载 base model：{model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
        )
        base_model.eval()
        _, base_pred_labels, base_metrics = evaluate_model(
            base_model, tokenizer, records, "Base Model", args.max_new_tokens,
        )
        del base_model
        torch.cuda.empty_cache()

    # ── 评估微调后模型 ────────────────────────────────────────────────────────
    if not adapter_dir.exists():
        raise FileNotFoundError(f"未找到 LoRA adapter：{adapter_dir}\n请先运行 train.py 完成微调。")

    print(f"\n[init] 加载微调模型：{model_path} + {adapter_dir}")
    ft_base = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_dir))
    ft_model.eval()

    ft_raw, ft_pred_labels, ft_metrics = evaluate_model(
        ft_model, tokenizer, records, "Fine-tuned Model", args.max_new_tokens,
    )

    # ── 打印对比 ──────────────────────────────────────────────────────────────
    print_comparison(base_metrics, ft_metrics)

    # ── 打印样例 ──────────────────────────────────────────────────────────────
    n_show = min(3, len(records))
    print(f"\n[样例] 展示前 {n_show} 条：")
    ref_labels = [r["output"].strip().splitlines()[0].strip() for r in records]
    for i in range(n_show):
        print(f"\n--- 样例 {i + 1} ---")
        print(f"参考标签：{ref_labels[i]}")
        if base_pred_labels:
            print(f"Base 预测：{base_pred_labels[i]}")
        print(f"微调预测：{ft_pred_labels[i]}  | 完整输出：{ft_raw[i][:80]}")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    if args.output_file:
        result = {"num_samples": len(records), "fine_tuned": ft_metrics}
        if base_metrics:
            result["base"] = base_metrics
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[done] 结果已保存至：{out_path}")


if __name__ == "__main__":
    main()

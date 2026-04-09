"""
微调前 vs 微调后 关键信息提取指标对比评估脚本
底座：Qwen2-1.5B-Instruct
数据：data_preprocessing/processed/extraction/val.jsonl
适配器：outputs/extraction/best_adapter

指标：
  - JSON 解析成功率（输出是否为合法 JSON）
  - 字段命中率（各字段非 null 且与参考一致的比例）
  - 字段覆盖率（各字段非 null 的比例，衡量提取完整性）

用法：
    python eval_extraction.py
    python eval_extraction.py --max_samples 10
    python eval_extraction.py --no_base
    python eval_extraction.py --output_file outputs/extraction/eval_result.json
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent

SYSTEM_PROMPT = "你是一名专业的金融研报信息提取助手，擅长从研报中准确抽取结构化数据，只输出 JSON，不添加任何额外说明。"

TOP_FIELDS = ["公司名称", "股票代码", "评级", "目标价", "核心投资逻辑", "风险提示"]


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


def generate_output(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
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


def parse_json(text: str) -> dict | None:
    """尝试解析 JSON，去掉可能的代码块包裹。"""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def compute_metrics(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """
    计算三类指标：
    - json_parse_rate: JSON 解析成功率
    - field_coverage: 各字段非 null 覆盖率（模型是否提取到该字段）
    - field_match: 各字段与参考值匹配率（字符串包含匹配）
    """
    n = len(predictions)
    parse_ok = 0
    coverage  = {f: 0 for f in TOP_FIELDS}
    match     = {f: 0 for f in TOP_FIELDS}

    for pred_text, ref in zip(predictions, references):
        ref_obj = json.loads(ref["output"]) if isinstance(ref["output"], str) else ref["output"]
        pred_obj = parse_json(pred_text)

        if pred_obj is None:
            continue
        parse_ok += 1

        for field in TOP_FIELDS:
            pred_val = pred_obj.get(field)
            ref_val  = ref_obj.get(field)

            # 覆盖率：预测值非 null
            if pred_val is not None:
                coverage[field] += 1

            # 匹配率：参考值非 null 且预测值包含参考值关键字
            if ref_val is not None and pred_val is not None:
                ref_str  = str(ref_val) if not isinstance(ref_val, list) else "".join(str(v) for v in ref_val)
                pred_str = str(pred_val) if not isinstance(pred_val, list) else "".join(str(v) for v in pred_val)
                # 取参考值前10字做模糊匹配
                key = ref_str[:10].strip()
                if key and key in pred_str:
                    match[field] += 1

    return {
        "json_parse_rate": parse_ok / n,
        "field_coverage":  {f: coverage[f] / n for f in TOP_FIELDS},
        "field_match":     {f: match[f] / max(1, coverage[f]) for f in TOP_FIELDS},
    }


def evaluate_model(model, tokenizer, records: list[dict], label: str, max_new_tokens: int) -> tuple:
    predictions = []
    print(f"\n[eval] 正在评估：{label}")
    for rec in tqdm(records, desc=label, unit="样本"):
        prompt = build_prompt(rec, tokenizer)
        pred = generate_output(model, tokenizer, prompt, max_new_tokens)
        predictions.append(pred)
    metrics = compute_metrics(predictions, records)
    return predictions, metrics


def print_comparison(base_metrics: dict | None, ft_metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("关键信息提取指标对比")
    print("=" * 60)

    # JSON 解析率
    ft_rate = ft_metrics["json_parse_rate"]
    if base_metrics:
        base_rate = base_metrics["json_parse_rate"]
        delta = ft_rate - base_rate
        sign = "+" if delta >= 0 else ""
        print(f"\n{'JSON解析成功率':<16} 微调前: {base_rate:.2%}  微调后: {ft_rate:.2%}  ({sign}{delta:.2%})")
    else:
        print(f"\nJSON解析成功率: {ft_rate:.2%}")

    # 字段覆盖率 & 匹配率
    for metric_key, label_cn in [("field_coverage", "字段覆盖率"), ("field_match", "字段匹配率")]:
        print(f"\n── {label_cn} ──")
        if base_metrics:
            print(f"  {'字段':<12} {'微调前':>8} {'微调后':>8} {'提升':>8}")
            print(f"  {'-'*40}")
            for f in TOP_FIELDS:
                bv = base_metrics[metric_key][f]
                fv = ft_metrics[metric_key][f]
                d  = fv - bv
                sign = "+" if d >= 0 else ""
                print(f"  {f:<12} {bv:>8.2%} {fv:>8.2%} {sign}{d:>7.2%}")
        else:
            for f in TOP_FIELDS:
                print(f"  {f:<12} {ft_metrics[metric_key][f]:.2%}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="提取任务微调前后指标对比")
    parser.add_argument("--config",        default="lora_config.yaml")
    parser.add_argument("--max_samples",   type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_base",       action="store_true")
    parser.add_argument("--output_file",   type=str, default=None)
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
    base_metrics = None
    base_predictions = None

    if not args.no_base:
        print(f"\n[init] 加载 base model：{model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
        )
        base_model.eval()
        base_predictions, base_metrics = evaluate_model(
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

    ft_predictions, ft_metrics = evaluate_model(
        ft_model, tokenizer, records, "Fine-tuned Model", args.max_new_tokens,
    )

    # ── 打印对比 ──────────────────────────────────────────────────────────────
    print_comparison(base_metrics, ft_metrics)

    # ── 打印样例 ──────────────────────────────────────────────────────────────
    n_show = min(2, len(records))
    print(f"\n[样例] 展示前 {n_show} 条生成结果：")
    for i in range(n_show):
        ref_obj = json.loads(records[i]["output"])
        print(f"\n--- 样例 {i + 1} ---")
        print(f"参考：{json.dumps(ref_obj, ensure_ascii=False)[:200]}")
        if base_predictions:
            print(f"Base：{base_predictions[i][:200]}")
        print(f"微调：{ft_predictions[i][:200]}")

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

"""
微调前 vs 微调后 ROUGE 指标对比评估脚本
底座：Qwen/Qwen-7B-Instruct
数据：data_preprocessing/processed/summarization/val.jsonl
适配器：outputs/summarization/best_adapter

用法：
    python eval_rouge.py                          # 使用默认路径
    python eval_rouge.py --config lora_config.yaml
    python eval_rouge.py --max_samples 50         # 只评估前 50 条（节省时间）
    python eval_rouge.py --no_base               # 跳过 base model，只评估微调后
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 路径锚点 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

SYSTEM_PROMPT = "你是一名专业的金融研报分析师，擅长将研报提炼为简明摘要。"


# ── 工具函数 ─────────────────────────────────────────────────────────────────

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


def build_prompt(record: dict, tokenizer: AutoTokenizer) -> str:
    """构造推理时的输入（不含 assistant 回复）。"""
    instruction = record["instruction"]
    input_text = record["input"].strip()
    user_content = f"{instruction}\n\n【研报原文】\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 追加 <|im_start|>assistant\n
    )


# ── 推理 ─────────────────────────────────────────────────────────────────────

def generate_summary(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 贪心解码，保证结果可复现
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只解码新生成的 token
    new_ids = output_ids[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip()


# ── ROUGE 计算 ────────────────────────────────────────────────────────────────

def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """计算 ROUGE-1 / ROUGE-2 / ROUGE-L，返回 F1 均值。"""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,          # 中文不需要词干提取
        tokenizer=_CharTokenizer(), # 用字符级分词适配中文
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    return {k: v / n for k, v in totals.items()}


class _CharTokenizer:
    """将字符串拆分为单字符列表，适合中文 ROUGE 计算。"""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


# ── 主流程 ───────────────────────────────────────────────────────────────────

def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict],
    label: str,
    max_new_tokens: int = 256,
) -> tuple[list[str], dict[str, float]]:
    """对 records 逐条生成摘要，返回 (predictions, rouge_scores)。"""
    predictions = []
    references = [r["output"].strip() for r in records]

    print(f"\n[eval] 正在评估：{label}")
    for rec in tqdm(records, desc=label, unit="样本"):
        prompt = build_prompt(rec, tokenizer)
        pred = generate_summary(model, tokenizer, prompt, max_new_tokens)
        predictions.append(pred)

    rouge = compute_rouge(predictions, references)
    return predictions, rouge


def print_comparison(
    base_rouge: dict[str, float] | None,
    ft_rouge: dict[str, float],
) -> None:
    metrics = ["rouge1", "rouge2", "rougeL"]
    col_w = 12

    header = f"{'指标':<10}" + f"{'微调后':>{col_w}}"
    if base_rouge is not None:
        header = f"{'指标':<10}" + f"{'微调前':>{col_w}}" + f"{'微调后':>{col_w}}" + f"{'提升':>{col_w}}"

    print("\n" + "=" * len(header))
    print("ROUGE 指标对比（F1 均值）")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for m in metrics:
        ft_val = ft_rouge[m]
        if base_rouge is not None:
            base_val = base_rouge[m]
            delta = ft_val - base_val
            sign = "+" if delta >= 0 else ""
            print(
                f"{m:<10}"
                f"{base_val:>{col_w}.4f}"
                f"{ft_val:>{col_w}.4f}"
                f"{sign}{delta:>{col_w - 1}.4f}"
            )
        else:
            print(f"{m:<10}{ft_val:>{col_w}.4f}")

    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="微调前后 ROUGE 对比评估")
    parser.add_argument("--config", default="lora_config.yaml", help="LoRA 配置文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估多少条样本（默认全部）")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成摘要的最大 token 数")
    parser.add_argument("--no_base", action="store_true", help="跳过 base model 评估，只评估微调后模型")
    parser.add_argument("--output_file", type=str, default=None, help="将结果 JSON 保存到指定路径")
    args = parser.parse_args()

    cfg = load_config(SCRIPT_DIR / args.config)

    model_path = cfg["model_name_or_path"]
    val_file = resolve_path(SCRIPT_DIR, cfg["val_file"])
    output_dir = resolve_path(SCRIPT_DIR, cfg["output_dir"])
    adapter_dir = output_dir / "best_adapter"

    # ── 加载验证集 ──────────────────────────────────────────────────────────
    print(f"[data] 加载验证集：{val_file}")
    records = load_jsonl(val_file)
    if args.max_samples:
        records = records[: args.max_samples]
        print(f"[data] 截取前 {args.max_samples} 条样本")
    print(f"[data] 共 {len(records)} 条样本")

    # ── 加载 Tokenizer ──────────────────────────────────────────────────────
    print(f"\n[init] 加载 tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",    # 推理时左填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ── 评估 Base Model ──────────────────────────────────────────────────────
    base_rouge = None
    base_predictions = None

    if not args.no_base:
        print(f"\n[init] 加载 base model：{model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        base_model.config.use_cache = True
        base_model.eval()

        base_predictions, base_rouge = evaluate_model(
            base_model, tokenizer, records,
            label="Base Model",
            max_new_tokens=args.max_new_tokens,
        )

        # 释放显存
        del base_model
        torch.cuda.empty_cache()

    # ── 评估微调后模型 ────────────────────────────────────────────────────────
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"未找到 LoRA adapter：{adapter_dir}\n"
            "请先运行 train.py 完成微调，或检查 output_dir 配置。"
        )

    print(f"\n[init] 加载 base model（用于挂载 LoRA）：{model_path}")
    ft_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    print(f"[init] 加载 LoRA adapter：{adapter_dir}")
    ft_model = PeftModel.from_pretrained(ft_base, str(adapter_dir))
    ft_model.eval()

    ft_predictions, ft_rouge = evaluate_model(
        ft_model, tokenizer, records,
        label="Fine-tuned Model",
        max_new_tokens=args.max_new_tokens,
    )

    # ── 打印对比结果 ──────────────────────────────────────────────────────────
    print_comparison(base_rouge, ft_rouge)

    # ── 可选：打印部分生成样例 ────────────────────────────────────────────────
    n_show = min(3, len(records))
    print(f"\n[样例] 展示前 {n_show} 条生成结果：")
    for i in range(n_show):
        print(f"\n--- 样例 {i + 1} ---")
        print(f"参考摘要：{records[i]['output'].strip()[:150]}")
        if base_predictions is not None:
            print(f"Base 摘要：{base_predictions[i][:150]}")
        print(f"微调摘要：{ft_predictions[i][:150]}")

    # ── 可选：保存结果 JSON ───────────────────────────────────────────────────
    if args.output_file:
        result = {
            "num_samples": len(records),
            "fine_tuned": ft_rouge,
        }
        if base_rouge is not None:
            result["base"] = base_rouge
            result["delta"] = {k: ft_rouge[k] - base_rouge[k] for k in ft_rouge}

        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[done] 结果已保存至：{out_path}")


if __name__ == "__main__":
    main()
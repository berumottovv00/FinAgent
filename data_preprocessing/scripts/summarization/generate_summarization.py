"""
生成研报摘要任务的训练样本。
输入：data_preprocessing/raw/*.txt
输出：data_preprocessing/processed/summarization/train.jsonl
                                                   val.jsonl
每条样本格式（Alpaca）：
{
    "instruction": "...",
    "input": "...",
    "output": "...",
    "source": "filename"
}
"""

import json
import os
import random
import time
from pathlib import Path

import requests

# ── 路径 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "processed" / "summarization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELED_FILE = OUT_DIR / "labeled.jsonl"   # 中间文件，逐条写入，支持断点续跑
TRAIN_FILE   = OUT_DIR / "train.jsonl"
VAL_FILE     = OUT_DIR / "val.jsonl"

# ── 豆包 API 配置（从环境变量读取）─────────────────────────────────────────
DOUBAO_API_KEY = os.environ["DOUBAO_API_KEY"]
DOUBAO_URL     = os.environ["DOUBAO_URL"]
DOUBAO_MODEL   = os.environ.get("DOUBAO_MODEL", "deepseek-v3-250324")

# ── 参数 ────────────────────────────────────────────────────────────────────
VAL_RATIO   = 0.1    # 验证集比例
RANDOM_SEED = 42
RETRY_LIMIT = 3
RETRY_DELAY = 10     # 秒

INSTRUCTION = "请对以下金融研究报告进行摘要，提炼核心结论、关键数据和投资建议，100～150字以内，语言简洁专业。"

SYSTEM_PROMPT = "你是一名专业的金融研报分析师，擅长将研报提炼为简明摘要。"

USER_TEMPLATE = """{instruction}

【研报原文】
{text}"""

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_done(path: Path) -> set:
    """读取已完成的 source 列表，用于断点续跑。"""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["source"])
    return done


def call_api(text: str) -> str:
    """调用豆包 API，带重试。"""
    headers = {
        "Authorization": DOUBAO_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "model": DOUBAO_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                instruction=INSTRUCTION,
                text=text
            )}
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.post(
                DOUBAO_URL,
                headers=headers,
                data=json.dumps(data),
                timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{RETRY_LIMIT}] API error: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    raise RuntimeError("API 调用失败，已达重试上限")


def split_and_save(labeled_file: Path):
    """将 labeled.jsonl 划分为 train / val 并保存。"""
    data = [
        json.loads(line)
        for line in labeled_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.seed(RANDOM_SEED)
    random.shuffle(data)

    n_val = max(1, int(len(data) * VAL_RATIO))
    val_data   = data[:n_val]
    train_data = data[n_val:]

    for path, records in [(TRAIN_FILE, train_data), (VAL_FILE, val_data)]:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n数据集划分完成：train={len(train_data)}，val={len(val_data)}")
    print(f"  → {TRAIN_FILE}")
    print(f"  → {VAL_FILE}")


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"未找到任何 .txt 文件：{RAW_DIR}")

    done = load_done(LABELED_FILE)
    print(f"共 {len(txt_files)} 篇研报，已完成 {len(done)} 篇，剩余 {len(txt_files) - len(done)} 篇\n")

    with LABELED_FILE.open("a", encoding="utf-8") as f:
        for txt_path in txt_files:
            stem = txt_path.stem
            if stem in done:
                print(f"[跳过] {stem}")
                continue

            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                print(f"[空文件，跳过] {stem}")
                continue

            print(f"[生成] {stem} ... ", end="", flush=True)
            summary = call_api(text)

            record = {
                "instruction": INSTRUCTION,
                "input": text,
                "output": summary,
                "source": stem,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("OK")

    print("\n所有研报处理完成，开始划分数据集...")
    split_and_save(LABELED_FILE)


if __name__ == "__main__":
    main()
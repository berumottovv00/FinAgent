"""
生成研报关键信息提取任务的训练样本。
输入：data_preprocessing/raw/*.txt
输出：data_preprocessing/processed/extraction/train.jsonl
                                               val.jsonl

提取目标：投资评级、目标价、核心财务指标、风险提示
每条样本格式（Alpaca）：
{
    "instruction": "...",
    "input": "...",
    "output": "...",   # 结构化 JSON 字符串
    "source": "filename"
}
"""

import json
import random
import time
from pathlib import Path

import anthropic

# ── 路径 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "raw"
OUT_DIR = ROOT / "processed" / "extraction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELED_FILE = OUT_DIR / "labeled.jsonl"
TRAIN_FILE   = OUT_DIR / "train.jsonl"
VAL_FILE     = OUT_DIR / "val.jsonl"

# ── 参数 ────────────────────────────────────────────────────────────────────
VAL_RATIO   = 0.1
RANDOM_SEED = 42
RETRY_LIMIT = 3
RETRY_DELAY = 5

INSTRUCTION = (
    "请从以下金融研究报告中提取关键信息，以 JSON 格式输出，"
    "字段包括：公司名称、评级、目标价、核心财务指标（收入/利润/增速等）、主要风险。"
    "若原文未提及某字段则填 null。"
)

SYSTEM_PROMPT = "你是一名专业的金融研报信息提取助手，擅长从研报中准确抽取结构化数据，只输出 JSON，不添加任何额外说明。"

USER_TEMPLATE = """{instruction}

【研报原文】
{text}"""

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["source"])
    return done


def call_api(client: anthropic.Anthropic, text: str) -> str:
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        instruction=INSTRUCTION,
                        text=text
                    )
                }]
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{RETRY_LIMIT}] API error: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    raise RuntimeError("API 调用失败，已达重试上限")


def split_and_save(labeled_file: Path):
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
    client = anthropic.Anthropic()

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
            result = call_api(client, text)

            record = {
                "instruction": INSTRUCTION,
                "input": text,
                "output": result,
                "source": stem,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("OK")

    print("\n所有研报处理完成，开始划分数据集...")
    split_and_save(LABELED_FILE)


if __name__ == "__main__":
    main()

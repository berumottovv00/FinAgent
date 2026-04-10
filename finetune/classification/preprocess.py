import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "", "data")

CLASS_NAMES = {0: "neg", 1: "pos"}

# 1. 加载原始金融数据集
dataset = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
df = pd.DataFrame(dataset["train"])

# 2. 过滤中性样本，映射标签（原0=正面保留，原2=负面→1）
df = df[df["label"] != 1].copy()
df["label"] = df["label"].replace({2: 1})

# 3. 划分训练(70%) / 验证(15%) / 测试(15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


def save_txt(dataframe, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            text = row["sentence"].replace("\n", " ").replace("\t", " ")
            f.write(f"{text}\t{row['label']}\n")


def save_class(path):
    with open(path, "w", encoding="utf-8") as f:
        for label_id, name in sorted(CLASS_NAMES.items()):
            f.write(f"{label_id + 1}\t{label_id}:{name}\n")


os.makedirs(OUTPUT_DIR, exist_ok=True)
save_txt(train_df, os.path.join(OUTPUT_DIR, "train.txt"))
save_txt(dev_df,   os.path.join(OUTPUT_DIR, "dev.txt"))
save_txt(test_df,  os.path.join(OUTPUT_DIR, "test.txt"))
save_class(os.path.join(OUTPUT_DIR, "class.txt"))

print(f"训练集：{len(train_df)} 条")
print(f"验证集：{len(dev_df)} 条")
print(f"测试集：{len(test_df)} 条")
print(f"文件已保存至：{os.path.abspath(OUTPUT_DIR)}")
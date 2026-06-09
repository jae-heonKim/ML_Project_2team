"""
이메일 긴급도 분류 - 모델 비교 학습/평가
- 4개 모델 x 2개 데이터(균형/불균형) = 8회 학습
- 설정: Epoch=3, Batch=8, LR=2e-5, MaxLen=256 (기존 발표와 동일)
- 평가: Accuracy, F1, Precision, Recall
- 결과를 balanced&imbalanced_results.csv에 누적 저장 (중간에 끊겨도 이어서 가능)

필요 파일 (같은 폴더에):
  balanced_train.csv, balanced_test.csv
  imbalanced_train.csv, imbalanced_test.csv
"""
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# 설정
# ==========================================
MODELS = {
    "koELECTRA":     "monologg/koelectra-base-v3-discriminator",
    "kcELECTRA":     "beomi/KcELECTRA-base-v2022",
    "klue-roberta":  "klue/roberta-base",
    "klue-bert":     "klue/bert-base",
}

DATASETS = {
    "balanced":   ("balanced_train.csv",   "balanced_test.csv"),
    "imbalanced": ("imbalanced_train.csv", "imbalanced_test.csv"),
}

# 학습 하이퍼파라미터 (기존 발표와 동일)
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01

RESULTS_FILE = "balanced&imbalanced_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }


def already_done(model_name, data_name):
    """이미 완료된 조합인지 확인 (중단 후 재개용)"""
    if not os.path.exists(RESULTS_FILE):
        return False
    done = pd.read_csv(RESULTS_FILE)
    return ((done["model"] == model_name) & (done["dataset"] == data_name)).any()


def save_result(row):
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")


def train_one(model_name, model_path, data_name, train_file, test_file):
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} × {data_name} 학습 시작")
    print(f"{'='*60}")

    # 데이터 로드
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    # 모델
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # 학습 설정
    args = TrainingArguments(
        output_dir=f"./ckpt_{model_name}_{data_name}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="no",          # 체크포인트 저장 안 함 (디스크 절약)
        logging_steps=200,
        report_to="none",
        fp16=torch.cuda.is_available(),  # GPU면 자동 혼합정밀 (속도↑, 메모리↓)
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()

    print(f"\n📊 {model_name} × {data_name} 결과:")
    print(f"   Accuracy:  {metrics['eval_accuracy']:.4f}")
    print(f"   F1:        {metrics['eval_f1']:.4f}")
    print(f"   Precision: {metrics['eval_precision']:.4f}")
    print(f"   Recall:    {metrics['eval_recall']:.4f}")

    save_result({
        "model": model_name,
        "dataset": data_name,
        "accuracy": round(metrics["eval_accuracy"], 4),
        "f1": round(metrics["eval_f1"], 4),
        "precision": round(metrics["eval_precision"], 4),
        "recall": round(metrics["eval_recall"], 4),
    })

    # 메모리 정리
    del model, trainer
    torch.cuda.empty_cache()


def main():
    print(f"💻 Device: {DEVICE}")
    if DEVICE == "cpu":
        print("⚠️  GPU가 감지되지 않았습니다. 학습이 매우 느릴 수 있습니다.")

    total = len(MODELS) * len(DATASETS)
    count = 0

    for data_name, (train_file, test_file) in DATASETS.items():
        for model_name, model_path in MODELS.items():
            count += 1
            print(f"\n\n[{count}/{total}] {model_name} × {data_name}")

            if already_done(model_name, data_name):
                print(f"   ⏭️  이미 완료됨, 건너뜀")
                continue

            try:
                train_one(model_name, model_path, data_name, train_file, test_file)
            except torch.cuda.OutOfMemoryError:
                print(f"   ❌ VRAM 부족! batch_size를 줄이거나 이 모델은 건너뜁니다.")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   ❌ 오류: {e}")
                torch.cuda.empty_cache()

    # 최종 결과 출력
    print(f"\n\n{'='*60}")
    print("🏁 전체 학습 완료! 최종 결과:")
    print(f"{'='*60}")
    if os.path.exists(RESULTS_FILE):
        results = pd.read_csv(RESULTS_FILE)
        print(results.to_string(index=False))
        print(f"\n💾 결과 저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

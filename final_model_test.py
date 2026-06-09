"""
============================================================
최종 모델(klue-roberta) 학습 + 저장
============================================================
[목적] 비교 실험에서 1등한 klue-roberta를 최종 모델로 확정하고:
  1. 균형 데이터(중복 제거)로 학습
  2. ./final_model 폴더에 저장  -> 이후 API 서버가 이 모델을 사용

[설정] 비교 실험과 동일 (Epoch=3, Batch=8, LR=2e-5, MaxLen=256)
============================================================
"""
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_PATH = "klue/roberta-base"            # 최종 선정 모델
TRAIN_FILE = "dedup_balanced_train.csv"
TEST_FILE = "dedup_balanced_test.csv"
SAVE_DIR = "./final_model"                  # 학습된 모델 저장 위치
MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(eval_pred):
    """4개 지표 계산 (학습 후 성능 확인용)"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def main():
    print(f"Device: {DEVICE}\n")

    # ===== 데이터 로드 + 토큰화 =====
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tok, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tok, batched=True)

    # ===== 모델 로드 =====
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

    args = TrainingArguments(
        output_dir="./ckpt_final",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        learning_rate=LR, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=200, report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # ===== 학습 =====
    print("학습 시작...")
    trainer.train()

    # ===== 최종 성능 출력 =====
    metrics = trainer.evaluate()
    print(f"\n{'='*55}\n최종 모델 성능\n{'='*55}")
    print(f"  Acc {metrics['eval_accuracy']:.4f} | F1 {metrics['eval_f1']:.4f} "
          f"| P {metrics['eval_precision']:.4f} | R {metrics['eval_recall']:.4f}")

    # ===== 모델 저장 (API 서버가 이 폴더를 불러씀) =====
    print(f"\n모델 저장: {SAVE_DIR}")
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"완료! 모델은 {SAVE_DIR} 에 저장됨")


if __name__ == "__main__":
    main()
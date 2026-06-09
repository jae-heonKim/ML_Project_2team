"""
============================================================
이메일 긴급도 분류 - 4개 모델 비교 학습/평가 (중복 제거 데이터)
============================================================
[목적]
4개의 한국어 BERT 계열 모델을, 균형/불균형 두 데이터로 각각 학습해서
어떤 모델 + 어떤 데이터 조합이 가장 좋은지 비교한다. (4 x 2 = 총 8회 학습)

[공정한 비교를 위해]
모든 학습 설정(Epoch=3, Batch=8, LR=2e-5, MaxLen=256)을 기존 발표와 동일하게 고정.
-> 달라진 건 '데이터 품질'뿐이므로, 성능 차이는 데이터 덕분임을 보일 수 있다.

[평가 지표] Accuracy, F1, Precision, Recall
[출력] results_dedup.csv 에 8개 결과를 누적 저장 (중간에 끊겨도 이어서 가능)
============================================================
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
# 비교할 4개 모델 (이름: HuggingFace 모델 경로)
MODELS = {
    "koELECTRA":     "monologg/koelectra-base-v3-discriminator",
    "kcELECTRA":     "beomi/KcELECTRA-base-v2022",
    "klue-roberta":  "klue/roberta-base",
    "klue-bert":     "klue/bert-base",
}

# 2개 데이터셋 (균형 50:50 / 불균형 = 실제 비율)
DATASETS = {
    "balanced":   ("dedup_balanced_train.csv",   "dedup_balanced_test.csv"),
    "imbalanced": ("dedup_imbalanced_train.csv", "dedup_imbalanced_test.csv"),
}

# 학습 하이퍼파라미터 (기존 발표와 동일하게 고정)
EPOCHS = 3          # 전체 데이터를 3번 반복 학습
BATCH_SIZE = 8      # 한 번에 8개씩 묶어서 학습
LR = 2e-5           # 학습률 (가중치를 얼마나 크게 업데이트할지)
MAX_LENGTH = 256    # 입력 문장 최대 토큰 길이 (넘으면 잘림)
WEIGHT_DECAY = 0.01 # 과적합 방지용 정규화

RESULTS_FILE = "results_dedup.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 있으면 GPU 사용


def compute_metrics(eval_pred):
    """모델 예측 결과로 4개 지표를 계산하는 함수 (Trainer가 자동 호출)"""
    logits, labels = eval_pred              # logits=모델 출력 점수, labels=정답
    preds = np.argmax(logits, axis=-1)      # 점수가 높은 쪽을 예측값으로
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }


def already_done(model_name, data_name):
    """이미 끝낸 조합인지 확인 (중단 후 재개 시 건너뛰기 위해)"""
    if not os.path.exists(RESULTS_FILE):
        return False
    done = pd.read_csv(RESULTS_FILE)
    return ((done["model"] == model_name) & (done["dataset"] == data_name)).any()


def save_result(row):
    """결과 한 줄을 results_dedup.csv에 이어붙이기(append)"""
    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")


def train_one(model_name, model_path, data_name, train_file, test_file):
    """모델 1개 x 데이터 1개를 학습하고 평가해서 결과를 저장하는 함수"""
    print(f"\n{'='*60}\n{model_name} x {data_name} 학습 시작\n{'='*60}")

    # -- 1. 데이터 로드 --
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    # -- 2. 토크나이저 (문장 -> 숫자 토큰 변환기) --
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(batch):
        # 문장을 토큰으로 변환, MAX_LENGTH 넘으면 자름(truncation)
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    # pandas -> HuggingFace Dataset 으로 변환 후 토큰화
    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    # -- 3. 모델 로드 (num_labels=2 -> 긴급/비긴급 2개 분류) --
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # -- 4. 학습 설정 --
    args = TrainingArguments(
        output_dir=f"./ckpt_dedup_{model_name}_{data_name}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",       # 매 epoch마다 평가
        save_strategy="no",          # 체크포인트 저장 안 함 (디스크 절약)
        logging_steps=200,
        report_to="none",
        fp16=torch.cuda.is_available(),  # GPU면 혼합정밀(16bit) -> 속도↑ 메모리↓
    )

    # 동적 패딩: 배치 안에서 가장 긴 문장에 맞춰 패딩 (메모리 효율↑)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -- 5. Trainer 생성 (학습/평가를 자동 관리) --
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # -- 6. 학습 + 평가 --
    trainer.train()
    metrics = trainer.evaluate()

    print(f"\n{model_name} x {data_name} 결과:")
    print(f"   Acc {metrics['eval_accuracy']:.4f} | F1 {metrics['eval_f1']:.4f} "
          f"| P {metrics['eval_precision']:.4f} | R {metrics['eval_recall']:.4f}")

    # -- 7. 결과 저장 --
    save_result({
        "model": model_name, "dataset": data_name,
        "accuracy": round(metrics["eval_accuracy"], 4),
        "f1": round(metrics["eval_f1"], 4),
        "precision": round(metrics["eval_precision"], 4),
        "recall": round(metrics["eval_recall"], 4),
    })

    # -- 8. 메모리 정리 (다음 모델 학습 위해 GPU 비움) --
    del model, trainer
    torch.cuda.empty_cache()


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("GPU가 없습니다. 학습이 매우 느릴 수 있습니다.")

    total = len(MODELS) * len(DATASETS)  # 8회
    count = 0

    # 데이터 2개 x 모델 4개 = 8회 반복
    for data_name, (train_file, test_file) in DATASETS.items():
        for model_name, model_path in MODELS.items():
            count += 1
            print(f"\n\n[{count}/{total}] {model_name} x {data_name}")

            # 이미 한 조합이면 건너뛰기 (재개용)
            if already_done(model_name, data_name):
                print(f"   이미 완료됨, 건너뜀")
                continue

            try:
                train_one(model_name, model_path, data_name, train_file, test_file)
            except torch.cuda.OutOfMemoryError:
                # VRAM 부족 시 그 조합만 건너뛰고 계속
                print(f"   VRAM 부족! 이 조합은 건너뜁니다.")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"   오류: {e}")
                torch.cuda.empty_cache()

    # 전체 결과 출력
    print(f"\n\n{'='*60}\n전체 학습 완료! 최종 결과:\n{'='*60}")
    if os.path.exists(RESULTS_FILE):
        print(pd.read_csv(RESULTS_FILE).to_string(index=False))
        print(f"\n결과 저장: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

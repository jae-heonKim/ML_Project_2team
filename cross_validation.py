"""
============================================================
klue-roberta 5-Fold 교차검증 + 애매한 문장 테스트
============================================================
[교차검증이란]
학습/평가를 딱 한 번만 나눠서 하면 '운'이 섞일 수 있다.
(우연히 쉬운 데이터가 평가셋에 몰리면 점수가 높게 나오는 식)
-> 데이터를 5등분해서, 5번 번갈아 학습/평가하고 평균을 낸다.
   표준편차가 작으면 = 어떤 데이터 조합에서도 일관됨 = 신뢰할 수 있는 성능.

[추가로]
마지막에 애매한(긴급인지 비긴급인지 모호한) 문장들을 넣어서
모델이 경계선에서 어떻게 판단하는지(확률이 0.5 근처인지) 확인한다.

[출력] cv_results.csv (fold별 점수), ambiguous_test.csv (애매한 문장 결과)
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
from sklearn.model_selection import StratifiedKFold   # 라벨 비율 유지하며 5등분
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_PATH = "klue/roberta-base"
TRAIN_FILE = "dedup_balanced_train.csv"
TEST_FILE = "dedup_balanced_test.csv"
MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
N_FOLDS = 5          # 5등분
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    print(f"Device: {DEVICE}\n")

    # 균형 데이터의 train+test를 다시 합친다 (5-fold가 알아서 나눌 거라 통째로 사용)
    df = pd.concat([pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)], ignore_index=True)
    df["text"] = df["text"].astype(str)
    print(f"전체 데이터: {len(df)}개 (5-fold로 분할)")
    print(f"   라벨 분포: {df['label'].value_counts().to_dict()}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    # StratifiedKFold: 긴급/비긴급 비율을 유지하면서 5등분
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    X = df["text"].values
    y = df["label"].values

    # ===== 5번 반복: 매번 다른 1/5을 평가셋으로 사용 =====
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*55}\nFold {fold}/{N_FOLDS}\n{'='*55}")

        # 이번 fold의 학습셋(4/5)과 평가셋(1/5) 분리
        tr_df = df.iloc[tr_idx][["text", "label"]]
        va_df = df.iloc[va_idx][["text", "label"]]

        tr_ds = Dataset.from_pandas(tr_df).map(tok, batched=True)
        va_ds = Dataset.from_pandas(va_df).map(tok, batched=True)

        # 매 fold마다 모델을 새로 초기화해서 학습 (이전 학습 영향 X)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

        args = TrainingArguments(
            output_dir=f"./ckpt_cv_fold{fold}",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=16,
            learning_rate=LR, weight_decay=0.01,
            eval_strategy="no", save_strategy="no",
            logging_steps=500, report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=tr_ds, eval_dataset=va_ds,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        trainer.train()
        m = trainer.evaluate()   # 이번 fold의 점수

        print(f"  Fold {fold}: Acc={m['eval_accuracy']:.4f} F1={m['eval_f1']:.4f} "
              f"P={m['eval_precision']:.4f} R={m['eval_recall']:.4f}")
        fold_results.append({
            "fold": fold,
            "accuracy": m["eval_accuracy"], "f1": m["eval_f1"],
            "precision": m["eval_precision"], "recall": m["eval_recall"],
        })

        del model, trainer
        torch.cuda.empty_cache()

    # ===== 5개 fold 결과 집계 (평균 ± 표준편차) =====
    res = pd.DataFrame(fold_results)
    res.to_csv("cv_results.csv", index=False, encoding="utf-8-sig")

    print(f"\n\n{'='*55}\n5-Fold 교차검증 결과\n{'='*55}")
    print(res.to_string(index=False))
    print(f"\n{'-'*55}")
    for metric in ["accuracy", "f1", "precision", "recall"]:
        mean = res[metric].mean()
        std = res[metric].std()   # 표준편차: 작을수록 안정적
        print(f"  {metric:<12}: {mean:.4f} +- {std:.4f}")
    print(f"{'-'*55}")
    print("-> 표준편차가 작을수록 성능이 안정적 (운에 좌우되지 않음)")

    # ===== 애매한 문장 테스트 =====
    # 전체 데이터로 모델 하나를 학습한 뒤, 경계선 문장들을 넣어본다.
    print(f"\n\n{'='*55}\n애매한(경계선) 문장 테스트\n{'='*55}")
    print("전체 데이터로 모델 1개 학습 후 테스트...\n")

    full_ds = Dataset.from_pandas(df[["text", "label"]]).map(tok, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    args = TrainingArguments(
        output_dir="./ckpt_cv_full", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, learning_rate=LR, weight_decay=0.01,
        save_strategy="no", logging_steps=500, report_to="none", fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=full_ds,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    trainer.train()

    # 일부러 애매하게 만든 문장들 (사람도 헷갈릴 만한 것)
    ambiguous = [
        "제목: 회의 일정 관련\n본문: 다음 주 회의 언제가 좋으실까요? 시간 되실 때 알려주세요.",
        "제목: 보고서 검토 부탁\n본문: 첨부한 보고서 한번 봐주시면 감사하겠습니다. 급한 건 아닙니다.",
        "제목: 시스템 점검 예정\n본문: 이번 주말 시스템 점검이 예정되어 있으니 참고 바랍니다.",
        "제목: 결제 오류 문의\n본문: 어제 결제가 두 번 된 것 같은데 확인 가능하실까요?",
        "제목: 프로젝트 진행 상황\n본문: 현재 진행률은 60% 정도입니다. 일정대로 가고 있습니다.",
        "제목: 답변 기다리고 있습니다\n본문: 지난번 문의드린 건 어떻게 되었는지 궁금합니다.",
        "제목: 내일까지 자료 부탁\n본문: 가능하시면 내일 오전까지 자료 공유 부탁드려요.",
        "제목: 단순 공유\n본문: 참고하실 만한 기사 링크 보내드립니다.",
    ]

    model.eval()
    print("(확률이 0.5 근처일수록 모델이 애매해하는 문장)\n")
    rows = []
    for s in ambiguous:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            logit = model(**inputs).logits.cpu().numpy()
        prob = softmax(logit)[0, 1]
        label = "긴급" if prob >= 0.5 else "비긴급"
        title = s.split("\n")[0].replace("제목: ", "")
        marker = "  <- 애매" if 0.35 <= prob <= 0.65 else ""
        print(f"  [{label:>3}] 긴급확률 {prob:.3f} | {title}{marker}")
        rows.append({"문장": title, "긴급확률": round(float(prob), 3), "분류": label})

    pd.DataFrame(rows).to_csv("ambiguous_test.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: cv_results.csv, ambiguous_test.csv\n전체 완료!")


if __name__ == "__main__":
    main()

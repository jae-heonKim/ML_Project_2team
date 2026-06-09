# 이메일 긴급도 분류 모델 개선

Enron 이메일 데이터셋을 기반으로 한국어 이메일의 긴급/비긴급을 분류하는 모델을 개선한 프로젝트입니다.
기존 팀 발표 모델(F1 0.852)을 데이터 품질 개선을 통해 F1 0.901로 향상시켰습니다.

## 핵심 결과

| 구분 | F1-Score |
|------|----------|
| 기존 (모델 최적화) | 0.852 |
| 개선 (데이터 품질) | **0.901** |

**핵심: 모델 선택보다 데이터 품질이 성능을 좌우한다.**

## 프로젝트 개요

이메일을 긴급/비긴급으로 자동 분류하고, FastAPI 서버로 배포하여
n8n 워크플로우(Gmail 수신 → 긴급도 판정 → Telegram 알림)와 연동했습니다.

### 진행 단계

**STEP 1 — 모델 최적화 (기존 발표 단계)**
- 7개 한국어 모델 벤치마크 → RoBERTa(KLUE)가 F1 85.19%로 1위
- Large vs Base 비교 → 큰 모델이 항상 좋지 않음 (Base 선택)
- 앙상블 시도 → 자원 한계로 단일 모델 채택
- 임계값 튜닝
- **결론: 모델 최적화만으로는 F1 85% 수준에서 정체**

**STEP 2 — 데이터 품질 개선 (본 프로젝트)**
- 번역 모델 전환: NLLB-200 → GPT-4o-mini
- 잔여 노이즈(이메일 헤더·주소·제어문자) 정제
- 키워드 기반 라벨 → GPT 문맥 기반 재라벨링 (26.9% 변경)
- 중복 메일 제거 (data leakage 방지)
- 균형/불균형 데이터 비교 실험
- 5-Fold 교차검증으로 안정성 검증
- **결과: F1 0.852 → 0.901**

## 데이터셋

- **출처**: [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
- 영어 이메일을 한국어로 번역 후 긴급도 분류
- 최종 학습 데이터: 균형 9,808개 (긴급:비긴급 = 50:50)
- 데이터 파일은 용량 관계로 저장소에 포함하지 않았습니다.

## 코드 구성

| 파일 | 설명 |
|------|------|
| `clean_data.py` | 이메일 헤더·주소·제어문자 등 잔여 노이즈 정제 |
| `translate_and_label.py` | GPT-4o-mini로 번역 + 긴급도 라벨링 (병렬 처리, 체크포인트) |
| `train_models.py` | 4개 한국어 BERT 모델 × 균형/불균형 데이터 학습·비교 |
| `final_model_test.py` | 최종 모델(klue-roberta) 학습 및 저장 |
| `cross_validation.py` | 5-Fold 교차검증 + 경계선 문장 분석 |
| `model_api_v2.py` | FastAPI 기반 분류 API 서버 (n8n 연동) |

## 실행 순서

```
enron_quality_improved.csv (원본)
   ↓ clean_data.py
정제된 데이터
   ↓ translate_and_label.py
번역 + 라벨링 완료 데이터
   ↓ (중복 제거 + train/test 분할)
학습 데이터셋
   ↓ train_models.py
모델 비교 결과
   ↓ final_model_test.py
최종 모델 (final_model/)
   ↓ cross_validation.py
교차검증 결과
   ↓ model_api_v2.py
API 서버 배포 → n8n 연동
```

## 학습 설정

모든 모델을 동일 조건으로 학습하여 데이터 품질의 영향을 비교했습니다.

- 모델: klue/roberta-base (최종 선정)
- Epoch: 3, Batch size: 8, Learning rate: 2e-5, Max length: 256
- 평가 지표: Accuracy, F1, Precision, Recall

## 주요 발견

1. **데이터 품질 > 모델 선택**: 7개 모델이 모두 84~85% 구간에 분포 → 모델보다 데이터가 핵심
2. **정확도의 함정**: 불균형 데이터는 정확도 0.94지만 F1 0.78 → 균형 데이터 + F1 평가 채택
3. **Data Leakage 해소**: 중복 메일 제거로 과대평가된 성능(약 0.01)을 바로잡음
4. **성능 안정성**: 5-Fold 교차검증 F1 0.901 ± 0.004 (표준편차 작음)

## 환경

- Python 3.11
- PyTorch, Transformers, Datasets, scikit-learn

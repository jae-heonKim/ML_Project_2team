"""
============================================================
Enron 이메일 번역 + 긴급도 라벨링 (GPT-4o-mini)
============================================================
"""
import pandas as pd
import openai
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 설정
# ==========================================
OPENAI_API_KEY = "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"        # GPT API 키 입력
INPUT_FILE = "translate_input.csv" # [입력] translate_input.csv  (text = 정제된 영어 이메일)
OUTPUT_FILE = "translate_output.csv" # [출력]  translate_output.csv (+ label_new = GPT 라벨, text_ko = 한국어 번역)
MODEL = "gpt-4o-mini"
MAX_WORKERS = 10                 # 동시에 보낼 요청 수 (병렬 처리)
CHECKPOINT_EVERY = 500           # 몇 개 처리할 때마다 중간 저장할지
MAX_RETRIES = 5                  # 요청 1건당 최대 재시도 횟수

# OpenAI 클라이언트 생성
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 프롬프트: GPT에게 보낼 지시문
# ==========================================
def make_prompt(text):
    """영어 이메일 한 건을 받아서, GPT에게 줄 지시문(프롬프트)을 만든다."""
    return f"""다음 영어 이메일을 분석해줘.

이메일:
{text}

아래 JSON 형식으로만 답해줘. 다른 말은 절대 하지 마:
{{"label": 0 또는 1, "text_ko": "한국어 번역"}}

label 기준 (주어진 텍스트 내용만으로 판단):
- 1 (긴급): 즉각적인 조치나 응답이 필요 (마감 임박, 시스템 장애, 긴급 요청, 즉시 회신 요구 등)
- 0 (비긴급): 일반 업무, 정보 공유, 일상적 소통, 공지 등

번역은 자연스러운 한국어로. 제목과 본문을 모두 번역."""


def parse_json(content):
    """
    GPT 응답에서 JSON만 추출해서 파이썬 dict로 변환.
    GPT가 가끔 ```json ... ``` 코드블록으로 감싸서 주기 때문에 그 경우도 처리.
    """
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.lstrip().startswith("json"):
            content = content.lstrip()[4:]
    return json.loads(content.strip())


# ==========================================
# 단일 요청 처리 (재시도 로직 포함)
# ==========================================
def translate_one(idx, text):
    """
    이메일 1건을 GPT에 보내서 (라벨, 번역)을 받아온다.
    실패하면 MAX_RETRIES 번까지 재시도.
    반환: (행번호, 라벨, 번역문, 에러메시지)
    """
    for attempt in range(MAX_RETRIES):
        try:
            # GPT API 호출
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=1200,
                temperature=0,   # 0 = 항상 일관된 답 (창의성 X, 재현성 ↑)
                messages=[{"role": "user", "content": make_prompt(text)}]
            )
            # 응답에서 JSON 파싱 -> 라벨과 번역 추출
            result = parse_json(resp.choices[0].message.content)
            return idx, result.get("label"), result.get("text_ko"), None

        except openai.RateLimitError:
            # 429 에러(요청 과다): 점점 더 오래 기다렸다 재시도 (지수 백오프)
            # 1초 -> 2초 -> 4초 -> 8초 ... 최대 30초
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
        except (openai.APITimeoutError, openai.APIConnectionError):
            # 네트워크 문제: 마찬가지로 기다렸다 재시도
            time.sleep(min(2 ** attempt, 30))
        except json.JSONDecodeError as e:
            # GPT가 JSON 형식을 안 지킨 경우: 1번만 더 시도하고 포기
            if attempt >= 1:
                return idx, None, None, f"JSON parse error: {str(e)[:50]}"
            time.sleep(1)
        except Exception as e:
            # 그 외 예상 못한 에러: 즉시 포기하고 에러 기록
            return idx, None, None, str(e)[:80]
    return idx, None, None, "max retries exceeded"


# ==========================================
# 메인
# ==========================================
def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"입력: {len(df)}개")

    # -- 체크포인트 이어하기 --
    # 이미 OUTPUT_FILE이 있으면(이전에 돌리다 끊긴 경우)
    # 거기서부터 이어서 진행한다.
    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        if "label_new" not in done_df.columns:
            done_df["label_new"] = None
            done_df["text_ko"] = None
        df = done_df
        done_mask = df["label_new"].notna()
        print(f"체크포인트 발견: {done_mask.sum()}개 완료됨, 나머지 이어서 진행")
    else:
        df["label_new"] = None
        df["text_ko"] = None

    # 아직 처리 안 된 행만 골라낸다 (label_new가 비어있는 것)
    todo = df[df["label_new"].isna()]
    todo_indices = todo.index.tolist()
    print(f"처리할 항목: {len(todo_indices)}개 (병렬 {MAX_WORKERS}개)\n")

    if len(todo_indices) == 0:
        print("이미 전부 완료됨!")
        return

    start = time.time()
    completed = 0
    error_count = 0

    # -- 병렬 처리 시작 --
    # ThreadPoolExecutor: 동시에 MAX_WORKERS개의 요청을 보낸다.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 모든 할 일을 executor에 제출 (futures = 진행중인 작업들)
        futures = {executor.submit(translate_one, idx, df.at[idx, "text"]): idx
                   for idx in todo_indices}

        # 완료되는 순서대로 결과를 받는다
        for future in as_completed(futures):
            idx, label, text_ko, error = future.result()
            completed += 1

            if error:
                # 실패한 건 label_new = -1로 표시 (나중에 재실행하면 재시도됨)
                error_count += 1
                df.at[idx, "label_new"] = -1
                df.at[idx, "text_ko"] = f"ERROR: {error}"
            else:
                # 성공: 라벨과 번역 저장
                df.at[idx, "label_new"] = label
                df.at[idx, "text_ko"] = text_ko

            # 500개마다 중간 저장 + 진행상황/남은시간 출력
            if completed % CHECKPOINT_EVERY == 0:
                df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                elapsed = time.time() - start
                rate = completed / elapsed
                remaining = (len(todo_indices) - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{len(todo_indices)} 완료 "
                      f"| 에러 {error_count}개 "
                      f"| 경과 {elapsed/60:.1f}분 "
                      f"| 남은시간 약 {remaining/60:.0f}분")

    # 최종 저장
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    elapsed = time.time() - start

    print(f"\n{'='*50}")
    print(f"완료! 총 {completed}개 처리 / 소요 {elapsed/60:.1f}분 / 에러 {error_count}개")

    # 결과 요약 (정상 처리된 것의 라벨 분포)
    valid = df[df["label_new"].isin([0, 1])]
    print(f"\n재라벨링 결과 (정상 {len(valid)}개):")
    print(f"   라벨 분포: {valid['label_new'].value_counts().to_dict()}")
    if error_count > 0:
        print(f"\n에러 {error_count}개는 label_new=-1. 다시 실행하면 그것만 재시도함.")


if __name__ == "__main__":
    main()

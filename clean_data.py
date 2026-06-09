"""
============================================================
Enron 이메일 데이터 정제
============================================================
[목적]
기존 발표에서 만든 enron_quality_improved.csv 에는 정제가 됐다고는 하나
이메일 헤더(From/To/Subject), 이메일 주소, 제어문자 등이 여전히 남아있었음.
→ 이 잔여 노이즈를 추가로 제거해서 번역/학습 품질을 높이는 것이 목적.

[입력]  enron_quality_improved.csv  (컬럼: text, label)
[출력]  enron_cleaned.csv           (컬럼: text_original, text, label, too_short)
        - text_original : 원본 백업 (정제 전)
        - text          : 정제된 본문 (이후 번역에 사용)
        - too_short     : 정제 후 30자 미만이면 True (자동생성 로그성 메일 식별용)

============================================================
"""
import pandas as pd
import re

INPUT_FILE = "enron_quality_improved.csv" 
OUTPUT_FILE = "enron_cleaned.csv"
MIN_LENGTH = 30  # 정제 후 이 길이 미만이면 too_short 플래그를 단다


def clean_strong(text):
    """
    이메일 한 건의 텍스트를 받아서 헤더/메타정보를 제거하고
    실제 본문만 남겨서 반환하는 함수.
    위에서부터 순서대로 단계별로 정제한다.
    """
    # 결측치(NaN)면 빈 문자열 반환
    if pd.isna(text):
        return ""
    t = str(text)

    # -- 1. 제어문자 제거 --------------------------------
    # \x00~\x1f 범위의 제어문자(깨진 글자, 널 문자 등)를 공백으로.
    # 예: "안녕\x01하세요" -> "안녕 하세요"
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', t)

    # -- 2. 구조 태그 변환 -------------------------------
    # 기존 데이터의 [SUBJECT], [BODY] 태그를 한국어 라벨로 바꾼다.
    # [BODY]는 뒤 단계(헤더 제거)에서 실수로 지워지지 않도록
    # 임시 마커(|||BODY|||)로 보호했다가 마지막에 복원한다.
    t = t.replace('[SUBJECT]', '제목:').replace('[BODY]', '|||BODY|||')

    # -- 3. 인용/전달(답장) 구분선 제거 ------------------
    # 답장할 때 딸려오는 원본 인용문을 제거한다.
    # 예: "-----Original Message-----", "----- Forwarded by ... -----"
    t = re.sub(r'-+\s*Original Message\s*-+', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'-+\s*Forwarded by.*?-+', ' ', t, flags=re.IGNORECASE | re.DOTALL)
    # ">>> ... AM/PM" 형태의 인용 라인도 제거
    t = re.sub(r'>{2,}.*?(AM|PM|>>>)', ' ', t)

    # -- 4. 이메일 헤더 블록 제거 ------------------------
    # "From: ... To: ... Subject: ..." 처럼 헤더가 통째로 붙어있는 블록을 제거.
    # (.*? 와 DOTALL로 From부터 Subject까지 한 덩어리로 잡아서 지움)
    t = re.sub(r'From:.*?Subject:[^\n|]*', ' ', t, flags=re.IGNORECASE | re.DOTALL)
    # 위에서 못 잡은 개별 헤더 라인(From:, To:, Cc: 등)도 한 줄씩 제거
    t = re.sub(r'\b(From|To|Cc|Bcc|Sent|Date|Subject):\s*[^\n|]*', ' ', t, flags=re.IGNORECASE)

    # -- 5. 이메일 주소 제거 -----------------------------
    # 예: "hong@enron.com" -> " "
    t = re.sub(r'\S+@\S+', ' ', t)

    # -- 6. 이름/조직 잔재 제거 --------------------------
    # Enron 데이터 특유의 "이름/ENRON", ".../HOU", ".../ECT" 형태 제거
    # 예: "John Smith/ENRON" -> " "
    t = re.sub(r'\b[\w.]+\s*/\s*(ENRON|HOU|Corp|ECT|NA)\S*', ' ', t, flags=re.IGNORECASE)

    # -- 7. 날짜/시간 잔재 제거 --------------------------
    # 예: "12/25/2001 09:30 AM" -> " "
    t = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}\s*\d{0,2}:?\d{0,2}\s*(AM|PM)?', ' ', t)

    # -- 8. 마커 복원 + 최종 공백 정리 -------------------
    # 2단계에서 보호했던 본문 마커를 "본문:" 으로 복원
    t = t.replace('|||BODY|||', '\n본문:')
    t = re.sub(r'\?{3,}', ' ', t)        # "???????" 같은 깨진 기호 묶음 제거
    t = re.sub(r'[ \t]+', ' ', t)        # 연속된 공백/탭을 하나로
    t = re.sub(r'\n\s*\n+', '\n', t)     # 연속된 빈 줄을 하나로
    t = t.strip()                        # 앞뒤 공백 제거
    return t


def main():
    # CSV 로드
    print(f"{INPUT_FILE} 로드 중...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   총 {len(df)}개 행")

    # 원본은 text_original로 백업하고, 정제 결과를 새 text 컬럼에 저장
    print("정제 중...")
    df = df.rename(columns={"text": "text_original"})
    df["text"] = df["text_original"].apply(clean_strong)  # 모든 행에 정제 함수 적용

    # 정제 후 너무 짧아진(=내용 거의 없는) 데이터에 플래그
    df["too_short"] = df["text"].str.len() < MIN_LENGTH

    # -- 정제 효과 통계 출력 --
    orig_len = df["text_original"].str.len()
    clean_len = df["text"].str.len()
    removed_pct = (1 - clean_len / orig_len) * 100  # 글자가 몇 % 줄었나

    print("\n=== 정제 결과 ===")
    print(f"  평균 제거율: {removed_pct.mean():.1f}%")
    print(f"  정제 후 평균 길이: {clean_len.mean():.0f}자 (원본 {orig_len.mean():.0f}자)")
    print(f"  too_short({MIN_LENGTH}자 미만) 플래그: {df['too_short'].sum()}개")

    # 컬럼 순서 정리 후 저장
    df = df[["text_original", "text", "label", "too_short"]]
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_FILE}")

    # 정제 전후 비교 샘플 5개 (눈으로 확인용)
    print("\n=== 정제 전후 샘플 ===")
    for i in range(5):
        print(f"\n[원본] {df['text_original'].iloc[i][:120]}")
        print(f"[정제] {df['text'].iloc[i][:120]}")


if __name__ == "__main__":
    main()

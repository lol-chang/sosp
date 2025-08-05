import pickle

def inspect_pickle_keys(path, index=0):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print("🔍 데이터 타입:", type(data))

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        print(f"📌 인덱스 {index}의 항목 key 목록:")
        print(list(data[index].keys()))

    # DataFrame일 때 컬럼명 출력
    elif 'pandas' in str(type(data)):
        print("📌 DataFrame 컬럼명:")
        print(data.columns.tolist())

    else:
        print("❌ 예상한 구조가 아닙니다 (list of dict 또는 DataFrame 형태 아님)")

if __name__ == "__main__":
    path = "./train.pkl"  # DataFrame 피클 파일 경로
    inspect_pickle_keys(path, index=0)

import pandas as pd

txt_path = "./MicroLens-100k_comment_en.txt"  # 원본 txt 파일 경로
output_csv = "./video_label.csv"

# 데이터 저장용 딕셔너리
video_comment_count = {}

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        video_id = parts[1]
        if video_id in video_comment_count:
            video_comment_count[video_id] += 1
        else:
            video_comment_count[video_id] = 1

# DataFrame 변환
df = pd.DataFrame(list(video_comment_count.items()), columns=["video_id", "comment_count"])

# 숫자형 video_id만 필터링
df = df[df["video_id"].str.isnumeric()]

# 정수형으로 변환 후 정렬
df["video_id"] = df["video_id"].astype(int)
df = df.sort_values(by="video_id", ascending=True)

# 저장
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 저장 완료 (video_id 기준 오름차순 정렬): {output_csv}")

import pickle
import pandas as pd
import math

def log2_of_label(label):
    if label <= 0:
        raise ValueError("the label must be a positive integer")
    return math.log2(label)

def add_label_log2_from_csv(pkl_path, csv_path, output_path):
    # 1) data.pkl 불러오기
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 2) 댓글 수 csv 불러오기 (video_id -> comment_count)
    df = pd.read_csv(csv_path)
    label_dict = dict(zip(df['video_id'].astype(str), df['comment_count']))

    # 3) 각 아이템에 댓글 수 할당 후 log2 변환 저장
    for item in data:
        item_id = str(item['item_id'])
        comment_count = label_dict.get(item_id, None)

        if comment_count is None:
            raise ValueError(f"item_id {item_id}에 대한 댓글 수가 CSV에 없습니다.")
        if comment_count <= 0:
            raise ValueError(f"item_id {item_id}의 댓글 수가 0 이하입니다: {comment_count}")

        item['label'] = log2_of_label(comment_count)

    # 4) 결과 저장
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"[완료] data.pkl에 댓글 수 기반 log2(label) 저장됨: {output_path}")

if __name__ == "__main__":
    pkl_path = "./data.pkl"
    csv_path = "/home/changjin/바탕화면/sosp2/MMRA/video_label.csv"
    output_path = "./data.pkl"
    add_label_log2_from_csv(pkl_path, csv_path, output_path)

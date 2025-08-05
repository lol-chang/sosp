import pandas as pd
import numpy as np
import pickle
import os
from textual_engineering import load_angle_bert_model, angle_bert_textual_feature_extraction
from visual_engineering import load_vit_model, vit_visual_feature_extraction
from tqdm import tqdm
import torch

# 1. 경로 설정
csv_path = "./MicroLens-50k_titles.csv"
frame_dir = "./data/MicroLens-100k/video_frames"
output_pkl_path = "data.pkl"
frames_per_video = 10

# 2. 데이터 불러오기
df = pd.read_csv(csv_path)
titles = df['title'].tolist()
item_ids = df['item'].tolist()  # id 리스트

# 3. 모델 준비
angle_model = load_angle_bert_model()
processor, vit_model = load_vit_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
vit_model.eval()

data = []

for idx, item_id in tqdm(list(enumerate(item_ids)), desc="전체 아이템 처리", unit="item"):
    title = titles[idx]
    # 텍스트 임베딩
    text_feat = angle_bert_textual_feature_extraction(angle_model, title)

    # 시각 임베딩: 프레임 여러 장에서 추출해서 [10, 768]로 스택
    visual_features = []
    for frame_idx in range(frames_per_video):
        frame_path = os.path.join(frame_dir, f"{item_id}_{frame_idx}.jpg")
        if not os.path.exists(frame_path):
            print(f"⚠️ 프레임 누락: {frame_path}")
            break  # 프레임 다 없으면 이 아이템 스킵
        feature = vit_visual_feature_extraction(processor, vit_model, frame_path, device=device)
        visual_features.append(feature)
    if len(visual_features) != frames_per_video:
        print(f"❌ 아이템 {item_id}는 프레임 {len(visual_features)}개라서 스킵")
        continue

    visual_feat_array = np.stack(visual_features)  # [10, 768]

    entry = {
        "item_id": str(item_id),
        "text": title,
        "textual_feature_embedding": text_feat,
        "visual_feature_embedding_cls": visual_feat_array
    }
    data.append(entry)

print(f"✅ 총 {len(data)}개 항목 생성됨")

# 4. 피클로 저장
with open(output_pkl_path, "wb") as f:
    pickle.dump(data, f)

print(f"✅ 저장 완료: {output_pkl_path}")
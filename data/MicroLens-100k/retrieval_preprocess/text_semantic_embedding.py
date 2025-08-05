import os
import pandas as pd
import numpy as np
import pickle
import torch
from angle_emb import AnglE, Prompts
from tqdm import tqdm

# 모델 로드
def loading_model():
    model_name = "WhereIsAI/UAE-Large-V1"  # Hugging Face 모델 경로
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"✅ UAE 모델 로딩 중... (device: {device})")
    angle = AnglE.from_pretrained(model_name, pooling_strategy='cls').to(device)
    # angle.set_prompt(prompt=Prompts.C) # 에러 발생해서 일단 제외 -> 없어도 된다고 함 (버전 이슈)
    return angle, device

# 텍스트 → 벡터
# def convert_text_to_embedding(angle, text):
#     vec = angle.encode({'text': text}, to_numpy=True)
#     return vec
def convert_text_to_embedding(angle, text):
    vec = angle.encode(text, to_numpy=True)  # 딕셔너리 제거!
    return vec


def main():
    caption_csv_path = "./blip_caption_results.csv"
    data_pkl_path = "./data.pkl"
    output_pkl_path = "./data.pkl"

    # 1. caption 불러오기
    caption_df = pd.read_csv(caption_csv_path)
    caption_df["item_id"] = caption_df["item_id"].astype(str)

    caption_dict = {}
    for _, row in caption_df.iterrows():
        item_id = row["item_id"]
        captions = [str(row[f"caption_{i}"]).strip() for i in range(10) if pd.notna(row[f"caption_{i}"])]
        caption_text = ". ".join(captions)
        caption_dict[item_id] = caption_text

    # 2. data.pkl 불러오기
    with open(data_pkl_path, "rb") as f:
        data = pickle.load(f)

    # 3. UAE 모델 로드
    angle, device = loading_model()

    # 4. 각 항목에 retrieval_feature 추가
    updated = 0
    for entry in tqdm(data, desc="Adding retrieval_feature"):
        item_id = entry["item_id"]
        if item_id not in caption_dict:
            print(f"⚠️ caption 누락: {item_id}")
            continue

        title = entry["text"]
        caption = caption_dict[item_id]
        full_text = f"{title}. {caption}"
        embedding = convert_text_to_embedding(angle, full_text)
        entry["retrieval_feature"] = embedding
        updated += 1

    print(f"✅ retrieval_feature 추가 완료: {updated}/{len(data)}")

    # 5. 저장
    with open(output_pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ 저장 완료: {output_pkl_path}")

if __name__ == "__main__":
    main()
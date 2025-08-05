import pickle
import numpy as np

def add_cls_mean_vector(pkl_path, output_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)  # list of dict 구조

    for i, item in enumerate(data):
        if 'visual_feature_embedding_cls' not in item:
            print(f"[경고] 인덱스 {i}에 'visual_feature_embedding_cls' 없음")
            continue

        embedding = np.array(item['visual_feature_embedding_cls'])  # shape (10, 1024)
        
        if embedding.ndim != 2 or embedding.shape[1] != 768:
            print(f"[경고] 인덱스 {i} embedding shape 이상함: {embedding.shape}")
            continue

        # 평균 벡터 계산
        mean_vector = embedding.mean(axis=0)  
        item['visual_feature_embedding_mean'] = mean_vector  # 새로운 키 추가

    # 저장
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"[완료] visual_feature_embedding_cls_mean 추가 후 저장됨: {output_path}")

# 실행 예시
if __name__ == "__main__":
    pkl_path = "./data.pkl"
    output_path = "./data.pkl"
    add_cls_mean_vector(pkl_path, output_path)

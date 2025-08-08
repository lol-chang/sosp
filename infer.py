import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model.MicroLens100k.MMRA import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.serialization.add_safe_globals([Model])
model = torch.load(
    './train_results/train_MMRA_MicroLens-100k_MSE_2025-08-06_19-39-25/trained_model/model_11.pth',
    map_location=device,
    weights_only=False
)
model.to(device)
model.eval()

df = pd.read_pickle('./data/MicroLens-100k/all_data.pkl')

@torch.no_grad()
def get_mmra_scores(row, model, device):
    visual_feature = torch.tensor(np.array(row['visual_feature_embedding_cls']), dtype=torch.float32).unsqueeze(0)
    textual_feature = torch.tensor(np.array(row['textual_feature_embedding']), dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    N = len(row['retrieved_item_id_list'])
    scores = []
    for i in range(N):
        retrieved_visual_feature = torch.tensor(np.array(row['retrieved_visual_feature_embedding_cls'])[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        retrieved_textual_feature = torch.tensor(np.array(row['retrieved_textual_feature_embedding'])[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        similarity = torch.tensor([row['retrieved_item_similarity_list'][i]], dtype=torch.float32).unsqueeze(0)
        retrieved_label = torch.tensor([row['retrieved_label'][i]], dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        vf = visual_feature.to(device)
        tf = textual_feature.to(device)
        rvf = retrieved_visual_feature.to(device)
        rtf = retrieved_textual_feature.to(device)
        sim = similarity.to(device)
        lbl = retrieved_label.to(device)
        output = model(
            visual_feature=vf,
            textual_feature=tf,
            similarity=sim,
            retrieved_visual_feature=rvf,
            retrieved_textual_feature=rtf,
            retrieved_label=lbl
        )
        score = output.cpu().squeeze().item()
        scores.append(score)
    return scores

recommend_col = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="각 id별 recommend 생성"):
    scores = get_mmra_scores(row, model, device)
    ids = row['retrieved_item_id_list']
    recommend = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    recommend_col.append(recommend)

df['recommend'] = recommend_col

# DataFrame → dict의 리스트로 저장하려면 to_dict('records')!
out_path = './data/MicroLens-100k/all_data_with_recommend.pkl'
df.to_pickle(out_path)
print("✅ recommend 컬럼이 추가된 DataFrame 저장 완료!")
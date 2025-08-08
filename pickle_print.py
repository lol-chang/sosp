import pickle
import pprint

path = "./data/MicroLens-100k/all_data_with_recommend.pkl"

with open(path, 'rb') as f:
    df = pickle.load(f)

# item_id 컬럼을 정수형으로 변환
df['item_id'] = df['item_id'].astype(int)

target_id = 8400

# 해당 item_id가 있는지 확인 후 recommend 리스트 출력
if target_id in df['item_id'].values:
    recommend_list = df.loc[df['item_id'] == target_id, 'recommend'].values[0]
    print(f"item_id {target_id}의 recommend 리스트:")
    pprint.pprint(recommend_list)
else:
    print(f"❌ item_id {target_id}가 데이터에 없습니다.")

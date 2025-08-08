import pandas as pd
import pickle

df = pd.read_csv('./stamp/filtered_15to30s.csv')

df['item_id'] = df['session_info'].str.extract(r'\(item: (\d+)\)').astype(int)
df['next_item'] = df['session_info'].str.extract(r'-> .* \(item: ([\d\.]+)\)').iloc[:,0].astype(float).astype('Int64')

df = df.sort_values(by=['user', 'session_info'])

first_items = df.groupby('user').first().reset_index()[['user', 'item_id']]
first_item_map = dict(zip(first_items['user'], first_items['item_id']))

with open("./data/MicroLens-100k/all_data_with_recommend.pkl", 'rb') as f:
    item_df = pickle.load(f)
item_df['item_id'] = item_df['item_id'].astype(int)

recommend_dict = item_df.set_index('item_id')['recommend'].apply(
    lambda recs: [int(r[0]) for r in recs]
).to_dict()

def cache_hit_first_view(row, first_item_map, recommend_dict):
    user = row['user']
    first_item = first_item_map.get(user, None)
    next_item = row['next_item']
    if pd.isna(next_item) or first_item is None:
        return False

    # next_item을 확실히 int로 변환 (nullable int는 float 타입도 있음)
    try:
        next_item_int = int(next_item)
    except:
        return False

    rec_list = recommend_dict.get(first_item, [])
    # 추천 리스트도 모두 int로 맞춰주기
    rec_list_int = [int(x) for x in rec_list]

    return next_item_int in rec_list_int

df['cache_hit'] = df.apply(lambda row: cache_hit_first_view(row, first_item_map, recommend_dict), axis=1)

df_check = df[df['next_item'].notna()]

df_check[['user', 'item_id', 'next_item', 'cache_hit']].to_csv('./all_range_cache_simulation.csv', index=False)

total_checks = len(df_check)
total_hits = df_check['cache_hit'].sum()
hit_rate = total_hits / total_checks if total_checks > 0 else 0

print(f"전체 캐시 체크 횟수: {total_checks}")
print(f"캐시 히트 횟수: {total_hits}")
print(f"캐시 히트율: {hit_rate:.4f}")

import pandas as pd
import pickle

file_infos = [
    ('./stamp/filtered_15to30s.csv', '15초 ~ 30초'),
    ('./stamp/filtered_15to60s.csv', '15초 ~ 60초'),
    ('./stamp/filtered_15to300s.csv', '15초 ~ 300초'),
]

# 피클 추천 리스트 불러오기
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
    try:
        next_item_int = int(next_item)
    except:
        return False
    rec_list = recommend_dict.get(first_item, [])
    rec_list_int = [int(x) for x in rec_list]
    return next_item_int in rec_list_int

summary_rows = []

for file_path, time_range in file_infos:
    print(f"==== Processing file: {file_path} ====")
    df = pd.read_csv(file_path)

    df['item_id'] = df['session_info'].str.extract(r'\(item: (\d+)\)').astype(int)
    df['next_item'] = df['session_info'].str.extract(r'-> .* \(item: ([\d\.]+)\)').iloc[:,0].astype(float).astype('Int64')

    df = df.sort_values(by=['user', 'session_info'])

    first_items = df.groupby('user').first().reset_index()[['user', 'item_id']]
    first_item_map = dict(zip(first_items['user'], first_items['item_id']))

    df['cache_hit'] = df.apply(lambda row: cache_hit_first_view(row, first_item_map, recommend_dict), axis=1)

    df_check = df[df['next_item'].notna()]

    total_checks = len(df_check)
    total_hits = df_check['cache_hit'].sum()
    hit_rate = total_hits / total_checks if total_checks > 0 else 0

    print(f"전체 캐시 체크 횟수: {total_checks:,}")
    print(f"캐시 히트 횟수: {total_hits:,}")
    print(f"캐시 히트율: {hit_rate:.4f}\n")

    summary_rows.append({
        'File': file_path.split('/')[-1],
        'Time Range': time_range,
        'Total Checks': total_checks,
        'Total Hits': total_hits,
        'Hit Rate': hit_rate
    })

print("===== Summary =====")
header = f"{'File':<28} {'Time Range':<15} {'Total Checks':>15} {'Total Hits':>13} {'Hit Rate':>11}"
print(header)
print("-" * len(header))

for row in summary_rows:
    print(f"{row['File']:<28} {row['Time Range']:<15} {row['Total Checks']:>15,} {row['Total Hits']:>13,} {row['Hit Rate']:>11.4f}")

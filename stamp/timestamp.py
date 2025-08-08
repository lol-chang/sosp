import pandas as pd

df = pd.read_csv('./MicroLens-100k_pairs.csv')
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.sort_values(by=['user', 'datetime'])

# 다음 영상과의 시청 간격(초) 계산
df['next_datetime'] = df.groupby('user')['datetime'].shift(-1)
df['next_item'] = df.groupby('user')['item'].shift(-1)

df['gap_to_next_sec'] = (df['next_datetime'] - df['datetime']).dt.total_seconds()

# 10초 이상 30초 이하인 기록만 필터링
df_filtered = df[(df['gap_to_next_sec'] >= 15) & (df['gap_to_next_sec'] <= 300)]

# 원하는 형태로 포맷팅
df_filtered['session_info'] = df_filtered.apply(
    lambda r: f"{r['datetime'].strftime('%Y-%m-%d %H:%M:%S')} (item: {r['item']}) -> {r['next_datetime'].strftime('%Y-%m-%d %H:%M:%S')} (item: {r['next_item']})",
    axis=1
)

output_df = df_filtered[['user', 'session_info']]

output_df.to_csv('./stamp/filtered_15to300s.csv', index=False, header=['user', 'session_info'])

print(output_df)

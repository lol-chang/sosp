import pandas as pd

# CSV 불러오기
df = pd.read_csv('./MicroLens-100k_pairs.csv')

# timestamp -> datetime 변환
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# user, datetime 기준 정렬
df = df.sort_values(by=['user', 'datetime'])

# 시청 간격 계산
df['gap_sec'] = df.groupby('user')['datetime'].diff().dt.total_seconds()

# 10분 넘으면 새 세션 시작 표시
df['new_session'] = (df['gap_sec'] > 600) | (df['gap_sec'].isna())

# session_id 생성
df['session_id'] = df.groupby('user')['new_session'].cumsum()

# 세션별 기록 수 계산
session_sizes = df.groupby(['user', 'session_id']).size().reset_index(name='count')

# 2개 이상 기록 있는 세션만 필터링
valid_sessions = session_sizes[session_sizes['count'] > 1]

# 원본에서 valid 세션만 필터링
df_valid = df.merge(valid_sessions[['user', 'session_id']], on=['user', 'session_id'])

# session별로 datetime + item 정보 포맷팅
def format_session_times(group):
    return '\n'.join([f"{dt.strftime('%Y-%m-%d %H:%M:%S')} (item: {item})" for dt, item in zip(group['datetime'], group['item'])])

session_groups = df_valid.groupby(['user', 'session_id']).apply(format_session_times).reset_index(name='session_times')

# user별로 여러 세션을 '\n\n'으로 합침
result = session_groups.groupby('user')['session_times'].apply('\n\n'.join).reset_index()

# CSV 저장
result.to_csv('./stamp/user_sessions_with_items.csv', index=False)

print(result)

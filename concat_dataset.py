import pandas as pd

train_path = './data/MicroLens-100k/train.pkl'
valid_path = './data/MicroLens-100k/valid.pkl'
test_path = './data/MicroLens-100k/test.pkl'

# 데이터 로드
df_train = pd.read_pickle(train_path)
df_valid = pd.read_pickle(valid_path)
df_test = pd.read_pickle(test_path)

# 데이터 합치기
df_all = pd.concat([df_train, df_valid, df_test], axis=0).reset_index(drop=True)

# 합친 데이터 저장
all_data_path = './data/MicroLens-100k/all_data.pkl'
df_all.to_pickle(all_data_path)

print(f"All data saved to {all_data_path}")








import pandas as pd

all_data_path = './data/MicroLens-100k/all_data.pkl'
df_all = pd.read_pickle(all_data_path)

# item_id 기준 오름차순 정렬
df_all_sorted = df_all.sort_values(by='item_id').reset_index(drop=True)

# 정렬된 데이터 다시 저장
df_all_sorted.to_pickle(all_data_path)
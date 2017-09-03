import pandas as pd

train_data = pd.read_csv('train_reorder.csv', nrows=50)
user_log = pd.read_csv('user_log_ver2.csv', nrows=500)
train_data["visit_cat"] = '0'
train_data["visit_seller"] = '0'
train_data["visit_brand"] = '0'
train_data["buy_cat"] = '0'
train_data["buy_seller"] = '0'
train_data["buy_brand"] = '0'

for ix, row in train_data.iterrows():
    index = row["user_id"]
    train_data.iloc[ix, 3] = user_log.iloc[index-1, 1]
    train_data.iloc[ix, 4] = user_log.iloc[index - 1, 2]
    train_data.iloc[ix, 5] = user_log.iloc[index - 1, 3]
    train_data.iloc[ix, 6] = user_log.iloc[index - 1, 4]
    train_data.iloc[ix, 7] = user_log.iloc[index - 1, 5]
    train_data.iloc[ix, 8] = user_log.iloc[index - 1, 6]

    if ix % 200 == 0:
        print('processing %d' %int(ix/200))

train_data.to_csv('train_pair.csv', index=False)





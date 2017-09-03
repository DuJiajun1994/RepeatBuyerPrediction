import pandas as pd
import numpy as np

user_log = pd.read_csv('user_log_format1.csv')
user_log = user_log.drop('item_id', 1)
user_log = user_log.drop('cat_id', 1)
user_log = user_log.drop('seller_id', 1)
user_log = user_log.drop('brand_id', 1)
#7 months + 11.11 mul  4 action_type
result = [0] * (8*4+1)
Fin_result = [0] * 33
Fin_result = np.array(Fin_result)
user_id = user_log.iloc[0, 0]

for ix, row in user_log.iterrows():

    if ix == 0:
        index = (int(row["time_stamp"] / 100) - 5) * 4 + row["action_type"]
        result[index] += 1
        result[32] = user_id

    elif row["user_id"] == user_id:
        if row["time_stamp"] == 1111:
            index = int(28 + row["action_type"])
        else:
            index = (int(row["time_stamp"] / 100) - 5) * 4 + row["action_type"]
        result[index] += 1

    else:
        result = np.array(result)
        Fin_result = np.row_stack((Fin_result, result))
        result = [0] * (8 * 4+1)
        user_id = row["user_id"]
        if row["time_stamp"] == 1111:
            index = int(28 + row["action_type"])
        else:
            index = (int(row["time_stamp"] / 100) - 5) * 4 + row["action_type"]
        result[index] += 1
        result[32] = user_id

    if ix % 10000 == 0:
        print('processing %d w' % int(ix/10000))

Fin_result = np.row_stack((Fin_result, result))
Fin_result = np.delete(Fin_result, 0, axis=0)
Fin_result = Fin_result[np.lexsort(Fin_result.T)]
Fin_result = np.delete(Fin_result, [32], axis=1)
np.savetxt('user_date_ver2.txt', Fin_result, fmt="%d")


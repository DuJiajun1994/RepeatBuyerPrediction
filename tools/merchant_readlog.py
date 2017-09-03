import pandas as pd

merchant_ordered = pd.read_csv('mer_ord_ver1.csv', dtype=str)
final_result = pd.read_csv('mer_ord_ver1.csv', nrows=1, dtype=str)

temp_ser = pd.Series('0', index=["cat_id", "seller_id", "brand_id"],
                     name=0)

fin_i = 1

for ix, row in merchant_ordered.iterrows():
    #清洗NAN
    if isinstance(row['cat_id'], float):
        row['cat_id'] = '-1'
    if isinstance(row['brand_id'], float):
        row['brand_id'] = '-1'

    if ix == 0:
        temp_ser = row
        temp_ser.name = fin_i

    #id相同进行拼接
    elif row["seller_id"] == temp_ser["seller_id"]:
        temp_ser["cat_id"] += ',' + row["cat_id"]
        temp_ser["brand_id"] += ',' + row["brand_id"]
    #id不同更新临时series，把拼接结果append到最终结果
    else:
        final_result = final_result.append(temp_ser)
        fin_i += 1
        temp_ser = row
        temp_ser.name = fin_i

    if ix % 10000 == 0:
        print('processing %d w' % int(ix / 10000))


#拼接最后一个id
final_result = final_result.append(temp_ser)
#去除初始化时的第一条
final_result = final_result.drop(0)
final_result.to_csv('Mer_ord_ver1.csv', index=False)







'''
user_log = user_log.drop('item_id', 1)
user_log = user_log.drop('time_stamp', 1)
user_log = user_log.drop('action_type', 1)
user_log = user_log.drop('user_id', 1)

merchant_log = user_log.sort_values(by='seller_id')

merchant_log.to_csv('mer_ord_ver1.csv', index= False)
'''
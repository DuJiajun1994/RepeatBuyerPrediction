import pandas as pd
#该文件处理原始数据，把相同id的特征进行拼接
'''
def action_identify(tpseries):
    if tpseries["action_type"] == '0':
        tpseries["click_count"] += 1
    elif tpseries["action_type"] == '1':
        tpseries["add_cart_count"] += 1
    else:
        tpseries["add_fav_count"] += 1
    return tpseries

def action_id_cross(tpseries, row):
    if row["action_type"] == '0':
        tpseries["click_count"] += 1
    elif row["action_type"] == '1':
        tpseries["add_cart_count"] += 1
    else:
        tpseries["add_fav_count"] += 1
    return tpseries
'''

#读入原始数据
user_log = pd.read_csv('user_log_format1.csv', nrows=500, dtype=str)
#初始化一个同结构dataframe用于存储最终结果
final_result = pd.read_csv('user_log_format1.csv', nrows=1, dtype=str)
final_result['buy_cat'] = '-1'
final_result['buy_seller'] = '-1'
final_result['buy_brand'] = '-1'
final_result = final_result.drop('item_id', axis=1)
final_result = final_result.drop('time_stamp', axis=1)
'''
final_result['click_count'] = 0
final_result['add_cart_count'] = 0
final_result['buy_count'] = 0
final_result['add_fav_count'] = 0
'''
#初始化一个series用于临时比较
temp_ser = pd.Series('0', index=["user_id", "item_id", "cat_id", "seller_id", "brand_id", "time_stamp", "action_type"],
                     name=0)

#最终存储标号
fin_i = 1

for ix, row in user_log.iterrows():
    #清洗NAN
    if isinstance(row['cat_id'], float):
        row['cat_id'] = '-1'
    if isinstance(row['brand_id'], float):
        row['brand_id'] = '-1'

    if ix == 0:
        temp_ser = row
        temp_ser.name = fin_i
        temp_ser['buy_cat'] = '-1'
        temp_ser['buy_seller'] = '-1'
        temp_ser['buy_brand'] = '-1'
        temp_ser = temp_ser.drop('item_id')
        temp_ser = temp_ser.drop('time_stamp')
        '''
        temp_ser['click_count'] = 0
        temp_ser['add_cart_count'] = 0
        temp_ser['buy_count'] = 0
        temp_ser['add_fav_count'] = 0

        if temp_ser["action_type"] == '2':
            temp_ser["buy_count"] += 1
        else:
            temp_ser = action_identify(temp_ser)
        '''

    #id相同进行拼接
    elif row["user_id"] == temp_ser["user_id"]:
        if row["action_type"] == '2':
            temp_ser["buy_cat"] += ',' + row["cat_id"]
            temp_ser["buy_seller"] += ',' + row["seller_id"]
            temp_ser["buy_brand"] += ',' + row["brand_id"]
            #temp_ser["buy_count"] += 1
        else:
            temp_ser["cat_id"] += ',' + row["cat_id"]
            temp_ser["seller_id"] += ',' + row["seller_id"]
            temp_ser["brand_id"] += ',' + row["brand_id"]
            #temp_ser = action_id_cross(temp_ser, row)
    #id不同更新临时series，把拼接结果append到最终结果
    else:
        final_result = final_result.append(temp_ser)
        fin_i += 1
        temp_ser = row
        temp_ser.name = fin_i
        temp_ser['buy_cat'] = '-1'
        temp_ser['buy_seller'] = '-1'
        temp_ser['buy_brand'] = '-1'
        temp_ser = temp_ser.drop('item_id')
        temp_ser = temp_ser.drop('time_stamp')
        '''
        temp_ser['click_count'] = 0
        temp_ser['add_cart_count'] = 0
        temp_ser['buy_count'] = 0
        temp_ser['add_fav_count'] = 0
        if temp_ser["action_type"] == '2':
            temp_ser["buy_count"] += 1
        else:
            temp_ser = action_identify(temp_ser)
        '''

    if ix % 10000 == 0:
        print('processing %d w' % int(ix / 10000))


#拼接最后一个id
final_result = final_result.append(temp_ser)
#去除初始化时的第一条
final_result = final_result.drop(0)
final_result.to_csv('user_log_ver1.csv', index=False)


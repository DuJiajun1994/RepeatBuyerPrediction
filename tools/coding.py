import pandas as pd
import numpy as np

'''
#按merchant_id重排
mer_join = pd.read_csv('Mer_join_ver1.csv')
mer_join = mer_join.sort_values(by='seller_id')
mer_join.to_csv('Mer_join_ver2.csv', index=False)
'''
mer_join = pd.read_csv('Mer_join_ver2.csv')
#结果存储为numpy数组，前1671维为cat_id,后8477维为brand
result = [0] * (1671+8477)
result = np.array(result)

for ix, row in mer_join.iterrows():

    temp_cat_code = [0] * 1671
    temp_brand_code = [0] * 8477

    cat_ser = row["cat_id"]
    tp_cat_ser = cat_ser.split(',')
#处理string，对应位置加和
    for j in range(len(tp_cat_ser)):
        tp_cat_ser[j] = int(tp_cat_ser[j])
        if tp_cat_ser[j] != -1:
            temp_cat_code[tp_cat_ser[j]-1] += 1

    brand_ser = row["brand_id"]
    tp_brand_ser = brand_ser.split(',')

    for i in range(len(tp_brand_ser)):
        tp_brand_ser[i] = int(tp_brand_ser[i])
        if tp_brand_ser[i] != -1:
            temp_brand_code[tp_brand_ser[i]-1] += 1

#两个list拼接
    temp_result = temp_cat_code + temp_brand_code
    temp_result = np.array(temp_result)
    #array拼接
    result = np.row_stack((result, temp_result))
    if ix % 50 == 0:
        print('processing %d' % int(ix/50))

result = np.delete(result, 0, axis=0)
np.savetxt('mer_code_ver2.txt', result, fmt="%d")


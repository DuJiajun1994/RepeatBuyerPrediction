import pandas as pd

def droprepeat(stri):
    stri = stri.split(',')

    '''
    m = 0
    for l in range(len(stri)):
        if stri[l-m] == '-1':
            del stri[l-m]
            m += 1
    '''
    count = [1] * len(stri)
    cat_str = ''
    k = 1
    for i in range(len(stri) - 1, -1, -1):

        if stri[i] == '-1':
            del stri[i]
            del count[i]
            continue


        for j in range(0, len(stri) - k, 1):
            if stri[i] == stri[j]:
                del stri[i]
                count[j] += count[i]
                del count[i]
                k -= 1
                break
        k += 1

    for g in range(len(stri)):
        cat_str += stri[g] + '*' + str(count[g]) + ','

    if cat_str != '':
        cat_str = cat_str[:-1]


    return cat_str

data = pd.read_csv('user_log_ver2.csv')

for ix, row in data.iterrows():
    data.iloc[ix, 1] = droprepeat(row["cat_id"])
    data.iloc[ix, 2] = droprepeat(row["seller_id"])
    data.iloc[ix, 3] = droprepeat(row["brand_id"])
    data.iloc[ix, 4] = droprepeat(row["buy_cat"])
    data.iloc[ix, 5] = droprepeat(row["buy_seller"])
    data.iloc[ix, 6] = droprepeat(row["buy_brand"])

    if ix % 470 == 0:
        print('processing %d' %int(ix/470))

data.to_csv('user_log_count.csv', index=False)

import pandas as pd


def add_item(dic, item):
    if item in dic:
        dic[item] += 1
    else:
        dic[item] = 1


def print_info(dic):
    keys = list(dic.keys())
    keys.sort()
    for key in keys:
        print('{}\t{}'.format(key, dic[key]))

df = pd.read_csv('../data/user_log_format1.csv')

categories = {}
brands = {}
for ix, row in df.iterrows():
    if pd.notnull(row['cat_id']):
        cat_id = int(row['cat_id'])
        add_item(categories, cat_id)
    if pd.notnull(row['brand_id']):
        brand_id = int(row['brand_id'])
        add_item(brands, brand_id)
    if ix % 10000 == 0:
        print('processing %d w' % int(ix / 10000))

print('categories number: {}'.format(len(categories)))
print_info(categories)
print('brands number: {}'.format(len(brands)))
print_info(brands)


import pandas as pd


def group_and_count(df1, column_name):
    df2 = pd.DataFrame({column_name: df1[column_name], 'number': df1['item_id']})
    df3 = df2.groupby(column_name).count()
    print(df3)


df = pd.read_csv('../data/user_log_format1.csv')
group_and_count(df, 'brand_id')
group_and_count(df, 'cat_id')

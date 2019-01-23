# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_item = pd.read_csv('item.csv')
df_record = pd.read_csv('record.csv')
# df_data = pd.DataFrame(columns=['订单号', '商品名称', '商品一级分类', '订单创建时间'])

data_file = open('data.csv', 'w', encoding='utf-8')
data_file.write(','.join(['订单号', '商品名称', '商品一级分类', '订单创建时间']) + '\n')

record_time_dict = dict()

for index, row in df_record.iterrows():
    # print(row['订单创建时间'])
    # print(row['商品名称'])
    record_id = row['订单ID/采购单ID']
    record_time_dict[record_id] = row['订单创建时间']

    # item_names = row['商品名称'].split(';')
    # for item_name in item_names:
    #     temp = item_name.replace('(', ' ').replace(')', '')
    #     [name, quantity] = temp.split(' ')
    #     print(name, quantity)

for index, row in df_item.iterrows():
    record_id = row['订单号']
    item_name = str(row['商品名称'])
    item_type = str(row['商品一级分类'])
    if record_id not in record_time_dict:
        # print('error: record id not found')
        continue
    if item_name.startswith(('特价水果', '直接付款')):
        continue
    if '水果' not in item_type:
        continue
    data_file.write(','.join([
        str(record_id), str(item_name),
        str(item_type), str(record_time_dict[record_id])
    ]) + '\n')
    # df_data = df_data.append({
    #     '订单号': record_id,
    #     '商品名称': item_name,
    #     '商品一级分类': item_type,
    #     '订单创建时间': record_time_dict[record_id]
    # }, ignore_index=True)

# df_data.to_csv('data.csv', index=False)

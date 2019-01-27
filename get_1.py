# # -*- coding: UTF-8 -*-
import xlrd
import pandas as pd

import xlwt

#
# # df = pd.read_excel('MCM_NFLIS_Data.xlsx')
#
# from xlrd import open_workbook
# book = open_workbook('MCM_NFLIS_Data.xlsx')
# sheet = book.sheet_by_name('Sheet2')
# keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]
# dict_list =[]
#
# for row_index in range(1, sheet.nrows):
#     d= {keys[col_index]: sheet.cell(row_index, col_index).value
#         for col_index in range(sheet.ncols)}
#     dict_list.append(d)
#
# print(dict_list)
#
# #
# # data1 = pd.DataFrame(df, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
# # print(df)
# #
# # data1 = data1.sort_values(by='6')
# # data1.to_excel('string' + '.xls', sheet_name='string', encoding='utf-8')
#
#
# # data_file = open('data.csv', 'w', encoding='utf-8')
# # data_file.write(','.join(['code', 'number']) + '\n')

#
# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import xlrd
#
#
# def Read_execl_ranking(FileName, WorkTable='Sheet1'):
#     # 对数据排序，优化版本。
#     workbook = xlrd.open_workbook(FileName)
#     sheet_name = workbook.sheet_names()[1]
#     sheet = workbook.sheet_by_index(1)
#     sheet = workbook.sheet_by_name(WorkTable)
#     # print("各项排名！")
#     # print('-' * 20)
#     for j in range(1, sheet.ncols):
#         lie = {}
#         print(sheet.cell(9, j).value + "：")
#         for i in range(6, sheet.nrows):
#             lie[sheet.cell(i, 0).value] = sheet.cell(i, j).value
#             i = i + 1
#         newlie = sorted(lie.items(), key=lambda d: d[1], reverse=True)
#         for i in range(0, len(newlie)):
#             print(newlie[i][0] + ":" + str(newlie[i][1]) + "分，", end=' ')
#             i = i + 1
#         print("")
#
#
# if __name__ == "__main__":
#     FileName = 'MCM_NFLIS_Data.xlsx'
#     Read_execl_ranking(FileName=FileName, WorkTable='Data')


book = xlrd.open_workbook('MCM_NFLIS_Data.xlsx')
sheet = book.sheet_by_name('Data')
rows = sheet.nrows
columns = sheet.ncols
print(rows, columns)
# rowvalue=sheet.row_values(1)
dict1 = {}
for i in range(1, sheet.nrows):
    if sheet.cell(i, 5).value in dict1.keys():
        continue
    dict1[int(sheet.cell(i, 5).value)] = sheet.cell(i, 8).value

y = pd.DataFrame(list(dict1.items()), columns=['A', 'B'])
# df.sort(columns=['A'], axis=0, ascending=True)
y = y.sort_values(by='A', axis=0, ascending=True)
col = y.iloc[:, 1]
arrs = col.values

print(len(arrs))

# 输出结果

# print(arrs)

#
# frame = pd.DataFrame({"a": [9, 2, 5, 1], "b": [4, 7, -3, 2], "c": [6, 5, 8, 3]})
# frame.sort_values(by = 'a',axis = 0,ascending = True)

# # 功能：将一字典写入到csv文件中
# # 输入：文件名称，数据字典
# def createDictCSV(fileName="", dataDict={}):
#     with open(fileName, "wb") as csvFile:
#         csvWriter = csv.writer(csvFile)
#         for k,v in dataDict.iteritems():
#             csvWriter.writerow([k,v])
#         csvFile.close()


# colvalue1 = sheet.col_values(5)
# print(colvalue1)
# colvalue2 = sheet.col_values(8)
# print(colvalue2)
# cell1 = sheet.cell(1, 5)
# print(cell1)

# dic = {}
# print(sheet.cell(1, 9).value)
# for i in sheet.get_rows():
#     if sheet.cell(i, 8).value in dic.keys():
#         continue
#     dic[sheet.cell(i, 8).value] = sheet.cell(i, 5).value
#     break
#
# for name in dic.keys():
#     print(name)
#     print('\t')
#     print(dic[name])
#     print('\n')

# workbook = xlrd.open_workbook('MCM_NFLIS_Data.xlsx')
# sheet_name = workbook.sheet_names()[1]
# sheet = workbook.sheet_by_index(1)
# sheet = workbook.sheet_by_name('Data'/)


# print(sheet)


# df_data = pd.DataFrame()
#
# data_file = open('data.csv', 'w', encoding='utf-8')
# data_file.write(','.join(['订单号', '商品名称', '商品一级分类', '订单创建时间', '商品数量', '商品单位', '商品实收金额']) + '\n')
#
# record_time_dict = dict()
#
# for index, row in df_record.iterrows():
#     # print(row['订单创建时间'])
#     # print(row['商品名称'])
#     record_id = row['订单ID/采购单ID']
#     record_time_dict[record_id] = row['订单创建时间']
#
#     # item_names = row['商品名称'].split(';')
#     # for item_name in item_names:
#     #     temp = item_name.replace('(', ' ').replace(')', '')
#     #     [name, quantity] = temp.split(' ')
#     #     print(name, quantity)
#
# for index, row in df_item.iterrows():
#     record_id = str(row['订单号'])
#     item_name = str(row['商品名称'])
#     item_type = str(row['商品一级分类'])
#
#     item_number = row['商品数量']
#     item_unit = str(row['商品单位'])
#     item_total = row['商品实际成交金额']
#
#     if record_id not in record_time_dict:
#         # print('error: record id not found')
#         continue
#
#     if item_name.startswith(('特价', '直接付款')):
#         continue
#
#     if '水果' not in item_type:
#         continue
#     # if '广西沙糖桔' not in item_name:
#     #     continue
#     data_file.write(','.join([
#         str(record_id), str(item_name),
#         str(item_type), str(record_time_dict[record_id]),
#         str(item_number), str(item_unit), str(item_total)
#     ]) + '\n')
#

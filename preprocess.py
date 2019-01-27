# -*- coding: UTF-8 -*-


import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
# from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ACS_10_5YR_DP02_with_ann.csv', header=[0, 1])
df.columns = df.columns.get_level_values(0)
# df = df.filter(regex='^(HC01|GEO)', axis=1)
df = df.filter(regex='^(HC01)', axis=1)
# df.drop(['HC01_VC93'], axis=1)

#
# data = json.load(open('data.json'))
#
# N = len(df)
# M = int(N * 0.67)

D = df.shape[1]
N = len(df)
# print(D)
print(N)
# print('processing %d features' % D)
df = df.astype(float)
X = df.values

X = preprocessing.normalize(X, norm='l2')

pca = PCA(n_components=10)
# pca = PCA(n_components=10)
model = pca.fit(X)
print(pca.explained_variance_ratio_)
A = pca.transform(X)



from sklearn.linear_model import LinearRegression


# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# # y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
# reg.coef_
# reg.intercept_
# reg.predict(np.array([[3, 5]]))


#
# def meanX(dataX):
#     return np.mean(dataX,axis=0)#axis=0表示依照列来求均值。假设输入list,则axis=1
#
# def pca(XMat, k):
#     average = meanX(XMat)
#     m, n = np.shape(XMat)
#     data_adjust = []
#     avgs = np.tile(average, (m, 1))
#     data_adjust = XMat - avgs
#     covX = np.cov(data_adjust.T)   #计算协方差矩阵
#     featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
#     index = np.argsort(-featValue) #依照featValue进行从大到小排序
#     finalData = []
#     if k > n:
#         print( "k must lower than feature number")
#         return
#     else:
#         #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
#         selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
#         finalData = data_adjust * selectVec.T
#         reconData = (finalData * selectVec) + average
#     return finalData, reconData
#
#
# def loaddata(datafile):
#     return np.array(pd.read_csv(datafile,sep="\t",header=-1)).astype(np.float)
#
#
# def plotBestFit(data1, data2):
#     dataArr1 = np.array(data1)
#     dataArr2 = np.array(data2)
#
#     m = np.shape(dataArr1)[0]
#     axis_x1 = []
#     axis_y1 = []
#     axis_x2 = []
#     axis_y2 = []
#     for i in range(m):
#         axis_x1.append(dataArr1[i,0])
#         axis_y1.append(dataArr1[i,1])
#         axis_x2.append(dataArr2[i,0])
#         axis_y2.append(dataArr2[i,1])
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
#     ax.scatter(axis_x2, axis_y2, s=50, c='blue')
#     plt.xlabel('x1'); plt.ylabel('x2');
#     plt.savefig("outfile.png")
#     plt.show()
#
#
# def main():
#     # datafile = df
#     XMat = X
#     k = 2
#     return pca(XMat, k)
#
#
# if __name__ == "__main__":
#     finalData, reconMat = main()
#     plotBestFit(finalData, reconMat)







# df_record = pd.read_csv('record.csv')
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
#
#     # df_data = df_data.append({
#     #     '订单号': record_id,
#     #     '商品名称': item_name,
#     #     '商品一级分类': item_type,
#     #     '订单创建时间': record_time_dict[record_id]
#     # }, ignore_index=True)
#
# df_data.to_csv('data.csv', index=False)

# #
# # df_data = pd.read_csv('data.csv')
# # data_file1 = open('data1.csv', 'w', encoding='utf-8')
# # data_file.write(','.join(['商品名称', '商品一级分类', '订单创建时间', '商品数量', '商品单位', '商品实收金额']) + '\n')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('data.csv')
df['订单创建时间'] = df['订单创建时间'].map(lambda x: x[0:10])

df_orange = df[(df['商品名称'] == '海南圣女果') & (df['商品一级分类'] == '精品水果')]

g_items = df_orange.groupby('订单创建时间').sum()
df = pd.DataFrame(g_items)
print(df)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
df.plot(y='商品数量')
plt.show()

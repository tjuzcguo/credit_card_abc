import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy import stats
import copy

train_data = pd.read_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\cs-training.csv')
train_data = train_data.iloc[:, 1:]
# train_data.info()

# 处理缺失值
mData = train_data.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
train_known = mData[mData.MonthlyIncome.notnull()].values
train_unknown = mData[mData.MonthlyIncome.isnull()].values
train_X = train_known[:, 1:]
train_y = train_known[:, 0]
rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
rfr.fit(train_X, train_y)
predicted_y = rfr.predict(train_unknown[:, 1:]).round(0)
train_data.loc[train_data.MonthlyIncome.isnull(), 'MonthlyIncome'] = predicted_y

train_data = train_data.dropna()
train_data = train_data.drop_duplicates()

# train_box = train_data.iloc[:, [3, 7, 9]]
# train_box.boxplot()
# plt.show()

# 处理异常值
train_data = train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
train_data = train_data[train_data.age > 0]
train_data['SeriousDlqin2yrs'] = 1 - train_data['SeriousDlqin2yrs']

#cut training / test
from sklearn.model_selection import train_test_split

y = train_data.iloc[:, 0]
X = train_data.iloc[:, 1:]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
ntrain_data = pd.concat([train_y, train_X], axis=1)
ntest_data = pd.concat([test_y, test_X], axis=1)

ntrain_data.to_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\training.csv')
ntest_data.to_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\test.csv')

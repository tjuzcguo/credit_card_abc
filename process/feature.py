import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy import stats
import copy

from prepare.util import mono_bin

ntrain_data = pd.read_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\training.csv')

# ntrain_data.info()
# mi = ntrain_data[['MonthlyIncome']]
# sns.distplot(mi,bins=80)
# plt.show()

train_y = ntrain_data.iloc[:, 1]
train_X = ntrain_data.iloc[:, 2:]


def woe_value(d1):
    d2 = d1.groupby('Bucket', as_index=True)
    good = train_y.sum()
    bad = train_y.count() - good
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / good) / ((1 - d3['rate']) / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min')).reset_index(drop=True)
    woe = list(d4['woe'].round(3))
    return d4, iv, woe


x1_d, x1_iv, x1_cut, x1_woe = mono_bin(train_y, train_X.RevolvingUtilizationOfUnsecuredLines)

x2_d, x2_iv, x2_cut, x2_woe = mono_bin(train_y, train_X.age)

x4_d, x4_iv, x4_cut, x4_woe = mono_bin(train_y, train_X.DebtRatio)

x5_d, x5_iv, x5_cut, x5_woe = mono_bin(train_y, train_X.MonthlyIncome)

d1 = pd.DataFrame({"X": train_X['NumberOfTime30-59DaysPastDueNotWorse'], "Y": train_y})
d1['Bucket'] = d1['X']
d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
d1_x1.loc[:, 'Bucket'] = "(-inf,0]"

d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
d1_x2.loc[:, 'Bucket'] = "(0,1]"

d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 3)]
d1_x3.loc[:, 'Bucket'] = "(1,3]"

d1_x4 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
d1_x4.loc[:, 'Bucket'] = "(3,5]"

d1_x5 = d1.loc[(d1['Bucket'] > 5)]
d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])

x3_d, x3_iv, x3_woe = woe_value(d1)
x3_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]

d1 = pd.DataFrame({"X": train_X['NumberOfOpenCreditLinesAndLoans'], "Y": train_y})
d1['Bucket'] = d1['X']
d1_x1 = d1.loc[(d1['Bucket'] <= 1)]
d1_x1.loc[:, 'Bucket'] = "(-inf,1]"

d1_x2 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 2)]
d1_x2.loc[:, 'Bucket'] = "(1,2]"

d1_x3 = d1.loc[(d1['Bucket'] > 2) & (d1['Bucket'] <= 3)]
d1_x3.loc[:, 'Bucket'] = "(2,3]"

d1_x4 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
d1_x4.loc[:, 'Bucket'] = "(3,5]"

d1_x5 = d1.loc[(d1['Bucket'] > 5)]
d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])

x6_d, x6_iv, x6_woe = woe_value(d1)
x6_cut = [float('-inf'), 1, 2, 3, 5, float('+inf')]

d1 = pd.DataFrame({"X": train_X['NumberOfTimes90DaysLate'], "Y": train_y})
d1['Bucket'] = d1['X']
d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
d1_x1.loc[:, 'Bucket'] = "(-inf,0]"

d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
d1_x2.loc[:, 'Bucket'] = "(0,1]"

d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 3)]
d1_x3.loc[:, 'Bucket'] = "(1,3]"

d1_x4 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
d1_x4.loc[:, 'Bucket'] = "(3,5]"

d1_x5 = d1.loc[(d1['Bucket'] > 5)]
d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])

x7_d, x7_iv, x7_woe = woe_value(d1)
x7_cut = [float('-inf'), 0, 1, 3, 5, float('+inf')]

d1 = pd.DataFrame({"X": train_X['NumberRealEstateLoansOrLines'], "Y": train_y})
d1['Bucket'] = d1['X']
d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
d1_x1.loc[:, 'Bucket'] = "(-inf,0]"

d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
d1_x2.loc[:, 'Bucket'] = "(0,1]"

d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 2)]
d1_x3.loc[:, 'Bucket'] = "(1,2]"

d1_x4 = d1.loc[(d1['Bucket'] > 2) & (d1['Bucket'] <= 3)]
d1_x4.loc[:, 'Bucket'] = "(2,3]"

d1_x5 = d1.loc[(d1['Bucket'] > 3)]
d1_x5.loc[:, 'Bucket'] = "(3,+inf)"
d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x5])

x8_d, x8_iv, x8_woe = woe_value(d1)
x8_cut = [float('-inf'), 0, 1, 2, 3, float('+inf')]

d1 = pd.DataFrame({"X": train_X['NumberOfTime60-89DaysPastDueNotWorse'], "Y": train_y})
d1['Bucket'] = d1['X']
d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
d1_x1.loc[:, 'Bucket'] = "(-inf,0]"

d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
d1_x2.loc[:, 'Bucket'] = "(0,1]"

d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 3)]
d1_x3.loc[:, 'Bucket'] = "(1,3]"

d1_x5 = d1.loc[(d1['Bucket'] > 3)]
d1_x5.loc[:, 'Bucket'] = "(3,+inf)"
d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x5])

x9_d, x9_iv, x9_woe = woe_value(d1)
x9_cut = [float('-inf'), 0, 1, 3, float('+inf')]

d1 = pd.DataFrame({"X": train_X['NumberOfDependents'], "Y": train_y})
d1['Bucket'] = d1['X']
d1_x1 = d1.loc[(d1['Bucket'] <= 0)]
d1_x1.loc[:, 'Bucket'] = "(-inf,0]"

d1_x2 = d1.loc[(d1['Bucket'] > 0) & (d1['Bucket'] <= 1)]
d1_x2.loc[:, 'Bucket'] = "(0,1]"

d1_x3 = d1.loc[(d1['Bucket'] > 1) & (d1['Bucket'] <= 2)]
d1_x3.loc[:, 'Bucket'] = "(1,2]"

d1_x4 = d1.loc[(d1['Bucket'] > 2) & (d1['Bucket'] <= 3)]
d1_x4.loc[:, 'Bucket'] = "(2,3]"

d1_x6 = d1.loc[(d1['Bucket'] > 3) & (d1['Bucket'] <= 5)]
d1_x6.loc[:, 'Bucket'] = "(3,5]"

d1_x5 = d1.loc[(d1['Bucket'] > 5)]
d1_x5.loc[:, 'Bucket'] = "(5,+inf)"
d1 = pd.concat([d1_x1, d1_x2, d1_x3, d1_x4, d1_x6, d1_x5])

x10_d, x10_iv, x10_woe = woe_value(d1)
x10_cut = [float('-inf'), 0, 1, 2, 3, 5, float('+inf')]


# informationValue = []
# informationValue.append(x1_iv)
# informationValue.append(x2_iv)
# informationValue.append(x3_iv)
# informationValue.append(x4_iv)
# informationValue.append(x5_iv)
# informationValue.append(x6_iv)
# informationValue.append(x7_iv)
# informationValue.append(x8_iv)
# informationValue.append(x9_iv)
# informationValue.append(x10_iv)
# informationValue


# index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
# index_num = range(len(index))
# ax = plt.bar(index_num, informationValue, tick_label=index)
# plt.show()


def trans_woe(var, var_name, x_woe, x_cut):
    woe_name = var_name + '_woe'
    for i in range(len(x_woe)):
        if i == 0:
            var.loc[(var[var_name] <= x_cut[i + 1]), woe_name] = x_woe[i]
        elif (i > 0) and (i <= len(x_woe) - 2):
            var.loc[((var[var_name] > x_cut[i]) & (var[var_name] <= x_cut[i + 1])), woe_name] = x_woe[i]
        else:
            var.loc[(var[var_name] > x_cut[len(x_woe) - 1]), woe_name] = x_woe[len(x_woe) - 1]
    return var


x1_name = 'RevolvingUtilizationOfUnsecuredLines'
x2_name = 'age'
x3_name = 'NumberOfTime30-59DaysPastDueNotWorse'
x7_name = 'NumberOfTimes90DaysLate'
x9_name = 'NumberOfTime60-89DaysPastDueNotWorse'

# train_X = trans_woe(train_X, x1_name, x1_woe, x1_cut)
# train_X = trans_woe(train_X, x2_name, x2_woe, x2_cut)
# train_X = trans_woe(train_X, x3_name, x3_woe, x3_cut)
# train_X = trans_woe(train_X, x7_name, x7_woe, x7_cut)
# train_X = trans_woe(train_X, x9_name, x9_woe, x9_cut)
# train_X = train_X.iloc[:, -5:]

import statsmodels.api as sm
from sklearn.externals import joblib

# X1 = sm.add_constant(train_X)
# logit = sm.Logit(train_y, X1)
# result = logit.fit()
# print(result.summary())
# joblib.dump(result, r'D:\Users\zcguo\PycharmProjects\credit_score\model\logr.pkl')

# import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#
#
#
# lgb_train=lgb.Dataset(train_X,train_y)
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'binary_logloss'},
#     'num_leaves': 64,
#     'num_trees': 100,
#     'learning_rate': 0.01,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
#
#
# # number of leaves,will be used in feature transformation
# num_leaf = 64
#
# print('Start training...')
# # train
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=100,
#                 valid_sets=lgb_train)
#
# print('Save model...')
# # save model to file
# gbm.save_model('model.txt')
#
# print('Start predicting...')
# # predict and get data on leaves, training data
# y_pred = gbm.predict(train_X, pred_leaf=True)
#
#
#
#
# print(np.array(y_pred).shape)
# print(y_pred[:10])


result1 = joblib.load(r'D:\Users\zcguo\PycharmProjects\credit_score\model\logr.pkl')

test_data = pd.read_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\test.csv')

test_X = test_data.iloc[:, 2:]
test_y = test_data.iloc[:, 1]
scoreTestX = test_X[
    ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
     'NumberOfTime60-89DaysPastDueNotWorse']]

test_X = trans_woe(test_X, x1_name, x1_woe, x1_cut)
test_X = trans_woe(test_X, x2_name, x2_woe, x2_cut)
test_X = trans_woe(test_X, x3_name, x3_woe, x3_cut)
test_X = trans_woe(test_X, x7_name, x7_woe, x7_cut)
test_X = trans_woe(test_X, x9_name, x9_woe, x9_cut)

test_X = test_X.iloc[:, -5:]

X3 = sm.add_constant(test_X)
resu = result1.predict(X3)
dataframe = pd.DataFrame({'Probability': resu})
dataframe.index = np.arange(1, len(resu) + 1)
dataframe.to_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\result.csv', index=True)

recall = metrics.recall_score(test_y, resu.round())
acc = metrics.accuracy_score(test_y, resu.round())
confusion=confusion_matrix(test_y,resu.round())
print('----------------confusion matrix----------------')
print(confusion)
print(recall)
print(acc)
# fpr, tpr, threshold = metrics.roc_curve(test_y, resu)
# rocauc = metrics.auc(fpr, tpr)
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('TPR')
# plt.xlabel('FPR')
# plt.show()





p = 20 / np.log(2)
q = 600 - 20 * np.log(20) / np.log(2)


def get_score(coe, woe, factor):
    scores = []
    for w in woe:
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores


x_coe = [2.6084, 0.6327, 0.5151, 0.5520, 0.5747, 0.4074]
baseScore = round(q + p * x_coe[0], 0)

x1_score = get_score(x_coe[1], x1_woe, p)
x2_score = get_score(x_coe[2], x2_woe, p)
x3_score = get_score(x_coe[3], x3_woe, p)
x7_score = get_score(x_coe[4], x7_woe, p)
x9_score = get_score(x_coe[5], x9_woe, p)

score = [x1_score, x2_score, x3_score, x7_score, x9_score]

cut_t = [x1_cut, x2_cut, x3_cut, x7_cut, x9_cut]


def compute_score(x):
    tot_score = baseScore
    cut_d = copy.deepcopy(cut_t)
    for j in range(len(cut_d)):
        cut_d[j].append(x[j])
        cut_d[j].sort()
        for i in range(len(cut_d[j])):
            if cut_d[j][i] == x[j]:
                tot_score = score[j][i - 1] + tot_score
    return tot_score


scores = []
for i in range(scoreTestX.shape[0]):
    print(i)
    scores.append(compute_score(scoreTestX.iloc[i]))

rrguo = pd.DataFrame({'score': scores, 'probility': resu})
rrguo.index = np.arange(1, len(resu) + 1)
rrguo.to_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\score_prob.csv', index=True)

testSam1 = [0.3, 44, 3, 3, 5]
score1 = compute_score(testSam1)

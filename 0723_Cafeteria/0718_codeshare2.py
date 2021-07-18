import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#1. DATA
train = pd.read_csv('../data/DACON/cafeteria_prediction/train.csv')
test = pd.read_csv('../data/DACON/cafeteria_prediction/test.csv')
submission = pd.read_csv('../data/DACON/cafeteria_prediction/sample_submission.csv')

# 메뉴 이름 빼기
drops = ['조식메뉴', '중식메뉴', '석식메뉴']

train = train.drop(drops, axis=1)
test = test.drop(drops, axis=1)

# 요일 -> 숫자
train['월'] = pd.DatetimeIndex(train['일자']).month
test['월'] = pd.DatetimeIndex(test['일자']).month

train['일'] = pd.DatetimeIndex(train['일자']).day
test['일'] = pd.DatetimeIndex(test['일자']).day

weekday = {
    '월': 1,
    '화': 2,
    '수': 3,
    '목': 4,
    '금': 5
}

train['요일'] = train['요일'].map(weekday)
test['요일'] = test['요일'].map(weekday)

print(train.head())
#            일자  요일  본사정원수  본사휴가자수  본사출장자수  본사시간외근무명령서승인건수  현본사소속재택근무자수     중식계    석식계  월  일
# 0  2016-02-01   1   2601      50     150             238          0.0  1039.0  331.0  2  1
# 1  2016-02-02   2   2601      50     173             319          0.0   867.0  560.0  2  2
# 2  2016-02-03   3   2601      56     180             111          0.0  1017.0  573.0  2  3
# 3  2016-02-04   4   2601     104     220             355          0.0   978.0  525.0  2  4
# 4  2016-02-05   5   2601     278     181              34          0.0   925.0  330.0  2  5


# 본사정원수 - 휴가자 - 재택근무자
train['식사가능자수'] = train['본사정원수'] - train['본사휴가자수'] - train['현본사소속재택근무자수']
test['식사가능자수'] = test['본사정원수'] - test['본사휴가자수'] - test['현본사소속재택근무자수']

train['중식참여율'] = train['중식계'] / train['식사가능자수']
train['석식참여율'] = train['석식계'] / train['식사가능자수']

features = ['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']
labels = ['중식계',	'석식계', '중식참여율', '석식참여율']

train = train[features+labels]
test = test[features]

print(train.head())
#    월  일  요일  식사가능자수  본사출장자수  본사시간외근무명령서승인건수     중식계    석식계     중식참여율     석식참여율
# 0  2  1   1  2551.0     150             238  1039.0  331.0  0.407291  0.129753
# 1  2  2   2  2551.0     173             319   867.0  560.0  0.339867  0.219522
# 2  2  3   3  2545.0     180             111  1017.0  573.0  0.399607  0.225147
# 3  2  4   4  2497.0     220             355   978.0  525.0  0.391670  0.210252
# 4  2  5   5  2323.0     181              34   925.0  330.0  0.398192  0.142058

print(test.head())
#    월   일  요일  식사가능자수  본사출장자수  본사시간외근무명령서승인건수
# 0  1  27   3  2537.0     182               5
# 1  1  28   4  2531.0     212             409
# 2  1  29   5  2419.0     249               0
# 3  2   1   1  2494.0     154             538
# 4  2   2   2  2548.0     186             455

weekday_rank4dinner = {
    1: 1,
    2: 2,
    3: 5,
    4: 3,
    5: 4,
}

train['요일(석식)'] = train['요일'].map(weekday_rank4dinner)
test['요일(석식)'] = test['요일'].map(weekday_rank4dinner)

print(train.head())
#    월  일  요일  식사가능자수  본사출장자수  본사시간외근무명령서승인건수     중식계    석식계     중식참여율     석식참여율  요일(석식)
# 0  2  1   1  2551.0     150             238  1039.0  331.0  0.407291  0.129753       1
# 1  2  2   2  2551.0     173             319   867.0  560.0  0.339867  0.219522       2
# 2  2  3   3  2545.0     180             111  1017.0  573.0  0.399607  0.225147       5
# 3  2  4   4  2497.0     220             355   978.0  525.0  0.391670  0.210252       3
# 4  2  5   5  2323.0     181              34   925.0  330.0  0.398192  0.142058       4

print(test.head())
#    월   일  요일  식사가능자수  본사출장자수  본사시간외근무명령서승인건수  요일(석식)
# 0  1  27   3  2537.0     182               5       5
# 1  1  28   4  2531.0     212             409       3
# 2  1  29   5  2419.0     249               0       4
# 3  2   1   1  2494.0     154             538       1
# 4  2   2   2  2548.0     186             455       2

sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
sns.heatmap(data=train.corr(),square=True, annot=True, cbar=True)
# plt.show()

print("====================================================")

from sklearn.ensemble import RandomForestRegressor

# 대회 규칙
# 평가산식 : MAE(Mean Absolute Error)
lunch_model = RandomForestRegressor(criterion='mae')
dinner_model = RandomForestRegressor(criterion='mae')

# Model
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

params = {
    'learning_rate': [0.0, 0.1, 0.09, 0.089, 0.08],
    'boosting_type': ['gbtree', 'gblinear', 'dart'],
    'max_depth' : [3,4,5],
    'seed' : [42] 
}

lunch_r = XGBRegressor(objective='reg:squarederror', random_state=2021)
dinner_r = XGBRegressor(objective='reg:squarederror', random_state=2021)

lunch_model = GridSearchCV(lunch_r, params, scoring='neg_mean_absolute_error')
dinner_model = GridSearchCV(dinner_r, params, scoring='neg_mean_absolute_error')

# 중식
# x = train[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
# >>>> 상관관계 높은 것만 남김

x = train[['본사출장자수', '본사시간외근무명령서승인건수', '요일(석식)']]
y = train['중식계']

print(x.head())

lunch_model.fit(x, y)
print(lunch_model.best_score_, lunch_model.best_params_)
lunch_model = lunch_model.best_estimator_

test_x = test[['본사출장자수', '본사시간외근무명령서승인건수', '요일(석식)']]
y_pred = lunch_model.predict(test_x)
submission['중식계'] = y_pred

print("====================================================")

# 석식
x = train[['본사출장자수', '본사시간외근무명령서승인건수', '요일(석식)']]
y = train['석식계']

dinner_model.fit(x, y)
print(dinner_model.best_score_, dinner_model.best_params_) 
dinner_model = dinner_model.best_estimator_

test_x = test[['본사출장자수', '본사시간외근무명령서승인건수', '요일(석식)']]
y_pred = dinner_model.predict(test_x)
submission['석식계'] = y_pred

print("====================================================")

submission.to_csv('../data/DACON/cafeteria_prediction/sub/0718_submit3.csv', index=False)
print(submission.head())

# ====================================================
#            일자          중식계         석식계
# 0  2021-01-27   932.294983  362.368591
# 1  2021-01-28   951.179932  502.264954
# 2  2021-01-29   667.589783  365.089081
# 3  2021-02-01  1194.056763  554.667908
# 4  2021-02-02   983.956726  518.471680

# 0718_submit3.csv
# 101.7248546667
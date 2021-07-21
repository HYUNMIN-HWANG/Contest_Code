import pandas as pd
import numpy as np

#1. DATA
train = pd.read_csv('../data/DACON/cafeteria_prediction/train.csv')
test = pd.read_csv('../data/DACON/cafeteria_prediction/test.csv')
submission = pd.read_csv('../data/DACON/cafeteria_prediction/sample_submission.csv')

print(train.shape)  # (1205, 12)
print(test.shape)   # (50, 10)
print(submission.shape) # (50, 3)


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

# 본사정원수 - 휴가자 - 재택근무자
train['식사가능자수'] = train['본사정원수'] - train['본사휴가자수'] - train['현본사소속재택근무자수']
test['식사가능자수'] = test['본사정원수'] - test['본사휴가자수'] - test['현본사소속재택근무자수']

train['중식참여율'] = train['중식계'] / train['식사가능자수']
train['석식참여율'] = train['석식계'] / train['식사가능자수']

features = ['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']
labels = ['중식계',	'석식계', '중식참여율', '석식참여율']

train = train[features+labels]
test = test[features]

# print(train[:20])
# print(test.head())

# Modeling
from sklearn.ensemble import RandomForestRegressor

# 대회 규칙
# 평가산식 : MAE(Mean Absolute Error)
lunch_model = RandomForestRegressor(criterion='mae')
dinner_model = RandomForestRegressor(criterion='mae')

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

params = {
    'learning_rate': [0.0, 0.1, 0.09, 0.089, 0.08],
    'boosting_type': ['gbtree', 'gblinear', 'dart']
}

lunch_r = XGBRegressor(objective='reg:squarederror')
dinner_r = XGBRegressor(objective='reg:squarederror')

lunch_model = GridSearchCV(lunch_r, params, scoring='neg_mean_absolute_error')
dinner_model = GridSearchCV(dinner_r, params, scoring='neg_mean_absolute_error')


# 요일별로 오름차순
train = train.sort_values(by='요일', ascending=True)
test = test.sort_values(by='요일', ascending=True)
submission['요일'] = test['요일']   # submission 에도 '요일' 컬럼 추가
submission = submission.sort_values(by='요일', ascending=True)
# print(test[:10])
# print(submission[:10])

# 결과를 임시로 저장할 프레임
temp_result = pd.DataFrame(index=['일자','중식계','석식계','요일'])
print(temp_result.shape)
print(temp_result)


# 요일별로 분석하기
for day in range(1,6):

    params = {
        'learning_rate': [0.0, 0.1, 0.09, 0.089, 0.08],
        'boosting_type': ['gbtree', 'gblinear', 'dart']
    }

    lunch_r = XGBRegressor(objective='reg:squarederror')
    dinner_r = XGBRegressor(objective='reg:squarederror')

    lunch_model = GridSearchCV(lunch_r, params, scoring='neg_mean_absolute_error')
    dinner_model = GridSearchCV(dinner_r, params, scoring='neg_mean_absolute_error')

    # 요일별로 분석하기        
    print(day, "요일 데이터만 추출===================")

    train_day = train[train['요일'] == day]
    print(train_day.shape)
    # print(train_day[:10])
    # 1 요일 데이터만 추출
    # (241, 10)
    # 2 요일 데이터만 추출
    # (240, 10)
    # 3 요일 데이터만 추출
    # (239, 10)
    # 4 요일 데이터만 추출
    # (244, 10)
    # 5 요일 데이터만 추출
    # (241, 10)

    # 예측 데이터
    test_day = test[test['요일'] == day]
    print(test_day.shape)
    predict_day = submission[submission['요일'] == day]
    print(predict_day.shape)

    # 중식
    x = train_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y = train_day['중식계']
    print(x.shape, y.shape)

    lunch_model.fit(x, y)
    print(lunch_model.best_score_, lunch_model.best_params_)
    lunch_model = lunch_model.best_estimator_

    test_x = test_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y_pred = lunch_model.predict(test_x)
    predict_day['중식계'] = y_pred
    print(predict_day)

    # 중식
    x = train_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y = train_day['석식계']
    print(x.shape, y.shape)

    dinner_model.fit(x, y)
    print(dinner_model.best_score_, dinner_model.best_params_)
    dinner_model = dinner_model.best_estimator_

    test_x = test_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y_pred = dinner_model.predict(test_x)
    predict_day['석식계'] = y_pred
    print(predict_day)

    temp_result = pd.concat([temp_result, predict_day])

    # print(temp_result)

final_submission = temp_result.drop('요일', axis=1)
final_submission = final_submission.sort_values(by='일자', ascending=True)
final_submission = final_submission[:-4]
print(final_submission)


print("=========================================")

final_submission.to_csv('../data/DACON/cafeteria_prediction/sub/0721_submit1.csv', index=False)
print(final_submission.shape)
print(final_submission.head())
#            일자          중식계         석식계
# 0  2021-01-27  1074.606079  456.600281
# 1  2021-01-28   945.466675  437.640320
# 2  2021-01-29   534.854736  216.664780
# 3  2021-02-01  1096.341797  449.633301
# 4  2021-02-02   957.244629  552.495422

print(" 💯💯💯💯 Done!!! ") 

# 0721_submit1.csv
# 73.6761986667	
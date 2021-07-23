import pandas as pd
import numpy as np
import xgboost

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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Lasso

RANDOM_SEED = 2021
PROBAS = True
FOLDS = 8
N_ESTIMATORS = 1000
lasso = Lasso()

# 대회 규칙
# 평가산식 : MAE(Mean Absolute Error)
lunch_model = RandomForestRegressor(criterion='mae')
dinner_model = RandomForestRegressor(criterion='mae')


XGB_params = {
    'learning_rate': [0.0, 0.1, 0.09, 0.089, 0.08]
}

lgb_params = {
    'metric': 'l1',
    'n_estimators': N_ESTIMATORS,
    'objective': 'regression',
    'random_state': RANDOM_SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}

ctb_params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': RANDOM_SEED,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': N_ESTIMATORS,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED
}

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

    lunch_r = XGBRegressor(objective='reg:squarederror')
    dinner_r = XGBRegressor(objective='reg:squarederror')

    lunch_model = GridSearchCV(lunch_r, XGB_params, scoring='neg_mean_absolute_error')
    dinner_model = GridSearchCV(dinner_r, XGB_params, scoring='neg_mean_absolute_error')

    lunch_cat = CatBoostRegressor(**ctb_params)
    dinner_cat = CatBoostRegressor(**ctb_params)

    lunch_LGBM = LGBMRegressor(**lgb_params)
    dinner_LGBM = LGBMRegressor(**lgb_params)

    lunch_KN = KNeighborsRegressor(n_neighbors = 1)
    dinner_KN = KNeighborsRegressor(n_neighbors = 1)


    lunch_stack = StackingCVRegressor(regressors=(lunch_model, lunch_cat, lunch_LGBM, lunch_KN),
                                meta_regressor=lasso,
                                random_state=RANDOM_SEED)

    dinner_stack = StackingCVRegressor(regressors=(dinner_model, dinner_cat, dinner_LGBM, dinner_KN),
                                meta_regressor=lasso,
                                random_state=RANDOM_SEED)                            

    # 요일별로 분석하기        
    print(day, "요일 데이터만 추출==================================")

    train_day = train[train['요일'] == day]
    print("train_day", train_day.shape)

    # 예측 데이터
    test_day = test[test['요일'] == day]
    print("test_day", test_day.shape)
    predict_day = submission[submission['요일'] == day]
    print("predict_day", predict_day.shape)

    # 중식
    x = train_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y = train_day['중식계']
    print(x.shape, y.shape)

    lunch_stack.fit(x, y)
    # print(lunch_stack.best_score_, lunch_stack.best_params_)
    # lunch_stack = lunch_stack.best_estimator_

    test_x = test_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y_pred = lunch_stack.predict(test_x)
    predict_day['중식계'] = y_pred
    print(predict_day)

    # 중식
    x = train_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y = train_day['석식계']
    print(x.shape, y.shape)

    dinner_stack.fit(x, y)
    # print(dinner_stack.best_score_, dinner_stack.best_params_)
    # dinner_stack = dinner_stack.best_estimator_

    test_x = test_day[['월', '일', '요일', '식사가능자수', '본사출장자수', '본사시간외근무명령서승인건수']]
    y_pred = dinner_stack.predict(test_x)
    predict_day['석식계'] = y_pred
    print(predict_day)

    temp_result = pd.concat([temp_result, predict_day])

    # print(temp_result)

final_submission = temp_result.drop('요일', axis=1)
final_submission = final_submission.sort_values(by='일자', ascending=True)
final_submission = final_submission[:-4]
print(final_submission)


print("=========================================")

final_submission.to_csv('../data/DACON/cafeteria_prediction/sub/0723_submit1.csv', index=False)
print(final_submission.shape)   # (50, 3)
print(final_submission.head())



print(" 💯💯💯💯 Done!!! ") 

# 0723_submit1.csv
# 80.7592468887	
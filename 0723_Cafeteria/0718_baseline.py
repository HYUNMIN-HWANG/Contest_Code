import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#1. DATA
train = pd.read_csv('../data/DACON/cafeteria_prediction/train.csv')
test = pd.read_csv('../data/DACON/cafeteria_prediction/test.csv')
submission = pd.read_csv('../data/DACON/cafeteria_prediction/sample_submission.csv')

print(train.head())
#            일자 요일  본사정원수  본사휴가자수  ...                                               중식메뉴                                               석식메뉴     중식계    석식
# 계
# 0  2016-02-01  월   2601      50  ...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 오징어찌개  쇠불고기 (쇠고기:호주산) 계란찜 ...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 육개장  자반고등어구이  두
# 부조림  건파래무침 ...  1039.0  331.0
# 1  2016-02-02  화   2601      50  ...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 김치찌개  가자미튀김  모둠소세지구이  마늘쫑무...  콩나물밥*양념장 (쌀,현미흑미:국내산) 어묵국  유산슬 (쇠고
# 기:호주산) 아삭고추무...   867.0  560.0
# 2  2016-02-03  수   2601      56  ...  카레덮밥 (쌀,현미흑미:국내산) 팽이장국  치킨핑거 (닭고기:국내산) 쫄면야채무침 ...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 청국장찌개  황태양념구이 (황태:러시아산) 고기...  1017.0  573.0
# 3  2016-02-04  목   2601     104  ...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 쇠고기무국  주꾸미볶음  부추전  시금치나물  ...  미니김밥*겨자장 (쌀,현미흑미:국내산) 우동  멕시칸샐러드  군 
# 고구마  무피클  포...   978.0  525.0
# 4  2016-02-05  금   2601     278  ...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 떡국  돈육씨앗강정 (돼지고기:국내산) 우엉잡채...  쌀밥/잡곡밥 (쌀,현미흑미:국내산) 차돌박이찌개 (쇠고기:호주 
# 산) 닭갈비 (닭고기:...   925.0  330.0

# [5 rows x 12 columns]

print(test.head())
#            일자 요일  본사정원수  ...                                               조식메뉴                                               중식메뉴
#          석식메뉴
# 0  2021-01-27  수   2983  ...  모닝롤/연유버터베이글 우유/주스 계란후라이/찐계란 단호박죽/흑미밥 우거지국 고기완자...  쌀밥/흑미밥/찰현미밥 대구지리 매운돈갈비찜 오꼬노미계란말이 상
# 추무침 포기김치 양상추...  흑미밥 얼큰순두부찌개 쇠고기우엉볶음 버섯햄볶음 (New)아삭이고추무절임 포기김치
# 1  2021-01-28  목   2983  ...  모닝롤/대만샌드위치 우유/주스 계란후라이/찐계란 누룽지탕/흑미밥 황태국 시래기지짐 ...  쌀밥/보리밥/찰현미밥 우렁된장찌개 오리주물럭 청양부추전 수제삼 
# 색무쌈 겉절이김치 양상...            충무김밥 우동국물 오징어무침 꽃맛살샐러드 얼갈이쌈장무침 석박지
# 2  2021-01-29  금   2983  ...  모닝롤/핫케익 우유/주스 계란후라이/찐계란 오곡죽/흑미밥 매생이굴국 고구마순볶음 양...  쌀밥/흑미밥/찰현미밥 팽이장국 수제돈까스*소스 가자미조림 동초나
# 물무침 포기김치 양상...            흑미밥 물만둣국 카레찜닭 숯불양념꼬지어묵 꼬시래기무침 포기김치
# 3  2021-02-01  월   2924  ...  모닝롤/촉촉한치즈케익 우유/주스 계란후라이/찐계란 누룽지탕/흑미밥 두부김칫국 새우완...  쌀밥/흑미밥/찰현미밥 배추들깨국 오리대패불고기 시금치프리타타 
# 부추고추장무침 포기김치...           흑미밥 동태탕 돈육꽈리고추장조림 당면채소무침 모자반무침 포기김치
# 4  2021-02-02  화   2924  ...  모닝롤/토마토샌드 우유/주스 계란후라이/찐계란 채소죽/흑미밥 호박맑은국 오이생채 양...  쌀밥/팥밥/찰현미밥 부대찌개 닭살데리야끼조림 버섯탕수 세발나물 
# 무침 알타리김치/사과푸...       흑미밥 바지락살국 쇠고기청경채볶음 두부구이*볶은김치 머위된장무침 백김치

# [5 rows x 10 columns]

print(submission.head())
#            일자  중식계  석식계
# 0  2021-01-27    0    0
# 1  2021-01-28    0    0
# 2  2021-01-29    0    0
# 3  2021-02-01    0    0
# 4  2021-02-02    0    0

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

x_train = train[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]

#2. Model
model1 = RandomForestRegressor(n_jobs=-1, random_state=42)
model2 = RandomForestRegressor(n_jobs=-1, random_state=42)

#3. Train
model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)

#4. Evaluate
pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)

submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('../data/DACON/cafeteria_prediction/sub/0718_baseline.csv', index=False)

# 0718_baseline.csv
# 121.83433


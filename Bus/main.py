import os
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import gc
import urllib.request, re
from imblearn.under_sampling import * # Scikit-Learn이 0.23 버젼 이상이어야 정상적으로 활용 가능
from imblearn.over_sampling import *
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import datetime
import lightgbm as lgb
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

## 데이터 불러오기
df = pd.read_csv("./data/FINAL_DATA.csv")
#print(df.head())
#print(df.isnull().sum()) # 결측 값 확인
#####################################################################################################
# 재차인원에 따라 버스 혼잡도를 구분
# 여유
# 재차인원이 범위에 속하면 CONGESTION 라는 컬럼을 만들어서 지정값을 넣어라
df.loc[df['STAY_CNT'] <= 0, 'CONGESTION'] = '0'
# 보통
df.loc[(df['STAY_CNT'] > 0) & (df['STAY_CNT'] <= 150), 'CONGESTION'] = '1'
# 혼잡
df.loc[(df['STAY_CNT'] > 150) & (df['STAY_CNT'] <= 240), 'CONGESTION'] = '2'
# 매우혼잡
df.loc[df['STAY_CNT'] > 240, 'CONGESTION'] = '3'
# ('여유'있는 경우가 데이터의 수가 많고 매우혼잡으로 갈 수록 데이터의 수가 적어집니다.
# 이는 재현율에 문제를 발생하여 추후 undersampling, oversampling으로 극복하였습니다.)
print(len(df[df['CONGESTION'] == '0']))
print(len(df[df['CONGESTION'] == '1']))
print(len(df[df['CONGESTION'] == '2']))
print(len(df[df['CONGESTION'] == '3']))
#dtypes = df.dtypes
#print(str(dtypes['YMD_ID']))
#print(df.head())
#print(df.dtypes)
#####################################################################################################
#인코딩
dtypes = df.dtypes # 데이터 타입 집합
encoders = {} # 딕셔너리 선언 {'키' : 값}
for column in df.columns: # 컬럼별 데이터 타입 반복
    if str(dtypes[column]) == 'object': # object타입 일때
        encoder = LabelEncoder() # 카테고리(범주형)형 데이터를 수치형 데이터로 변환
        encoder.fit(df[column]) # 피팅 -> 라벨숫자로 변환
        encoders[column] = encoder # {'키 = colume' : 값 = encoder}
df_num = df.copy()
for column in encoders.keys():
    encoder = encoders[column]
    df_num[column] = encoder.transform(df[column])
del df #dataframe 삭제
gc.collect() # 가비지컬렉션 메모리 반환(함수 종료시 임계값은 0)
# 머신러닝 시물레이션에서 실행마다 데이터의 분배가 랜덤으로 발생되는 변동성을 제외하기 위함
SEED=50
def seed_everything(seed=SEED):
    random.seed(seed) # 동일한 값을 주면 즉 동일한 순서대로 난수를 발생시킴
    os.environ['PYTHONHASHSEED'] = str(seed) # 난수값을 환경변수 'PYTHONHASHSEED'에 저장
    np.random.seed(seed) #50시드를 가진 난수 발생
seed_everything(SEED)
def rmse(y_true, y_pred):
    return np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
# mean_squared_error(y_true, y_pred)
# mean_squared_error = MSE(평균 제곱 오차 회귀 손실)
# y_true : 실측(정확한) 목표 값, y_pred : 예상 목표 값
# pow : 제곱함수 sqrt : 제곱근함수
# round : 반올림 함수
#####################################################################################################
# 학습된 모델로 일주일의 혼잡도를 예측하여 정확도를 확인하기 위해 데이터 분할
df_num_train = df_num[(df_num['YYYY'] != 2022) | (df_num['MM'] != 5 ) | (df_num['DD'].isin([25, 26, 27, 28, 29, 30]) )]
df_num_pred = df_num[(df_num['YYYY'] == 2022) & (df_num['MM'] == 5 ) & (df_num['DD'].isin([25, 26, 27, 28, 29, 30]) )]
print(len(df_num_train))
print(len(df_num_pred))
del df_num
gc.collect()
#####################################################################################################
# Undersampling과 Oversampling
# 데이터 클래스 비율이 너무 차이가 나면(highly-imbalanced data) 단순히 우세한 클래스를 택하는 모형의 정확도가 높아지므로 모형의 성능판별이 어려움
# 즉, 정확도(accuracy)가 높아도 데이터 갯수가 적은 클래스의 재현율(recall-rate)이 급격히 작아지는 현상이 발생
# Undersampling과 Oversampling을 혼합하여 imbalanced data problem 극복
df_num_under = df_num_train[df_num_train['CONGESTION'] <= 1]
df_num_over = df_num_train[df_num_train['CONGESTION'] >= 1]
del df_num_train
gc.collect()
features_columns = ['YYYY', 'MM', 'DD', 'HH_ID', 'DAY_CLS', 'WKDAY_HLDAY_CLS', 'BUS_ROUTE_ID', 'GETON_CNT', 'GETOFF_CNT', 'STAY_CNT', 'TEMP', 'RAIN', 'WIND', 'SNOW', 'COVID-19_GRD', 'STN_NM', 'CONGESTION']
X,y = df_num_under[features_columns], df_num_under['CONGESTION']
X_samp_under, y_samp_under = RandomUnderSampler(random_state=0).fit_resample(X, y)
# Under Sampling - RandomUnderSampler
# random_state : 수행마다 동일한 결과를 위한 파라미터
# fit_sample(X, y) 데이터세트를 다시 샘플링합니다.
# fit_sample(X, y) - > fit_resample 으로 대체함
X,y = df_num_over[features_columns], df_num_over['CONGESTION']
X_samp_over, y_samp_over = RandomOverSampler(random_state=0).fit_resample(X, y)
# Over Sampling - RandomOverSampler
#print(X_samp_over.head(50))
print(len(X_samp_under['CONGESTION'][X_samp_under['CONGESTION'] == 0]))
print(len(X_samp_under['CONGESTION'][X_samp_under['CONGESTION'] == 1]))
print(len(X_samp_over['CONGESTION'][X_samp_over['CONGESTION'] == 2]))
print(len(X_samp_over['CONGESTION'][X_samp_over['CONGESTION'] == 3]))
del df_num_under
del df_num_over
del X
del y
gc.collect()
# Undersampling과 Oversampling을 혼합하여 imbalanced data problem 극복
df_num_fin = pd.concat([X_samp_under[X_samp_under['CONGESTION'] < 1], X_samp_over])
# concat 데이터 프레임 결합
df_num_fin.reset_index(drop=True, inplace=True)
# df_num_fin.reset_index : 데이터 프레임의 인덱스 혹은 인덱스 레벨을 리셋함
# drop  : 리셋하려는 인덱스를 DataFrame의 칼럼으로 추가하는지 여부를 결정
# inplace : DataFrame을 제자리에서 수정할지 여부를 결정
del X_samp_under
del X_samp_over
del y_samp_under
del y_samp_over
gc.collect()
#####################################################################################################
# LGBM in K-fold
print(datetime.datetime.now())
# 교차검증을 위해 KFold 사용
features_columns = ['YYYY', 'MM', 'DD', 'HH_ID', 'DAY_CLS', 'WKDAY_HLDAY_CLS', 'BUS_ROUTE_ID', 'GETON_CNT', 'GETOFF_CNT', 'TEMP', 'RAIN', 'WIND', 'SNOW', 'COVID-19_GRD', 'STN_NM']
X, y = df_num_fin[features_columns], df_num_fin['STAY_CNT']
folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
fi_df = pd.DataFrame()
pred = np.zeros(len(df_num_pred))
print(datetime.datetime.now())
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(df_num_fin)):
    tr_data = lgb.Dataset(X.loc[trn_idx], label=y[trn_idx])
    vl_data = lgb.Dataset(X.loc[val_idx], label=y[val_idx])
    lgb_params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 700,
        'max_depth': -1,
        'min_child_weight': 5,
        'colsample_bytree': 0.9,
        'subsample': 0.7,
        'n_estimators': 20000,   # 오류 발생
        'gamma': 0,
        'verbose': -1,
        'seed': SEED,
        'early_stopping_rounds': 50
    }
    estimator = lgb.train(lgb_params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=500)
    fi_df = pd.concat([fi_df, pd.DataFrame(sorted(zip(estimator.feature_importance(), features_columns)),
                                           columns=['Value', 'Feature'])])
    print(datetime.datetime.now())
    pred += estimator.predict(df_num_pred[features_columns]) / 5
    del estimator
    gc.collect()
print(datetime.datetime.now())
#############################################################################################################
# 예측치에 대한 혼잡도 생성
# 여유
df_num_pred.loc[pred <= 0, 'CONGESTION_P'] = 0
# 보통
df_num_pred.loc[(pred > 0) & (pred <= 150), 'CONGESTION_P'] = 1
# 혼잡
df_num_pred.loc[(pred > 150) & (pred <= 240), 'CONGESTION_P'] = 2
# 매우혼잡
df_num_pred.loc[pred > 240, 'CONGESTION_P'] = 3

# Confusion Matrix를 생성한다.
y = df_num_pred['CONGESTION']
p = df_num_pred['CONGESTION_P']
conf_mat = confusion_matrix(y, p)
print(conf_mat)
fig, ax = plt.subplots(figsize = (10,6))
sns.heatmap(conf_mat, annot=True, ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['여유', '보통', '혼잡', '매우혼잡'])
ax.yaxis.set_ticklabels(['여유', '보통', '혼잡', '매우혼잡'])

# 정확도, 재현율
print('accuracy', metrics.accuracy_score(y, p) )
print('recall', metrics.recall_score(y, p,average='macro'))

#############################################################################################################
# Feature Importance (기능 중요도)
fi_df_sum = fi_df.groupby('Feature').sum().reset_index(drop=False).sort_values(by='Value',ascending=False)
print(fi_df_sum)
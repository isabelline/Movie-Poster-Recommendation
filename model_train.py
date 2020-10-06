import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import operator
import xgboost as xgb
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.ensemble import RandomForestRegressor



def get_y_from_demo_file(demo_file, end):
    """
    demo_file을 기반으로 타겟 Y 가 될 성별, 연령별 예매율 데이터 리스트로 반환
	:param demo_file: 2018 영화의 예매율 정보와 예고편 다운로드 정보가 있는 tsv 파일
	:param end: 2018년 영화 중 예고편을 다운로드한 영화의 개수. 캡쳐 폴더 이름이 1부터 영화 개수의 숫자로 되어 있음
    :return: 각 이미지에 대응되는 성별 예매율과 연령별 예매율 리스트
    """
    df = pd.read_csv(demo_file,sep="\t",encoding='utf-8')
    age_y = []
    gender_y = []
    for i in range(1, end):
        for j, row in df.iterrows():
            if row[0] == " ":
                continue
            elif i == int(row[0]):
                age = [float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11])]
                gender = [float(row[5]),float(row[6])]
                age_y.append(age)
                gender_y.append(gender)
            else:
                print("err. wrong path")
    return gender_y, age_y


def shuffle_array(list_one, list_two, list_three):
    """
    길이가 같은 3개의 리스트의 쌍을 유지하면서 shuffling
    :param list_one: 리스트1
    :param list_two: 리스트2
    :param list_three: 리스트3
    :return: 쌍은 유지하되 순서가 바뀐 3개의 리스트
    """
    index = list(range(len(list_one)))
    from random import shuffle
    shuffle(index)
    temp_one = []
    temp_two = []
    temp_three = []
    for i in index:
        temp_one.append(list_one[i])
        temp_two.append(list_two[i])
        temp_three.append(list_three[i])

    return temp_one, temp_two, temp_three







def make_kfold_dataset(total_feature, total_gender, total_age):
    """
    데이터를 10겹 교차 검증을 위해 10개의 데이터 단위로 분할 및
    :param total_feature: vision api로 만든 피쳐 데이터
    :param total_gender: 성별 예매율 데이터
    :param total_age: 연령별 예매율 데이터
    """
    kf = KFold(n_splits=10)
    cnt = 1

    for train_index, val_index in kf.split(total_feature):
        X_train, Y_train_gender, Y_train_age  = total_feature[train_index], total_gender[train_index], total_age[train_index]
        X_val, Y_val_gender, Y_val_age  = total_feature[val_index], total_gender[val_index], total_age[val_index]

        with open("drive/My Drive/dataset_cv/train_x_"+str(cnt)+".npy", 'wb') as f:
            np.save(f, X_train)

        with open("drive/My Drive/dataset_cv/train_y_age_"+str(cnt)+".npy", 'wb') as f:
            np.save(f, Y_train_age)

        with open("drive/My Drive/dataset_cv/train_y_gender_"+str(cnt)+".npy", 'wb') as f:
            np.save(f, Y_train_gender)

        with open("drive/My Drive/dataset_cv/val_x_"+str(cnt)+".npy", 'wb') as f:
            np.save(f, X_train)

        with open("drive/My Drive/dataset_cv/val_y_age_"+str(cnt)+".npy", 'wb') as f:
            np.save(f, Y_train_age)

        with open("drive/My Drive/dataset_cv/val_y_gender_"+str(cnt)+".npy", 'wb') as f:
            np.save(f, Y_train_gender)

        cnt += 1




def get_cv(cv_index):
    """
    1-10 CV번호에 대응하는 train과 val X, Y 값 반환
    :param cv_index: 1-10 사이의 CV 번호
    :return: 해당 cv의 훈련 피쳐 데이터, 성별 예매율 데이터, 연령별 데이터와 검증 피쳐 데이터, 성별 예매율 데이터, 연령별 데이터
    """
    train_x = np.load("drive/My Drive/dataset_cv/train_x_"+str(cv_index)+".npy")
    train_y_age = np.load("drive/My Drive/dataset_cv/train_y_age_"+str(cv_index)+".npy")
    train_y_gender = np.load("drive/My Drive/dataset_cv/train_y_gender_"+str(cv_index)+".npy")

    val_x = np.load("drive/My Drive/dataset_cv/val_x_"+str(cv_index)+".npy")
    val_y_age = np.load("drive/My Drive/dataset_cv/val_y_age_"+str(cv_index)+".npy")
    val_y_gender = np.load("drive/My Drive/dataset_cv/val_y_gender_"+str(cv_index)+".npy")

    return train_x, train_y_age, train_y_gender, val_x, val_y_age






def plot_label_count(feature_2019, valid_feature_index):
    """
    피쳐 행렬에서 특정 조건으로 필터를 적용하여 라벨이 등장하는 횟수 카운트
    :param feature_2019: 피쳐 파일
    :param valid_feature_index: 특정 조건의 인덱스, 예를 들면 남자 예매율이 높은 영화의 인덱스
    """
    df = pd.read_csv("drive/My Drive/class-descriptions.csv", encoding='utf-8', header=None)
    label_ids = list(df[0])
    label_names = list(df[1])
    print("Total label number is {}".format(str(len(label_ids))))

    dict = dict()
    feature = feature_2019.T
    indices = valid_feature_index

    for i, label in enumerate(label_names):
    label_score = np.sum(feature[i][valid_feature_index])
    dict[label] = label_score


    sorted_x = sorted(dict.items(), key=operator.itemgetter(1),reverse=True)

    labels = []
    scores = []

    # 가장 자주 나온 라벨 상위 60개
    for label, score in sorted_x[:60]:
        labels.append(label)
        scores.append(score)

    # 막대 그래프 그리기
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_axes([0,0,1,1])
    langs = labels
    students = scores
    plt.xticks(rotation='vertical')
    ax.bar(langs,students)
    plt.show()


def fully_connected_regg_cv():
    """
    Fully Connected model CV에 대하여 훈련과 검증
    """
    mse_list=[]
    mae_list = []
    corr_list = []

    for epochs in range(2,4):
        for cv_index in range(1,10):
            print("CV: " + str(cv_index))
            train_x, train_y_age, train_y_gender, val_x, val_y_age, val_y_gender = get_cv(cv_index)
            train_y_age = [x[1] for x in train_y_age]
            train_y_gender = [x[0] for x in train_y_gender]
            val_y_age = [x[1] for x in val_y_age]
            val_y_gender = [x[0] for x in val_y_gender]


            model = Sequential()
            model.add(Dense(200))
            model.add(Dense(64))
            model.add(Dense(1))
            model.add(Activation('softmax'))

            model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

            hist = model.fit(train_x, train_y_gender, epochs=epochs, validation_data=(val_x, val_y_gender), verbose=2)

            pred = model.predict(val_x)

            mse = mean_squared_error(val_y_gender, pred)
            mae = mean_absolute_error(val_y_gender, pred)
            print("mae: {} mse: {} ".format(str(mae), str(mse)))
            mse_list.append(mse)
            mae_list.append(mae)

        mean_mse = np.mean(mse_list)
        mean_mae = np.mean(mae_list)

        print("Epoch: " + str(epochs))
        print("avg mse: {} avg mae: {}".format(str(mean_mse),str(mean_mae)))



################



def random_forest_regg():
    """
    랜덤 포레스트 모델 CV에 대하여 훈련과 검증
    """
    train_features = np.load("drive/My Drive/dataset/total_x.npy")
    train_labels = np.load("drive/My Drive/dataset/total_y_gender.npy")

    mse_list = []
    mae_list = []

    for cv_index in range(1, 11):
        print("CV: " + str(cv_index))
        train_x, train_y_age, train_y_gender, val_x, val_y_age, val_y_gender = get_cv(cv_index)

        train_y_age = np.array([x[1] for x in train_y_age])
        train_y_gender = np.array([x[0] for x in train_y_gender])
        val_y_age = np.array([x[1] for x in val_y_age])
        val_y_gender = np.array([x[0] for x in val_y_gender])

        rf = RandomForestRegressor(n_estimators=10, random_state=0)

        regressor = rf
        regressor.fit(train_x, train_y_gender)
        best_grid = regressor
        pred = best_grid.predict(val_x)
        mse = mean_squared_error(val_y_gender, pred)
        mae = mean_absolute_error(val_y_gender, pred)
        print(" {} {}".format(str(mse), str(mae)))

        mse_list.append(mse)
        mae_list.append(mae)

    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)

    print("{} {} ".format(str(mean_mse), str(mean_mae)))


def random_forest_regg_grid_search():
    """
    랜덤 포레스트 모델 그리드 서치 적용하여 훈련
    """
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    rf = RandomForestRegressor()

    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                              cv = 10, n_jobs = -1, verbose = 2)

    train_features = np.load("drive/My Drive/dataset/total_x.npy")
    train_labels = np.load("drive/My Drive/dataset/total_y_gender.npy")

    grid_search.fit(train_features, train_labels)

    best_grid = grid_search.best_estimator_

    mse_list = []
    mae_list = []

    for cv_index in range(1,10):
        print("CV: " + str(cv_index))
        train_x, train_y_age, train_y_gender, val_x, val_y_age, val_y_gender = get_cv(cv_index)

        train_y_age = [x[1] for x in train_y_age]
        train_y_gender = [x[0] for x in train_y_gender]
        val_y_age = [x[1] for x in val_y_age]
        val_y_gender = [x[0] for x in val_y_gender]

        pred = best_grid.predict(val_x)
        mse = mean_squared_error(val_y_gender, pred)
        mae = mean_absolute_error(val_y_gender, pred)
        mse_list.append(mse)
        mae_list.append(mae)

    mean_mse = np.average(mse_list)
    mean_mae = np.average(mae_list)

    print("avg mse: {} avg mae: {} ".format(str(mean_mse),str(mean_mae)))


#####################



def xgb_regg():
    """
    CV에 대하여 xgb 모델 훈련
    """
    X = np.load("drive/My Drive/dataset/total_x.npy")
    Y = np.load("drive/My Drive/dataset/total_y_gender.npy")


    estimator= xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 10)

    mse_list = []
    mae_list = []

    for cv_index in range(1,11):
        print("CV: " + str(cv_index))
        train_x, train_y_age, train_y_gender, val_x, val_y_age, val_y_gender = get_cv(cv_index)
        train_y_age = np.array([x[1] for x in train_y_age])
        train_y_gender = np.array([x[0] for x in train_y_gender])
        val_y_age = np.array([x[1] for x in val_y_age])
        val_y_gender = np.array([x[0] for x in val_y_gender])

        estimator.fit(train_x,train_y_gender)
        pred = estimator.predict(val_x)
        mse = mean_squared_error(val_y_gender, pred)
        mae = mean_absolute_error(val_y_gender, pred)
        print(" {} {} ".format(str(mse), str(mae)))
        mse_list.append(mse)
        mae_list.append(mae)

    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)

    print("{} {} {}".format(str(mean_mse),str(mean_mae)))



def xgb_regg_grid_search():
    """
    grid search 를 적용한 xgboost 모델 훈련
    """
    X = np.load("drive/My Drive/dataset/total_x.npy")
    Y = np.load("drive/My Drive/dataset/total_y_gender.npy")

    parameters = {
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
    }


    estimator= xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 10)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring = 'roc_auc',
        n_jobs = 10,
        cv = 10,
        verbose=True
    )

    grid_search.fit(X, Y)
    best_grid = grid_search.best_estimator_

    corr_list = []
    mse_list = []
    mae_list = []

    for cv_index in range(1,10):
        print("CV: " + str(cv_index))
        train_x, train_y_age, train_y_gender, val_x, val_y_age, val_y_gender = get_cv(cv_index)
        train_y_age = [x[1] for x in train_y_age]
        train_y_gender = [x[0] for x in train_y_gender]
        val_y_age = [x[1] for x in val_y_age]
        val_y_gender = [x[0] for x in val_y_gender]

        pred = best_grid.predict(val_x)
        mse = mean_squared_error(val_y_gender, pred)
        mae = mean_absolute_error(val_y_gender, pred)
        mse_list.append(mse)
        mae_list.append(mae)

    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)
    mean_corr = np.mean(corr_list)

    print("{} {} {}".format(str(mean_mse),str(mean_mae)))


# 2018년 영화의 캡쳐 이미지에 해당하는 성별, 연령별 예매율 리스트 반환
gender_y_two, age_y_two = get_y_from_demo_file("drive/My Drive/2018_file_demo.tsv",54)
gender_y_two = np.array(gender_y_two)
age_y_two = np.array(age_y_two)


# 저장한 최종 2019년 영화 캡쳐 이미지의 vision api 기반 피쳐 읽기
feature_2019 = np.load("drive/My Drive/feature_2019.npy")

# 저장한 최종 2018년 영화 캡쳐 이미지의 vision api 기반 피쳐 읽기
feature_2018 = np.load("drive/My Drive/feature_2018.npy")

# 2019년 이미지 파일에 해당하는 예매율 데이터 반환
gender_y_one, age_y_one = get_y_from_demo_file("2019_demo_file.tsv",61)

# 데이터의 크기가 맞는 지 확인
assert feature_2019.shape[0] == len(gender_y_one)

gender_y_one = np.array(gender_y_one)
age_y_one = np.array(age_y_one)

# 2018년과 2019년 피쳐와 예매율 데이터 합치기
total_feature = np.concatenate([feature_2019,feature_2018],axis=0)
total_age = np.concatenate([age_y_one, age_y_two],axis=0)
total_gender = np.concatenate([gender_y_one, gender_y_two],axis=0)
total_feature, total_age, total_gender = shuffle_array(total_feature, total_age, total_gender)

# 데이터셋 저장
with open("drive/My Drive/dataset/total_x.npy", 'wb') as f:
    np.save(f, total_feature)

with open("drive/My Drive/dataset/total_y_age.npy", 'wb') as f:
    np.save(f, total_age)

with open("drive/My Drive/dataset/total_y_gender.npy", 'wb') as f:
    np.save(f, total_gender)

# FC 훈련
fully_connected_regg_cv()

# xgb 훈련
xgb_regg()


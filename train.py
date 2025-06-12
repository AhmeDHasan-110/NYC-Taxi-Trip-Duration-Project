import pandas as pd 
import numpy as np

from sklearn.linear_model import Ridge, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, root_mean_squared_error as rmse
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, KFold 
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from prepare_data import prepare

KFOLDS = 5
RANDOMSEED = 12

def train_model(x, y, model, name, grid) :
    numeric_pipe = Pipeline(steps=[
        ('polynomial' , PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler' , MinMaxScaler())
    ])

    transform = ColumnTransformer(transformers=[
        ('numerical_features', numeric_pipe, numerical),
        ('encoder' , OneHotEncoder(handle_unknown= 'ignore') , categorical),
    ], remainder= 'passthrough')

    pipe = Pipeline(steps=[
        ('transformer' , transform),
        ('selection' , SelectFromModel(LassoCV(cv=KFOLDS, alphas=[0.01, 0.1, 0.5, 1], max_iter=3000, random_state=RANDOMSEED))),
        (name , model)
    ])

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOMSEED)
    model = GridSearchCV(pipe, param_grid=grid, cv=kf, scoring='neg_root_mean_squared_error')
    model.fit(x, y)

    best_pipe = model.best_estimator_

    # evaluate the training error
    # rmse_error , r2 = evaluate(best_pipe, x, y)
    # print(f'best {name} model :- rmsle = {rmse_error:0.4}\tr2 score = {r2:0.4}\n')

    return best_pipe

def evaluate(model, x, y) :
    pred = model.predict(x)
    rmse_error = round(rmse(y, pred), 3)
    r2 = round(r2_score(y, pred), 3)

    return rmse_error, r2

def prep_data(data) :
    data['trip_duration'] = data['trip_duration'].apply(lambda x : np.log1p(x))
    x = data.drop('trip_duration', axis=1)
    return x, data['trip_duration']

if __name__ == '__main__' :
    train = pd.read_parquet('trip_duration_proj/prepared_data/train.parquet')
    test = pd.read_parquet('trip_duration_proj/data/nyc_taxi_trip_duration/test.parquet')
    test = prepare(test)    # prepare the test data

    x_train, y_train = prep_data(train)
    x_val, y_val = prep_data(test)

    categorical = ['vendor_id', 'passenger_count', 'weekday', 'hour', 'day', 'month']
    numerical = ['average temperature', 'precipitation', 'snow depth', 'manhattan_distance', 'haversine_distance'
                ,'pickup_longitude', 'pickup_latitude' ,'dropoff_longitude' ,'dropoff_latitude']

    models_info = {
        ('Ridge', Ridge()): {'Ridge__alpha' : [0.01, 0.1, 1]},
        ('MLPRegressor', MLPRegressor()): {
            'MLPRegressor__hidden_layer_sizes' : [(10, 10), (20, 10)],
            'MLPRegressor__activation' : ['relu', 'tanh'],
            'MLPRegressor__alpha' : [0.01, 0.1]
            }
        }

    # train the models
    best_models = {}
    for (name, model), grid in models_info.items() :
        best_pipe = train_model(x_train, y_train, model, name, grid)
        best_models[name] = best_pipe

    # evaluate on test data
    results = {}
    for name, model in best_models.items() :
        train_error = evaluate(model, x_train, y_train) 
        val_error = evaluate(model, x_val, y_val) 
        results[name] = [*train_error, *val_error]
    
    # save results in excel file
    index = ['train_RMSE', 'train_R2', 'val_RMSE', 'val_R2']
    df = pd.DataFrame(results, index=index)
    df.to_excel('trip_duration_proj/models_results.xlsx')





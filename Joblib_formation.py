import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

model_file = 'model.pkl'
pipeline_file = 'pipeline.pkl'

def build_pipeline(num_attribute,cat_attribute):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scalar',StandardScaler())   
    ])
    cat_pipeline = Pipeline([
        ('onehot' , OneHotEncoder(handle_unknown='ignore'))
    ])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline,num_attribute),
        ('cat', cat_pipeline,cat_attribute)
    ])
    return full_pipeline

if not(os.path.exists(model_file)):
    housing=pd.read_csv('housing.csv')
    housing['category'] = pd.cut(housing['median_income'],
                                 bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
                                 labels=['a', 'b', 'c', 'd', 'e'])
    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    for train, test in splits.split(housing, housing['category']):
        strat_train_set = housing.loc[train].drop(columns='category')
            # will work on this data only
        strat_test_set = housing.loc[test].drop(columns='category')
            # keep this data aside for now
    strat_test_set.to_csv('testing_house.csv',index=False)
    housing = strat_train_set
    housing_labels = housing['median_house_value'].copy()
    housing_feature = housing.drop(columns=['median_house_value'])

    num_attri = housing_feature.drop(columns='ocean_proximity').columns.tolist()
    cat_attri = ['ocean_proximity']

    pipeline = build_pipeline(num_attri,cat_attri)
    housing_prepared = pipeline.fit_transform(housing_feature)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,model_file)
    joblib.dump(pipeline,pipeline_file)
    print('model-trained')

else:
    model = joblib.load(model_file)
    pipeline = joblib.load(pipeline_file)

    input_data = pd.read_csv("testing_house.csv")
    transformed_input = pipeline.fit_transform(input_data)
    prediction = model.predict(transformed_input)
    input_data['median_house_value_predicted'] = prediction
    input_data.to_csv("output.csv",index=False)
    print('Inference is completed...!!!')    


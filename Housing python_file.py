import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1). Load the dataset
housing = pd.read_csv('housing.csv')
print(housing)

#2). Testing-Traninig split(Stratified Shuffle Split)
housing['category'] = pd.cut(housing['median_income'],bins=[0,1.5,3.0,4.5,6,np.inf],
labels=['a','b','c','d','e'])

splits = StratifiedShuffleSplit(n_splits=1 , test_size=0.20 , random_state=42)
for train,test in splits.split(housing,housing['category']):
    strat_train_set = housing.loc[train].drop(columns='category')
    # will work on this data only
    strat_test_set = housing.loc[test].drop(columns='category')
    # keep this data aside for now

# 3). Separate out label and features
housing_labels = housing['median_house_value'].copy()
housing = housing.drop(columns = ['median_house_value','category'])

# 4). List the numerical attributes and categorical attributes
num_attributes = housing.drop(columns = 'ocean_proximity').columns.tolist()
cat_attributes = ['ocean_proximity']

# 5). Pipeline start
# for numerical column
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = 'median')),
    ('scalar', StandardScaler())]
)

# catgorical pipeline
cat_pipeline = Pipeline([
    ('onehot',OneHotEncoder(handle_unknown = 'ignore'))
])

# construct full pipeline
full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_attributes),
    ('cat',cat_pipeline,cat_attributes)
])

#6). transform the data
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# training the model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred = lin_reg.predict(housing_prepared)

for i in zip(lin_pred,housing_labels):
    print(i)


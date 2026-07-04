# California-Housing-Price-Prediction

## Project Overview  
This project builds an end-to-end machine learning pipeline to predict median house prices in California. It covers data loading, stratified train/test splitting, feature engineering through a preprocessing pipeline, model training with a Random Forest Regressor, and persistent model serialisation for reuse.  

## Dataset Description  
  -  **longitude / latitude**: Geographic coordinates of the block group  
  -  **housing_median_age**: Median age of houses in the block group  
  -  **total_rooms**: Total number of rooms across all households  
  -  **population**: Block-group population  
  -  **households**: Number of households  
  -  **median_income**: Median household income (tens of thousands USD)  
  -  **ocean_proximity**: Distance category from the ocean  
  -  **median_house_value**: TARGET – median house value (USD)  
## Technology Used  
  -  **Python**: Programming Language  
  -  **Pandas**: Data Manipulation
  -  **Numpy**: Numerical Operations  
  -  **Scikit-Learn**: Machine Learning Models  
  -  **Joblib**: Model Saving & Loading
  -  **Pipeline**: Data Preprocessing Automation  
  -  **Operating System**: Management of files and directories
## Project Workflow  
**Step-1** – Load Dataset:  housing.csv is read with pd.read_csv(). Initial exploration (shape, dtypes, value_counts) is performed to understand distributions.  

**Step-2** – Income Category & Stratified Split: median_income is binned into 5 categories (a–e) using pd.cut(). StratifiedShuffleSplit (n_splits=1, test_size=0.20, random_state=42) ensures each income band is proportionally represented in both train and test sets.

**Step-3** – Feature/Label Separation:  median_house_value is extracted as the label array. The remaining columns become the feature matrix. The temporary category column is dropped.  

**Step-4** – Preprocessing Pipeline:  Numerical features: median imputation → StandardScaler. Categorical feature (ocean_proximity): OneHotEncoder (handle_unknown="ignore"). Both sub-pipelines are composed via ColumnTransformer.  

**Step-5** – Model Training:  Linear Regression and Random Forest Regressor models were trained on the preprocessed housing dataset to learn feature-price relationships and predict housing values.  

**Step-6** – Serialisation & Inference: The trained Machine Learning model and preprocessing pipeline were saved using Joblib (model.pkl and pipeline.pkl). This allows the model to be reused for future predictions without retraining.  

## Machine Learning Model Used  

### Linear Regression Model  
  -  Baseline regression model.  
  -  Establishes relationship between features and house prices.
### Random Regressor Model  
  -  Ensemble learning algorithm.
  -  Provides improved prediction accuracy.
  -  Handles non-linear relationships effectively.
### Decision Tree Model  
  -  Tree-based Machine Learning algorithm that splits data into decision nodes based on feature values.
  -  Captures non-linear relationships between housing features and house prices.
  -  Easy to visualize and interpret but may overfit on training data if not properly tuned.  

## Key Features  
  -  Automated preprocessing pipeline.  
  -  Missing value handling.  
  -  Feature scaling.  
  -  Categorical feature encoding.  
  -  Stratified train-test splitting.  
  -  Multiple machine learning models.  
  -  Model persistence using Joblib.  
  -  Prediction generation on unseen data.  

## Conclusion  
This project demonstrates the complete machine learning workflow, including data preprocessing, feature transformation, model training, prediction, and model deployment preparation. The Random Forest Regressor model is considered as the best suited model for the housing price prediction and can be extended into a real-world real estate analytics application.

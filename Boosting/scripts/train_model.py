import pandas as pd
import os
from project_config import ROOT_DIR
from data_preprocess import DataPreprocess
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


data = pd.read_parquet(os.path.join(ROOT_DIR, 'data', 'raw', 'cars.parquet'))

# Preprocess the data
data_preprocessor = DataPreprocess(data)
processed_data = (data_preprocessor
                  .FeatureEngineering()
                  .OHE()
                  .OrdinalEncoding()
                  .TargetModification())

processed_data.data.to_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'processed_data.csv'), index=False)

# Split the dataset
X = processed_data.data.drop(columns = {'set_type', 'price'})
y = processed_data.data['price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 20% test, 20% val

# Initialize and train a model.
# params received through Optuna
params = {'iterations': 943, 'depth': 3, 'learning_rate': 0.14762311962888797, 'l2_leaf_reg': 3.2708992320142545, 'colsample_bylevel': 0.9832366428139097}
model = CatBoostRegressor(**params, thread_count=1)
model.fit(X_train, y_train)

import joblib
joblib.dump(model, os.path.join(ROOT_DIR, 'models', 'model.pkl'))
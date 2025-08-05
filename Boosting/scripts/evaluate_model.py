import os
import joblib
import pandas as pd
import numpy as np
from project_config import ROOT_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


# processed_data = pd.read_csv('D:/it_academy/models for git/Boosting/data/processed/processed_data.csv')
processed_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'processed_data.csv'))

X = processed_data.drop(columns = {'set_type', 'price'})
y_log1p = processed_data['price']
y = np.expm1(y_log1p)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 20% test, 20% val

# Loading the trained model
# model = joblib.load('D:/it_academy/models for git/Boosting/models/model.pkl')
model = joblib.load(os.path.join(ROOT_DIR, 'models', 'model.pkl'))

y_pred_log1p = model.predict(X_test)
y_pred = np.expm1(y_pred_log1p)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Squared Error on Test Set: {mse}')
print(f'Mean Absolute Error on Test Set: {mae}')
print(f'Mean Absolute Percentage Error on Test Set: {mape}')
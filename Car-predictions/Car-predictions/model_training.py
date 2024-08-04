from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from car_data_prep import prepare_data
from sklearn.model_selection import train_test_split


# Load data
df = pd.read_csv('dataset.csv')

# Prepare data
df_prepared = prepare_data(df)


# Separate features and target variable
X = df_prepared.drop(columns=['Price'])
y = df_prepared['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Initialize ElasticNetCV model
elasticnet_cv = ElasticNet(random_state=42)

# Fit the model
elasticnet_cv.fit(X_train, y_train)

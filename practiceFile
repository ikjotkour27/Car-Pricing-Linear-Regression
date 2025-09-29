# Car Pricing Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# Step 1: Create Dataset
# ---------------------------

data = {
    'Car Age': [3, 5, 2, 7, 4, 1, 6, 3, 5, 2],
    'Mileage': [40000, 80000, 30000, 120000, 50000, 20000, 90000, 45000, 85000, 25000],
    'Engine Size': [2.0, 1.8, 2.5, 2.0, 1.6, 3.0, 2.2, 2.0, 1.8, 2.5],
    'Brand': ['Toyota', 'Honda', 'BMW', 'Toyota', 'Honda', 'BMW', 'Toyota', 'Honda', 'BMW', 'Toyota'],
    'Price': [20000, 15000, 35000, 12000, 18000, 40000, 16000, 21000, 33000, 37000]
}

df = pd.DataFrame(data)

# ---------------------------
# Step 2: One-Hot Encode Brand
# ---------------------------
encoder = OneHotEncoder(sparse=False)
brand_encoded = encoder.fit_transform(df[['Brand']])
brand_df = pd.DataFrame(brand_encoded, columns=encoder.get_feature_names_out(['Brand']))

# Combine features
X = pd.concat([df[['Car Age', 'Mileage', 'Engine Size']], brand_df], axis=1)
y = df['Price']

# ---------------------------
# Step 3: Split Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=

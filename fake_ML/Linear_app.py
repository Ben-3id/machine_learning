import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fake import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("financial_regression.csv")
data = data.drop(["GDP", "CPI", "us_rates_%"], axis=1)
data = data.dropna()

# Date features
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year.astype(int)
data["month"] = data["date"].dt.month.astype(int)
data["day"] = data["date"].dt.day.astype(int)
data = data.drop("date", axis=1)

# Features & target
X = data.drop(columns="gold close")
y = data["gold close"]   # Series

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scaling X ---
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# --- Scaling y ---
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Train model
model = LinearRegression(iter=1000,alpha=0)#itre=1e-7 alpha=1e-5
model.fit(X_train, y_train)

# Predictions (scaled)
y_pred_scaled = model.pred(X_test)

# رجع القيم للوضع الأصلي
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()


# حساب المقاييس بين y_test الأصلي و y_pred الأصلي
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test ,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

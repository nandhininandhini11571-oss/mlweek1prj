import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


df = pd.read_csv('pythonlearningdata.csv')
print("First few rows:")
print(df.head(5))


X = df[['hours_spent_learning_per_week']]
y = df['final_exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(f"\nMSE (sklearn): {mse_sklearn:.2f}")


x = df['hours_spent_learning_per_week'].values
y = df['final_exam_score'].values
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_xx = np.sum(x ** 2)

m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
c = (sum_y - m * sum_x) / n
print(f"Equation: final_exam_score = {m:.2f} * hours_spent_learning_per_week + {c:.2f}")

y_pred = m * x + c
mse = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse:.2f}")


x_new = float(input("Enter hours spent learning per week: "))
y_new = m * x_new + c
print(f"Predicted final exam score: {y_new:.2f}")
print(f"\nMSE comparison - sklearn: {mse_sklearn:.2f}, manual: {mse:.2f}")
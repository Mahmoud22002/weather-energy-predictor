# =========================================
# ⚡ Local Weather → Electricity Predictor MVP
# Dataset: weather-energy-data-update.csv
# =========================================
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# 1️⃣ Load Data
# -------------------------------
df = pd.read_csv(r"C:\Users\kaddo\PycharmProjects\Electricity relation with weather\weather-energy-data-update.csv")
print(df.head())

# -------------------------------
# 2️⃣ Basic Info
# -------------------------------
print(df.info())

# -------------------------------
# 3️⃣ Drop Missing
# -------------------------------
df_clean = df.dropna()
print(f"✅ Cleaned dataset: {df_clean.shape[0]} rows")


# -------------------------------
# 4️⃣ Correlation Heatmap (fixed)
# -------------------------------
numeric_df = df_clean.select_dtypes(include=[np.number])

plt.figure(figsize=(12,10))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -------------------------------
# 5️⃣ Features & Target
# -------------------------------
X = df_clean[['temp_dry', 'humidity', 'wind_speed', 'pressure', 'hour', 'day_of_week', 'month', 'is_weekend']]
y = df_clean['kWh']

print(f"✅ Features: {X.columns.tolist()}")

# -------------------------------
# 6️⃣ Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# -------------------------------
# 7️⃣ Train Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\n📈 Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

# -------------------------------
# 8️⃣ Predict & Evaluate
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ R² score: {r2:.3f}")
print(f"✅ MAE: {mae:.4f} kWh")

# -------------------------------
# 9️⃣ Plot Actual vs Predicted
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:200], label='Actual', marker='o')
plt.plot(y_pred[:200], label='Predicted', marker='x')
plt.title("Actual vs Predicted Electricity Usage (first 200 samples)")
plt.xlabel("Sample")
plt.ylabel("kWh")
plt.legend()
plt.show()

# -------------------------------
# 🔟 Residuals
# -------------------------------
residuals = y_test - y_pred

plt.figure(figsize=(8,4))
sns.histplot(residuals, kde=True, bins=40)
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error (kWh)")
plt.show()

print("\n🚀 MVP Complete! Now you can try Random Forest or XGBoost for better results.")

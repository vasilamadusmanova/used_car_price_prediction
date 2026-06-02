import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset yuklash
df = pd.read_csv("Used Car Dataset.csv")

# Keraksiz ustunni o'chirish
df = df.drop("Unnamed: 0", axis=1)

# Missing value larni to'ldirish
df["mileage(kmpl)"] = df["mileage(kmpl)"].fillna(df["mileage(kmpl)"].median())
df["engine(cc)"] = df["engine(cc)"].fillna(df["engine(cc)"].median())
df["max_power(bhp)"] = df["max_power(bhp)"].fillna(df["max_power(bhp)"].median())
df["torque(Nm)"] = df["torque(Nm)"].fillna(df["torque(Nm)"].median())

# Category to Number
le = LabelEncoder()

df["fuel_type"] = le.fit_transform(df["fuel_type"])
df["ownsership"] = le.fit_transform(df["ownsership"])
df["transmission"] = le.fit_transform(df["transmission"])

# X va y ni ajratish
X = df.drop(
    [
        "price(in lakhs)",
        "car_name",
        "registration_year",
        "insurance_validity",
        "manufacturing_year"
    ],
    axis=1
)

y = df["price(in lakhs)"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(X.dtypes)

# Linear Regression model
model = LinearRegression()

# Modelni o'qitish
model.fit(X_train, y_train)



print("Model trained successfully!")

# Prediction

y_pred = model.predict(X_test)

print("First 10 Predictions:")
print(y_pred[:10])

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)
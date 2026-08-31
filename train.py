import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("housing.csv")

# Remove null values
df.dropna(inplace=True)

# Log transformation
df["total_rooms"] = np.log(df["total_rooms"])
df["total_bedrooms"] = np.log(df["total_bedrooms"])
df["population"] = np.log(df["population"])
df["households"] = np.log(df["households"])

# One hot encoding
df = pd.get_dummies(df, columns=["ocean_proximity"])

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# SAVE COLUMNS
columns = X.columns.tolist()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    max_depth=10
)

# Train model
model.fit(X_train, y_train)

# Save model
with open("housing.pkl", "wb") as f:
    pickle.dump(model, f)

# Save columns
with open("columns.pkl", "wb") as f:
    pickle.dump(columns, f)

print("Everything Saved Successfully")

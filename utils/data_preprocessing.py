import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def load_and_preprocess_data():
    """Loads charging station dataset and preprocesses it for ML training."""
    df = pd.read_csv("data/charging_stations.csv")  # ✅ Correct CSV file

    # Convert categorical features to numerical
    df = pd.get_dummies(df, columns=["connector_type", "city", "nearby_facilities", "ev_type_preference"])

    # Define features (X) and target variable (y)
    features = [col for col in df.columns if col not in ["station_id", "demand"]]
    X = df[features]
    y = df["demand"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


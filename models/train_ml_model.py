# Script for Step 4: Demand Prediction Model Training D:\Desktop\ev_charging_optimization\models\train_ml_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

# --- Check for geohash library ---
# This check determines if location-based features can be engineered.
# Prioritize geohash2 as requested, fallback to geohash.
# Install using: pip install geohash2 geohash
try:
    import geohash2
    GEOHASH_LIB = geohash2
    HAS_GEOHASH = True
    print("Geohash2 library found. Location-based features will be engineered using geohash2.")
except ImportError:
    # If geohash2 is not found, try geohash
    try:
        import geohash
        GEOHASH_LIB = geohash
        HAS_GEOHASH = True
        print("Geohash library found. Location-based features will be engineered using geohash.")
    except ImportError:
        GEOHASH_LIB = None
        HAS_GEOHASH = False
        print("Warning: Neither 'geohash2' nor 'geohash' library found. Location-based features will be skipped.")


from datetime import datetime, timedelta

# Define directories
DATA_DIR = 'data'
MODELS_DIR = 'models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Helper Function to Generate Simulated Data (More Detailed for Training) ---
# This function generates data *with* features needed for Step 4's preprocessing
# Replace this function and its call if you have your charging_stations.csv
# with the necessary columns (timestamp, Connector Type, Network Provider, etc.)
def generate_simulated_training_data_with_features(num_records=10000):
    """Generates simulated historical charging station demand data with relevant features."""
    print(f"Generating {num_records} simulated training data records with additional features for demo...")

    # Base coordinates (e.g., center of San Francisco)
    base_lat, base_lon = 37.7749, -122.4194

    # Simulate station locations (within a small radius)
    lats = base_lat + (np.random.rand(num_records) - 0.5) * 0.1 # roughly within ~10km range
    lons = base_lon + (np.random.rand(num_records) - 0.5) * 0.1

    # Simulate Station Attributes (including categorical ones needed for encoding)
    station_ids = [f'Station_{i}' for i in range(num_records)]
    prices = np.random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 'Free'], num_records) # Price per kWh or Free
    availabilities = np.random.randint(0, 10, num_records) # Number of available connectors
    wait_times = np.random.randint(0, 60, num_records) # Simulated wait time in minutes
    maintenance = np.random.choice([True, False], num_records, p=[0.05, 0.95]) # Maintenance status (less frequent)
    connector_types = np.random.choice(['Type 1', 'Type 2', 'CCS', 'CHAdeMO', 'Tesla'], num_records)
    network_providers = np.random.choice(['ChargePoint', 'EVgo', 'Electrify America', 'Tesla Supercharger', 'Shell Recharge', 'Other'], num_records)
    power_kW = np.random.choice([50, 75, 150, 250, 350], num_records) # Power rating in kW

    # Simulate Timestamps (over a period, e.g., past 6 months)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    timestamps = [start_time + (end_time - start_time) * np.random.rand() for _ in range(num_records)]

    # Simulate Demand (Target Variable) - Simple heuristic
    # Influenced by time of day, day of week, availability, price, connector type, network, maintenance
    hours = np.array([t.hour for t in timestamps])
    days = np.array([t.weekday() for t in timestamps]) # Monday=0, Sunday=6

    # Base demand influenced by hour (higher during peak hours 8-10, 17-19) and day of week (higher on weekdays)
    demand_base = np.where(np.isin(days, [0, 1, 2, 3, 4]), # Weekdays
        np.where((hours >= 8) & (hours <= 10) | (hours >= 17) & (hours <= 19), 5, # Peak weekday
                 np.where((hours >= 11) & (hours <= 16), 3, 1)), # Off-peak weekday daytime
        np.where((hours >= 11) & (hours <= 18), 4, 2) # Weekend daytime
    )

    # Add influence from other factors (simplified)
    # Availability: Inverse relationship
    # Price: Lower price increases demand (except Free which has specific high demand)
    # Maintenance: Significantly reduces demand
    # Connector Type/Network: Assume some types/networks are more popular (needs real data for actual patterns)
    # Power_kW: Higher power might attract more users

    demand_modifier = np.ones(num_records)

    # Influence of Availability
    demand_modifier *= (1 + (10 - availabilities) / 20) # Range 0.5 to 1.5

    # Influence of Price
    # Robust FIX for ValueError using boolean indexing to separate string and numeric
    price_effect = np.zeros(num_records, dtype=float) # Initialize as float array

    # Find where price is 'Free' (boolean mask)
    is_free = (prices == 'Free')

    # Assign effect for 'Free' prices using the mask
    price_effect[is_free] = 1.5

    # Calculate effect for non-'Free' prices
    # Select non-'Free' prices using the inverse mask and convert ONLY those to float
    numeric_prices = prices[~is_free].astype(float)
    # Assign calculated effect back to the correct positions using the inverse mask
    price_effect[~is_free] = (0.6 - numeric_prices) / 0.6

    demand_modifier *= (1 + price_effect)


    # Influence of Maintenance
    demand_modifier *= np.where(maintenance, 0.1, 1) # 90% reduction if under maintenance

    # Influence of Connector Type (example weights)
    connector_weights = {'Type 1': 1.0, 'Type 2': 1.2, 'CCS': 1.5, 'CHAdeMO': 1.4, 'Tesla': 1.6}
    demand_modifier *= np.array([connector_weights.get(ct, 1.0) for ct in connector_types])

    # Influence of Network Provider (example weights)
    network_weights = {'ChargePoint': 1.3, 'EVgo': 1.4, 'Electrify America': 1.5, 'Tesla Supercharger': 1.8, 'Shell Recharge': 1.1, 'Other': 1.0}
    demand_modifier *= np.array([network_weights.get(npv, 1.0) for npv in network_providers])

    # Influence of Power (linear relationship)
    demand_modifier *= (1 + power_kW / 400) # Higher power, more demand

    # Combine base demand and modifiers, add noise
    demand = demand_base * demand_modifier * (1 + np.random.randn(num_records) * 0.2) # Add random noise

    # Ensure demand is non-negative integer (simulating connections/sessions)
    demand = np.maximum(0, np.round(demand)).astype(int) # Round to nearest integer

    data = pd.DataFrame({
        'Station ID': station_ids,
        'Lat': lats,
        'Long': lons,
        'Price': prices,
        'Availability': availabilities,
        'Wait Time': wait_times,
        'Maintenance': maintenance,
        'Connector Type': connector_types,
        'Network Provider': network_providers,
        'Power_kW': power_kW,
        'timestamp': timestamps,
        'Demand': demand # Target variable
    })

    return data


# --- Load or Generate Data ---
# Uncomment the section below if you have your charging_stations.csv with the necessary columns
# (at least timestamp, Lat, Long, Price, Availability, Wait Time, Maintenance, Connector Type, Network Provider, Power_kW, Demand)
# data_path = os.path.join(DATA_DIR, 'charging_stations.csv') # From Step 2
# if os.path.exists(data_path):
#     print(f"Loading training data from {data_path}...")
#     df = pd.read_csv(data_path)
#     # Ensure 'timestamp' is datetime if loading from CSV
#     if 'timestamp' in df.columns:
#          df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
# else:
#     print(f"Warning: {data_path} not found or missing columns. Generating simulated data with features for demo.")
df = generate_simulated_training_data_with_features(num_records=10000) # Generate richer data for the demo


# --- Step 4.2: Preprocess the data ---
print("\nPreprocessing data...")

# Drop rows with missing crucial values if necessary (e.g., Lat/Long, Demand, timestamp)
initial_rows = len(df)
crucial_cols = ['Lat', 'Long', 'timestamp', 'Demand']
# Check which crucial columns actually exist in the DataFrame before dropping
existing_crucial_cols = [col for col in crucial_cols if col in df.columns]
if existing_crucial_cols:
    # Use .copy() after dropna to prevent SettingWithCopyWarning downstream
    df = df.dropna(subset=existing_crucial_cols).copy()
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with missing values in {existing_crucial_cols}.")
else:
    print("Warning: No crucial columns found in data for dropping NaNs.")

# Handle cases where the DataFrame is empty after potential drops or starts empty
if df.empty:
    print("Error: DataFrame is empty after initial loading or dropping crucial NaNs. Cannot proceed with training.")
    exit() # Exit if no data to train on


# Handle missing values for other columns (simple imputation for demo)
# Identify columns that might still have NaNs after dropping crucial ones
# Exclude 'Station ID' as it's not a feature and might have NaNs, but we won't fill it
cols_to_impute = [col for col in df.columns if col != 'Station ID' and df[col].isnull().any()]

for col in cols_to_impute:
    # Check dtype defensively before imputing
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].mean(), inplace=True)
    elif df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
         df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True) # Use mode for categorical, default to 'Unknown'
    elif pd.api.types.is_bool_dtype(df[col]):
         df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else False, inplace=True) # Use mode for boolean, default to False
    else:
        print(f"Warning: Column '{col}' has unexpected type {df[col].dtype} with missing values remaining after dropping crucial rows. Consider specific handling.")


# Identify original categorical columns that will be encoded and then dropped
original_categorical_cols = ['Price', 'Maintenance', 'Connector Type', 'Network Provider']
categorical_cols_for_encoding = [col for col in original_categorical_cols if col in df.columns]

# Initialize dictionary to store fitted Label Encoders
label_encoders = {}

# Apply Label Encoding to identified categorical columns
if categorical_cols_for_encoding:
    print("\nApplying Label Encoding...")
    for col in categorical_cols_for_encoding:
        if col in df.columns: # Double check column exists
            # Ensure the column is treated as string for consistent encoding
            # Use .loc[:, col] for assignment to avoid SettingWithCopyWarning and ensure correct dtype
            df.loc[:, col] = df[col].astype(str)
            le = LabelEncoder()
            # Fit the encoder and transform the column
            # Use .loc[:, new_col] for assignment
            df.loc[:, f'{col}_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le # Store the fitted encoder object
            print(f"Encoded '{col}' into '{col}_encoded'. Found {len(le.classes_)} unique classes.")
        else:
             # This warning should be redundant if handled in list creation, but as a fallback
             print(f"Error: Column '{col}' unexpectedly missing during encoding loop.")
else:
    print("\nNo categorical columns identified for Label Encoding.")


# Engineer time-based features from 'timestamp'
print("\nEngineering time-based features...")
# Ensure 'timestamp' is datetime type - crucial if loading from CSV
if 'timestamp' in df.columns:
    try:
        # Ensure it's datetime, errors='coerce' turns unparseable dates into NaT
        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Drop rows where datetime conversion failed AFTER initial crucial drop
        initial_rows_ts = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        if len(df) < initial_rows_ts:
             print(f"Dropped {initial_rows_ts - len(df)} rows with invalid timestamp values.")
             if df.empty:
                  print("Error: DataFrame is empty after dropping invalid timestamps. Cannot proceed with training.")
                  exit()

        df.loc[:, 'hour'] = df['timestamp'].dt.hour.astype(int) # Ensure int type
        df.loc[:, 'dayofweek'] = df['timestamp'].dt.dayofweek.astype(int) # Ensure int type
        df.loc[:, 'dayofyear'] = df['timestamp'].dt.dayofyear.astype(int) # Ensure int type
        df.loc[:, 'weekofyear'] = df['timestamp'].dt.isocalendar().week.astype(int) # Ensure int type
        df.loc[:, 'month'] = df['timestamp'].dt.month.astype(int) # Ensure int type
        df.loc[:, 'year'] = df['timestamp'].dt.year.astype(int) # Ensure int type

        print(f"Engineered: hour, dayofweek, dayofyear, weekofyear, month, year from timestamp.")
        df = df.drop('timestamp', axis=1) # Drop original timestamp column
    except Exception as e:
         print(f"Error processing 'timestamp' column: {e}. Skipping time-based features.")
         if 'timestamp' in df.columns: # Check if column still exists before dropping
             df = df.drop('timestamp', axis=1) # Drop if engineering failed


# Engineer location-based features (Geohash)
print("\nEngineering location-based features...")
# Only proceed if a geohash library was successfully imported and Lat/Long columns exist
if HAS_GEOHASH and 'Lat' in df.columns and 'Long' in df.columns:
    # Use a precision that balances location detail and generalization
    GEOHASH_PRECISION = 6 # Adjust as needed (e.g., 5 for broader area, 7 for more specific)
    try:
        # Ensure Lat/Long are numeric
        df.loc[:, 'Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
        df.loc[:, 'Long'] = pd.to_numeric(df['Long'], errors='coerce')
        initial_rows_loc = len(df)
        # Drop rows if Lat/Long became NaN after coercion, BEFORE geohash encoding
        df = df.dropna(subset=['Lat', 'Long']).copy() # Use .copy()
        if len(df) < initial_rows_loc:
             print(f"Dropped {initial_rows_loc - len(df)} rows with invalid Lat/Long values.")
             if df.empty:
                  print("Error: DataFrame is empty after dropping invalid Lat/Longs. Cannot proceed with training.")
                  exit()

        # Use .loc for assigning the new 'geohash' column string
        # Check for potential errors during encoding using apply, although dropna above should handle NaNs
        try:
             # Use the imported GEOHASH_LIB (either geohash or geohash2)
             df.loc[:, 'geohash'] = df.apply(lambda row: GEOHASH_LIB.encode(row['Lat'], row['Long'], precision=GEOHASH_PRECISION), axis=1)
        except Exception as apply_err:
             print(f"Error during geohash string generation: {apply_err}. 'geohash' column might be incomplete.")
             df.loc[:, 'geohash'] = None # Ensure column exists even if apply failed


        # Geohash string is a high-cardinality categorical feature, also needs encoding
        if 'geohash' in df.columns and df['geohash'].notnull().any():
             # Use LabelEncoder for geohash strings
             le_geohash = LabelEncoder()
             # Need to handle potential NaNs in geohash column if apply failed for some rows
             non_null_geohashes_indices = df['geohash'].dropna().index # Get indices of non-null geohashes
             if not non_null_geohashes_indices.empty:
                 # Fit encoder only on non-null values and assign encoded values back using .loc
                 le_geohash.fit(df.loc[non_null_geohashes_indices, 'geohash'])
                 df.loc[non_null_geohashes_indices, 'geohash_encoded'] = le_geohash.transform(df.loc[non_null_geohashes_indices, 'geohash'])

                 label_encoders['geohash'] = le_geohash # Store the fitted encoder
                 print(f"Engineered and encoded geohash feature (precision {GEOHASH_PRECISION}). Found {len(le_geohash.classes_)} unique hashes.")
                 # Fill NaNs in the new encoded column with -1 (or another indicator) and ensure integer type
                 # FIX: Corrected typo from 'geash_encoded' to 'geohash_encoded'
                 df.loc[:, 'geohash_encoded'] = df['geohash_encoded'].fillna(-1).astype(int)
             else:
                 df.loc[:, 'geohash_encoded'] = -1 # Assign -1 if no valid geohashes were encoded at all
                 print("Warning: No valid geohashes encoded.")

        else:
             df.loc[:, 'geohash_encoded'] = -1 # Assign -1 if geohash column was empty or missing
             print("Warning: 'geohash' column not created or is empty after encoding attempt.")

        # Drop original Lat/Long and geohash string columns
        df = df.drop(['Lat', 'Long', 'geohash'], axis=1, errors='ignore')

    except Exception as e:
         print(f"Error during geohash engineering: {e}. Skipping geohash.")
         # Drop Lat/Long if geohash engineering fails
         df = df.drop(['Lat', 'Long'], axis=1, errors='ignore')

else:
    if not HAS_GEOHASH:
         print("Geohash library not available. Skipping geohash engineering.")
    else:
        print("Warning: 'Lat' or 'Long' columns not found. Skipping geohash engineering.")
    # Ensure Lat/Long are dropped if they exist but couldn't be used
    df = df.drop(['Lat', 'Long'], axis=1, errors='ignore')


# Apply numerical scaling to numerical features
# Identify columns that are numeric (int, float) excluding the target ('Demand')
# and excluding the already encoded integer categorical features AND integer time features
numerical_features_to_scale = df.select_dtypes(include=np.number).columns.tolist()
# List columns that are results of Label Encoding or Geohash Encoding
encoded_cols = [f'{col}_encoded' for col in categorical_cols_for_encoding if f'{col}_encoded' in df.columns]
if 'geohash_encoded' in df.columns:
    encoded_cols.append('geohash_encoded')

# List integer time-based features that shouldn't be scaled by MinMaxScaler
time_features = ['hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'year']
existing_time_features = [col for col in time_features if col in df.columns]


# List columns to explicitly exclude from scaling
excluded_cols = ['Demand', 'Station ID','Availability','Wait Time'] + encoded_cols + existing_time_features

# Filter numerical columns to get only those that should be scaled
numerical_features_to_scale = [col for col in numerical_features_to_scale if col not in excluded_cols]

scaler = None # Initialize scaler to None
if numerical_features_to_scale:
    print(f"\nApplying MinMaxScaler to numerical features: {numerical_features_to_scale}...")
    scaler = MinMaxScaler()
    # Fit and transform only the identified numerical features
    # Use .loc for assignment, MinMaxScaler outputs floats, ensure target cols accept floats
    for col in numerical_features_to_scale:
         # Scale column by column and ensure it's stored as float to avoid warnings
         df.loc[:, col] = scaler.fit_transform(df[[col]]).astype(float) # Explicitly cast to float

    print("MinMaxScaler applied.")
else:
    print("\nNo numerical features identified for scaling (excluding target, encoded, and time features).")


# --- Define the final set of features for the model ---
# These are all columns EXCEPT 'Station ID', 'Demand', AND the original categorical/geohash string columns
# Explicitly list original columns to drop
original_cols_to_drop = original_categorical_cols + ['Station ID', 'geohash', 'Lat', 'Long', 'timestamp'] # Include original geohash, lat/lon, timestamp

# Drop original columns that are not intended as features for the model
# Use errors='ignore' in case some columns didn't exist after earlier steps
df_processed = df.drop(columns=[col for col in original_cols_to_drop if col in df.columns], errors='ignore')


# Ensure the target column 'Demand' is still present and is a numerical type (int or float)
if 'Demand' not in df_processed.columns:
    print("Error: 'Demand' column missing after dropping features. Cannot train model.")
    exit()
# Ensure Demand is numerical (should be from simulation/initial loading, but double-check)
df_processed.loc[:, 'Demand'] = pd.to_numeric(df_processed['Demand'], errors='coerce')
df_processed.dropna(subset=['Demand'], inplace=True) # Drop rows where Demand is NaN after coercion

if df_processed.empty:
     print("Error: DataFrame is empty after ensuring 'Demand' is valid. Cannot train model.")
     exit()

# Get the final list of feature names from the processed DataFrame (excluding the target)
feature_names = [col for col in df_processed.columns if col != 'Demand']

# Ensure the order is consistent - important when loading data for prediction later
feature_names.sort() # Sorting ensures consistent order

print(f"\nProcessed data shape: {df_processed.shape}")
print("Processed data columns:", df_processed.columns.tolist()) # Print final columns
print("\nSample of processed data after preprocessing:")
print(df_processed.head())


# --- Step 4.3: Store feature names ---
print(f"\nFeatures used for training ({len(feature_names)}): {feature_names}")

try:
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=4) # Use indent for readability
    print(f"Feature names saved to {os.path.join(MODELS_DIR, 'feature_names.json')}")
except Exception as e:
    print(f"Error saving feature names: {e}")


# --- Step 4.4: Train LightGBM Regression Model ---
print("\nTraining LightGBM model...")

# Separate features (X) and target (y)
# Use the saved feature names list to ensure correct columns and order
X = df_processed[feature_names]
y = df_processed['Demand']

# Check if X is empty after preprocessing
if X.empty:
    print("Error: Feature DataFrame is empty after preprocessing. Cannot train model.")
    exit()

# Check if target y is empty
if y.empty:
     print("Error: Target Series is empty after preprocessing. Cannot train model.")
     exit()

# Convert X dtypes to numeric if any object types remain (shouldn't happen if dropping worked)
# This is a final defensive step before training
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"Warning: Column '{col}' is still object type before training. Attempting to convert to numeric.")
        X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')
        # After coercion, NaNs might be introduced, handle them (e.g., impute or drop)
        if X[col].isnull().any():
             print(f"Warning: NaNs introduced in '{col}' after numeric coercion. Imputing with mean.")
             X.loc[:, col].fillna(X[col].mean(), inplace=True) # Impute NaNs in X


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train LightGBM model
# Using parameters that often work well, you can tune these later (Step 20)
lgbm = lgb.LGBMRegressor(objective='regression_l1', # MAE objective, good for integer counts and robustness
                         metric='mae',          # Evaluate using MAE during training
                         n_estimators=1000,     # Number of boosting rounds - will use early stopping
                         learning_rate=0.05,
                         num_leaves=31,         # Standard value
                         max_depth=-1,          # No limit on tree depth
                         random_state=42,
                         n_jobs=-1,             # Use all available cores
                         colsample_bytree=0.8,  # Fraction of features used per tree
                         subsample=0.8,         # Fraction of data used per tree
                         reg_alpha=0.1,         # L1 regularization
                         reg_lambda=0.1)        # L2 regularization

# Train the model with early stopping
print("Starting model training...")
# Use X_test, y_test as evaluation set for early stopping
# Pass feature_name and categorical_feature for LightGBM warnings/optimizations
# Note: LightGBM can handle categorical features natively if specified by index or name,
# but since we've already LabelEncoded them, we treat them as numerical for LGBM input.
# This is a common and valid approach when preserving the LabelEncoder for later decoding.
try:
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             eval_metric='mae', # Monitor MAE on evaluation set
             callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]) # Stop if MAE doesn't improve for 50 rounds
    print("LightGBM model training complete.")

except Exception as e:
    print(f"Error during model training: {e}")
    # Exit or handle training failure
    exit()


# --- Step 4.5 & 4.6: Evaluate model performance ---
print("\nEvaluating model performance on the test set...")

try:
    y_pred = lgbm.predict(X_test)

    # Ensure predictions are non-negative integers, as demand is a count
    y_pred = np.maximum(0, np.round(y_pred)).astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred) # Calculate R-squared as per plan

    print(f"Model Evaluation on Test Set:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")
except Exception as e:
    print(f"Error during model evaluation: {e}")


# --- Step 4.7: Save the trained model and preprocessing artifacts ---
print("\nSaving trained model and preprocessing artifacts...")

try:
    # Save the trained LightGBM model
    model_path = os.path.join(MODELS_DIR, 'demand_model.joblib')
    joblib.dump(lgbm, model_path)
    print(f"Trained model saved to {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")

try:
    # Save the dictionary containing all fitted LabelEncoder objects
    # This is CRUCIAL for decoding later (Step 11)
    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.joblib')
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved to {encoders_path}")
except Exception as e:
    print(f"Error saving label encoders: {e}")

if scaler: # Only save the scaler if numerical scaling was applied
    try:
        # Save the fitted MinMaxScaler object
        # This is CRUCIAL for scaling new numerical data consistently (Step 11)
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    except Exception as e:
        print(f"Error saving scaler: {e}")
else:
    print("No scaler needed or saved as no numerical features were scaled.")


print("\nStep 4: Model training and saving complete.")
print(f"Artifacts available in {MODELS_DIR}: demand_model.joblib, label_encoders.joblib, feature_names.json" + (", scaler.joblib" if scaler else "."))

# --- Example Usage Note ---
# To load these artifacts later for prediction in Streamlit (Step 11):
# import joblib
# import json
# import os
# MODELS_DIR = 'models' # Define this path in your Streamlit app or import a config
#
# try:
#     loaded_model = joblib.load(os.path.join(MODELS_DIR, 'demand_model.joblib'))
#     loaded_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.joblib'))
#     loaded_feature_names = json.load(open(os.path.join(MODELS_DIR, 'feature_names.json')))
#     loaded_scaler = None
#     scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
#     if os.path.exists(scaler_path):
#         loaded_scaler = joblib.load(scaler_path)
#     else:
#         print("Warning: Scaler file not found. Assuming no scaling needed for prediction inputs.")
#
#     print("\nSuccessfully loaded model and preprocessing artifacts.")
#     # Now you can use loaded_model, loaded_encoders, loaded_scaler, loaded_feature_names
#
# except FileNotFoundError as e:
#     print(f"Error loading model artifacts: {e}. Make sure you have run train_demand_model.py first.")
# except Exception as e:
#     print(f"An unexpected error occurred during loading: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # The script automatically runs the training process when executed directly.
    # The print statements throughout the script will indicate progress and results.
    pass # The script's logic runs outside this block, so pass is fine here
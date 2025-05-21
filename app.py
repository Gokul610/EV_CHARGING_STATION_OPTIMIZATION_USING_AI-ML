import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from typing import Dict, Any, List, Optional
import sys
import datetime
import folium
from streamlit_folium import folium_static
import plotly.express as px
import geopandas as gpd
from streamlit_js_eval import streamlit_js_eval
import diskcache
from typing import Tuple
from streamlit_folium import st_folium

# Ensure utils directory is in the Python path (Step 9.1)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Set Streamlit page configuration (Step 17.1)
st.set_page_config(page_title="EV Charging Optimization", layout="wide")

# Initialize cache (Step 19.1)
cache = diskcache.Cache(os.path.join(PROJECT_ROOT, "cache"))

# Configure logging (Step 18.3)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fallback imports for utilities (Step 9.2)
try:
    from utils.llm_data_prep import fetch_latest_data_from_duckdb, format_data_for_llm
except ImportError:
    logging.warning("llm_data_prep.py not found. Using fallback data functions.")
    def fetch_latest_data_from_duckdb(table: str, lat: float = None, lon: float = None, radius: float = None, entity_ids: List[str] = None) -> pd.DataFrame:
        if table == 'station_status':
            return pd.DataFrame([
                {
                    'station_id': f'SIM_{i}',
                    'title': f'Simulated Station {i}',
                    'lat': (lat or 9.0548) + np.random.uniform(-0.01, 0.01),
                    'lon': (lon or 77.4335) + np.random.uniform(-0.01, 0.01),
                    'distance_m': np.random.uniform(100, radius or 5000),
                    'operational_status': 'operational',
                    'available_connectors': np.random.randint(1, 10),
                    'total_connectors': 10,
                    'price': np.random.uniform(0, 5),
                    'supported_connectors': np.random.choice(['CCS', 'Type 2', 'CHAdeMO']),
                    'network_provider': np.random.choice(['Electrify America', 'ChargePoint']),
                    'timestamp': datetime.datetime.now().isoformat()
                } for i in range(5)
            ])
        elif table == 'traffic_conditions':
            return pd.DataFrame([{'jam_factor': 3.0, 'timestamp': datetime.datetime.now().isoformat()}])
        elif table == 'weather_info':
            return pd.DataFrame([{'temperature': 20.0, 'humidity': 60.0, 'timestamp': datetime.datetime.now().isoformat()}])
        elif table == 'usage_events':
            return pd.DataFrame([{'station_id': entity_ids[0] if entity_ids else 'SIM_1', 'event_type': 'charging', 'timestamp': datetime.datetime.now().isoformat()}])
        return pd.DataFrame()
    def format_data_for_llm(stations_df: pd.DataFrame, traffic_df: pd.DataFrame, weather_df: pd.DataFrame) -> Dict[str, Any]:
        return {
            'stations': [f"{row['title']}: {row['supported_connectors']}, {row['available_connectors']} available" for _, row in stations_df.iterrows()],
            'traffic': [f"Jam factor: {row['jam_factor']}" for _, row in traffic_df.iterrows()],
            'weather': [f"Temp: {row['temperature']}°C, Humidity: {row['humidity']}%" for _, row in weather_df.iterrows()]
        }

try:
    from utils.llm_utils import get_llm_suggestion, STATION_SUGGESTION_PROMPT_TEMPLATE, DEMAND_PREDICTION_EXPLANATION_TEMPLATE, ROUTE_ADVICE_PROMPT_TEMPLATE, GEOSPATIAL_INSIGHTS_PROMPT_TEMPLATE
except ImportError:
    logging.warning("llm_utils.py not found. LLM features disabled.")
    def get_llm_suggestion(prompt_template: str, data_dict: Dict[str, Any], retries: int = 3) -> str:
        return "LLM unavailable due to missing llm_utils.py."
    STATION_SUGGESTION_PROMPT_TEMPLATE = ""
    DEMAND_PREDICTION_EXPLANATION_TEMPLATE = ""
    ROUTE_ADVICE_PROMPT_TEMPLATE = ""
    GEOSPATIAL_INSIGHTS_PROMPT_TEMPLATE = ""

try:
    from utils.geo_utils import run_geospatial_analysis, VORONOI_OUTPUT_PATH, GRID_ANALYSIS_OUTPUT_PATH, fetch_station_locations_from_duckdb
except ImportError:
    logging.warning("geo_utils.py not found. Using fallback geospatial data.")
    VORONOI_OUTPUT_PATH = "data/voronoi.geojson"
    GRID_ANALYSIS_OUTPUT_PATH = "data/grid_analysis.geojson"
    def fetch_station_locations_from_duckdb() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                'title': ['Simulated Station 1', 'Simulated Station 2'],
                'lat': [9.0548, 9.0648],
                'lon': [77.4335, 77.4435],
                'supported_connectors': ['CCS', 'Type 2'],
                'operational_status': ['operational', 'operational'],
                'available_connectors': [5, 3],
                'total_connectors': [10, 10],
                'network_provider': ['Electrify America', 'ChargePoint']
            },
            geometry=gpd.points_from_xy([77.4335, 77.4435], [9.0548, 9.0648])
        )
    def run_geospatial_analysis():
        voronoi_gdf = gpd.GeoDataFrame(
            {'Station ID': ['SIM_1', 'SIM_2'], 'distance_m': [1000, 1500]},
            geometry=[gpd.GeoSeries.from_wkt("POLYGON ((77.43 9.05, 77.44 9.05, 77.44 9.06, 77.43 9.06, 77.43 9.05))").iloc[0],
                      gpd.GeoSeries.from_wkt("POLYGON ((77.44 9.06, 77.45 9.06, 77.45 9.07, 77.44 9.07, 77.44 9.06))").iloc[0]]
        )
        grid_gdf = gpd.GeoDataFrame(
            {'station_count': [3, 1], 'index': [0, 1]},
            geometry=[gpd.GeoSeries.from_wkt("POLYGON ((77.43 9.05, 77.44 9.05, 77.44 9.06, 77.43 9.06, 77.43 9.05))").iloc[0],
                      gpd.GeoSeries.from_wkt("POLYGON ((77.44 9.06, 77.45 9.06, 77.45 9.07, 77.44 9.07, 77.44 9.06))").iloc[0]]
        )
        voronoi_gdf.to_file(VORONOI_OUTPUT_PATH)
        grid_gdf.to_file(GRID_ANALYSIS_OUTPUT_PATH)

try:
    from utils.api_utils import geocode_place, fetch_optimal_route_summary, fetch_real_time_station_data, fetch_route_geometry
except ImportError:
    logging.error("api_utils.py not found. API features disabled.")
    def geocode_place(place_name: str) -> Optional[Tuple[float, float]]:
        return (9.0548, 77.4335)
    def fetch_optimal_route_summary(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, **kwargs) -> Optional[Dict[str, Any]]:
        return {
            'distance_km': 10.0,
            'duration_min': 15.0,
            'traffic_delay_min': 2.0,
            'text': "Approx. 10.0 km, 15.0 min (2.0 min traffic delay)",
            'polyline': None
        }
    def fetch_real_time_station_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
        return fetch_latest_data_from_duckdb('station_status', lat, lon, radius)
    def fetch_route_geometry(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> Optional[List[Tuple[float, float]]]:
        return [(origin_lat, origin_lon), (dest_lat, dest_lon)]

# Model paths (Step 4)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DEMAND_MODEL_PATH = os.path.join(MODELS_DIR, 'demand_model.joblib')
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, 'label_encoders.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')

# Load model and artifacts (Step 4.1)
@st.cache_resource
def load_model_and_artifacts():
    try:
        model = joblib.load(DEMAND_MODEL_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        feature_names = json.load(open(FEATURE_NAMES_PATH))
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        logging.info("Model and artifacts loaded successfully")
        return model, label_encoders, scaler, feature_names
    except Exception as e:
        logging.error(f"Model loading error: {e}", exc_info=True)
        return None, None, None, None

demand_model, label_encoders, scaler, feature_names = load_model_and_artifacts()
if demand_model is None or label_encoders is None or feature_names is None:
    st.error("🔴 Application cannot run due to model loading failure.")
    st.stop()

# Preprocess input for prediction (Step 4.2)
def preprocess_input_for_prediction(input_data: Dict[str, Any], loaded_encoders: Dict[str, Any], loaded_scaler: Optional[Any], training_feature_names: List[str]) -> pd.DataFrame:
    input_df = pd.DataFrame([input_data])
    processed_input_df = pd.DataFrame(np.nan, index=[0], columns=training_feature_names)

    for col in input_df.columns:
        if col in processed_input_df.columns:
            processed_input_df[col] = input_df[col]

    numerical_cols = [
        col for col in training_feature_names
        if col not in [f'{c}_encoded' for c in loaded_encoders.keys()] + ['geohash_encoded', 'hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'year']
    ]
    for col in numerical_cols:
        if col in processed_input_df.columns:
            processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce').fillna(0).astype(float)

    now = datetime.datetime.now()
    if 'hour' in training_feature_names: processed_input_df['hour'] = int(now.hour)
    if 'dayofweek' in training_feature_names: processed_input_df['dayofweek'] = int(now.weekday())
    if 'dayofyear' in training_feature_names: processed_input_df['dayofyear'] = int(now.timetuple().tm_yday)
    if 'weekofyear' in training_feature_names: processed_input_df['weekofyear'] = int(now.isocalendar().week)
    if 'month' in training_feature_names: processed_input_df['month'] = int(now.month)
    if 'year' in training_feature_names: processed_input_df['year'] = int(now.year)

    raw_lat = input_data.get('latitude')
    raw_lon = input_data.get('longitude')
    if 'geohash_encoded' in training_feature_names:
        if raw_lat is not None and raw_lon is not None and pd.notnull(raw_lat) and pd.notnull(raw_lon):
            try:
                import geohash2
                geohash_string = geohash2.encode(raw_lat, raw_lon, precision=6)
                le_geohash = loaded_encoders.get('geohash')
                processed_input_df['geohash_encoded'] = le_geohash.transform([geohash_string])[0] if geohash_string in le_geohash.classes_ else -1
            except Exception as e:
                logging.error(f"Geohash encoding error: {e}")
                processed_input_df['geohash_encoded'] = -1
        else:
            processed_input_df['geohash_encoded'] = -1
        processed_input_df['geohash_encoded'] = processed_input_df['geohash_encoded'].fillna(-1).astype(int)

    for col in [c for c in loaded_encoders.keys() if c != 'geohash']:
        encoded_col = f'{col}_encoded'
        if encoded_col in training_feature_names:
            raw_value = input_data.get(col)
            if raw_value is not None and pd.notnull(raw_value):
                le = loaded_encoders[col]
                raw_value_str = str(raw_value)
                processed_input_df[encoded_col] = le.transform([raw_value_str])[0] if raw_value_str in le.classes_ else -1
            else:
                processed_input_df[encoded_col] = -1
            processed_input_df[encoded_col] = processed_input_df[encoded_col].fillna(-1).astype(int)
        if col in processed_input_df.columns and f'{col}_encoded' in training_feature_names:
            processed_input_df = processed_input_df.drop(columns=[col], errors='ignore')

    if loaded_scaler:
        numerical_features = [
            col for col in training_feature_names
            if col in processed_input_df.columns and pd.api.types.is_numeric_dtype(processed_input_df[col]) and
            col not in [f'{c}_encoded' for c in loaded_encoders.keys()] + ['geohash_encoded', 'hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'year', 'Availability', 'Wait Time']
        ]
        if numerical_features:
            try:
                scaled_values = loaded_scaler.transform(processed_input_df[numerical_features])
                processed_input_df[numerical_features] = scaled_values
            except Exception as e:
                logging.error(f"Scaler error: {e}")
                processed_input_df[numerical_features] = 0.0

    final_input_df = processed_input_df.reindex(columns=training_feature_names)
    cols_to_cast_to_int = [f'{col}_encoded' for col in loaded_encoders.keys()] + ['geohash_encoded', 'hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'year']
    for col in cols_to_cast_to_int:
        if col in final_input_df.columns:
            final_input_df[col] = pd.to_numeric(final_input_df[col], errors='coerce').fillna(-1).astype(int)
    for col in [c for c in training_feature_names if c not in cols_to_cast_to_int]:
        if col in final_input_df.columns:
            final_input_df[col] = pd.to_numeric(final_input_df[col], errors='coerce').fillna(0.0).astype(float)

    if list(final_input_df.columns) != training_feature_names:
        logging.error("Feature mismatch in prediction input")
        raise ValueError("Feature mismatch in prediction input")

    return final_input_df

# Decode categorical features (Step 11.1, 12.1, 16.1)
def decode_categorical_features(input_data: Dict[str, Any], loaded_encoders: Dict[str, Any]) -> Dict[str, Any]:
    decoded_data = {}
    for original_col, encoder in loaded_encoders.items():
        if original_col == 'geohash':
            continue
        if original_col in input_data and pd.notnull(input_data[original_col]):
            raw_value = str(input_data[original_col])
            decoded_data[original_col] = raw_value if raw_value in encoder.classes_ else f"{raw_value} (Unseen)"
        elif f'{original_col}_encoded' in input_data and pd.notnull(input_data[f'{original_col}_encoded']):
            encoded_value = input_data[f'{original_col}_encoded']
            try:
                decoded_data[original_col] = encoder.inverse_transform([encoded_value])[0] if 0 <= encoded_value < len(encoder.classes_) else f"Encoded: {encoded_value} (Unknown)"
            except Exception as e:
                decoded_data[original_col] = f"Decoding Error ({encoded_value})"
        else:
            decoded_data[original_col] = "N/A"

    raw_lat = input_data.get('latitude')
    raw_lon = input_data.get('longitude')
    if raw_lat is not None and raw_lon is not None and pd.notnull(raw_lat) and pd.notnull(raw_lon):
        try:
            import geohash2
            decoded_data['geohash'] = geohash2.encode(raw_lat, raw_lon, precision=6)
        except Exception:
            decoded_data['geohash'] = "Geohash encoding error"
    else:
        decoded_data['geohash'] = "Location not provided"
    return decoded_data

# Custom CSS (Step 17.1)
st.markdown("""
<style>
    .main .block-container { padding: 1rem; }
    .sidebar .sidebar-content { padding-top: 1rem; }
    h1, h2, h3 { color: #0E117F; }
    .stButton>button { color: #4F8BF9; border-radius: 0.5rem; padding: 0.5rem 1rem; }
    .stSuccess, .stWarning, .stError { padding: 1rem; border-radius: 0.5rem; }
    .tooltip { position: relative; display: inline-block; cursor: pointer; }
    .tooltip .tooltiptext { visibility: hidden; width: 120px; background-color: #555; color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -60px; opacity: 0; transition: opacity 0.3s; }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation (Step 9.3)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Navigation sidebar
st.sidebar.title("Navigation")
pages = {
    "🚗 Real-time Station Dashboard": "dashboard",
    "📊 Demand Prediction": "prediction",
    "🌍 Geospatial Analysis": "geospatial",
    "🗺️ Route Optimization": "route_optimization",
    "📡 Real-Time Data & Simulation": "simulation",
    "📂 Data & Insights Panel": "data_display"
}

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "dashboard"
current_page_index = list(pages.values()).index(st.session_state['current_page'])
selected_page_label = st.sidebar.radio("Go to:", list(pages.keys()), index=current_page_index)
st.session_state['current_page'] = pages[selected_page_label]

# Initialize session state for location
if 'user_lat' not in st.session_state:
    st.session_state['user_lat'] = None
    st.session_state['user_lon'] = None
    st.session_state['user_place_name'] = ""
if 'location_method' not in st.session_state:
    st.session_state['location_method'] = "Enter Place Name"  # Default to place name input

# Location input function
def get_user_location():
    """Get user location with permission-based geolocation and fallback options, without default coordinates."""
    st.subheader("Your Location")
    location_method = st.radio(
        "Location input method:",
        ("Detected Location", "Enter Lat/Lon", "Enter Place Name"),
        index=["Detected Location", "Enter Lat/Lon", "Enter Place Name"].index(st.session_state['location_method']),
        key="location_method_radio"
    )
    st.session_state['location_method'] = location_method

    latitude, longitude = st.session_state['user_lat'], st.session_state['user_lon']

    if location_method == "Detected Location":
        st.info("Click below to allow location access in your browser.")
        if st.button("Detect My Location", key="detect_location_button"):
            try:
                # Use a simpler JavaScript call to ensure permission prompt
                geolocation_result = streamlit_js_eval(
                    js_code="""
                        if (navigator.geolocation) {
                            return new Promise((resolve, reject) => {
                                navigator.geolocation.getCurrentPosition(
                                    (position) => {
                                        resolve(JSON.stringify({
                                            lat: position.coords.latitude,
                                            lon: position.coords.longitude
                                        }));
                                    },
                                    (error) => {
                                        resolve(JSON.stringify({error: error.message}));
                                    },
                                    {timeout: 10000, maximumAge: 60000}
                                );
                            });
                        } else {
                            return JSON.stringify({error: 'Geolocation not supported by your browser'});
                        }
                    """,
                    key='geolocation_request',
                    want_output=True
                )
                if geolocation_result and geolocation_result != 'streamlit_component_value_undefined':
                    geolocation_data = json.loads(geolocation_result)
                    if 'error' in geolocation_data:
                        st.error(f"Location detection failed: {geolocation_data['error']}. Switching to place name input.")
                        st.session_state['location_method'] = "Enter Place Name"
                        st.rerun()
                    elif geolocation_data.get('lat'):
                        latitude, longitude = geolocation_data['lat'], geolocation_data['lon']
                        st.session_state['user_lat'] = latitude
                        st.session_state['user_lon'] = longitude
                        st.session_state['user_place_name'] = "Automatically Detected"
                        st.success(f"✅ Detected location: ({latitude:.6f}, {longitude:.6f})")
                        st.rerun()
                else:
                    st.error("Geolocation request failed. Please try another method.")
                    st.session_state['location_method'] = "Enter Place Name"
                    st.rerun()
            except Exception as e:
                st.error(f"Geolocation error: {str(e)}. Switching to place name input.")
                logging.error(f"Geolocation error: {e}", exc_info=True)
                st.session_state['location_method'] = "Enter Place Name"
                st.rerun()
        if latitude is not None and longitude is not None:
            st.info(f"Current location: ({latitude:.6f}, {longitude:.6f})")
        else:
            st.warning("No location detected yet. Please click 'Detect My Location' or choose another method.")

    elif location_method == "Enter Lat/Lon":
        latitude = st.number_input(
            "Latitude:", value=latitude if latitude is not None else 0.0,
            min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f", key="lat_input"
        )
        longitude = st.number_input(
            "Longitude:", value=longitude if longitude is not None else 0.0,
            min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f", key="lon_input"
        )
        if latitude != 0.0 and longitude != 0.0 and -90 <= latitude <= 90 and -180 <= longitude <= 180:
            st.session_state['user_lat'], st.session_state['user_lon'] = latitude, longitude
            st.session_state['user_place_name'] = f"Manual: ({latitude:.6f}, {longitude:.6f})"
        else:
            st.warning("Please enter valid latitude (-90 to 90) and longitude (-180 to 180) values.")
            latitude, longitude = None, None

    else:  # Enter Place Name
        place_name = st.text_input("Place Name:", value=st.session_state['user_place_name'], key="place_name_input")
        if st.button("Geocode Place", key="geocode_button"):
            if place_name.strip():
                coords = geocode_place(place_name)
                if coords:
                    latitude, longitude = coords
                    st.session_state['user_lat'], st.session_state['user_lon'] = latitude, longitude
                    st.session_state['user_place_name'] = place_name
                    st.success(f"📍 Geocoded: {place_name} → ({latitude:.6f}, {longitude:.6f})")
                    st.rerun()
                else:
                    st.error(f"Could not geocode: {place_name}")
                    logging.error(f"Geocoding failed for: {place_name}")
                    latitude, longitude = None, None
            else:
                st.warning("Please enter a valid place name.")

    return latitude, longitude

# Real-time Station Dashboard
def show_real_time_dashboard():
    st.header("🚗 Real-Time Charging Station Dashboard")
    st.write("Find nearby charging stations with real-time data.")

    latitude, longitude = get_user_location()
    if latitude is None or longitude is None:
        st.error("Please provide a valid location to continue.")
        return
    st.info(f"Current location: {st.session_state['user_place_name']} ({latitude:.6f}, {longitude:.6f})")

    # Search options (Step 10.4)
    st.subheader("Search Options")
    radius = st.slider("Search Radius (meters):", 1000, 20000, 5000, step=500)
    limit = st.slider("Stations to Show:", 1, 50, 10)
    connector_options = ["All", "CCS", "Type 2", "CHAdeMO"]
    connector_filter = st.selectbox("Connector Type:", connector_options)
    price_filter = st.selectbox("Price Range:", ["All", "Free", "Low", "Medium", "High"])

    # Fetch stations (Step 10.5–10.9)
    if st.button("Find Stations"):
        with st.spinner("Fetching stations..."):
            try:
                cache_key = f"stations_{latitude}_{longitude}_{radius}"
                if cache_key in cache:
                    stations_df = cache[cache_key]
                else:
                    stations_df = fetch_latest_data_from_duckdb('station_status', lat=latitude, lon=longitude, radius=radius)
                    if stations_df.empty:
                        stations_df = fetch_real_time_station_data(latitude, longitude, radius)
                    cache[cache_key] = stations_df

                if not stations_df.empty:
                    # Apply filters
                    filtered_df = stations_df.copy()
                    if connector_filter != "All":
                        filtered_df = filtered_df[filtered_df['supported_connectors'] == connector_filter]
                    if price_filter != "All":
                        price_map = {"Free": 0, "Low": (0, 1), "Medium": (1, 3), "High": (3, float('inf'))}
                        if price_filter == "Free":
                            filtered_df = filtered_df[filtered_df['price'] == 0]
                        else:
                            low, high = price_map[price_filter]
                            filtered_df = filtered_df[(filtered_df['price'] > low) & (filtered_df['price'] <= high)]

                    if filtered_df.empty:
                        st.warning("No stations match the selected filters.")
                    else:
                        st.success(f"✅ Found {len(filtered_df)} stations.")
                        stations_df_display = filtered_df.sort_values(by='distance_m').head(limit)

                        # Folium map (Step 10.7)
                        m = folium.Map(location=[latitude, longitude], zoom_start=12)
                        folium.Marker([latitude, longitude], popup="Your Location", icon=folium.Icon(color='green')).add_to(m)
                        for _, row in stations_df_display.iterrows():
                            if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                                popup = (
                                    f"<b>{row['title']}</b><br>"
                                    f"Status: {row['operational_status']}<br>"
                                    f"Connectors: {row.get('available_connectors', '?')}/{row.get('total_connectors', '?')}<br>"
                                    f"Price: {row.get('price', 'N/A')}<br>"
                                    f"Network: {row.get('network_provider', 'N/A')}"
                                )
                                folium.Marker([row['lat'], row['lon']], popup=popup, tooltip=row['title'],icon=folium.Icon(icon='bolt', prefix='fa', color='blue')).add_to(m)
                        st_folium(m, width=700, height=500, returned_objects=[])

                        # Station data (Step 10.8)
                        st.subheader("Nearby Stations")
                        display_cols = ['title', 'distance_m', 'operational_status', 'available_connectors', 'total_connectors', 'price', 'supported_connectors', 'network_provider']
                        existing_cols = [col for col in display_cols if col in stations_df_display.columns]
                        st.dataframe(stations_df_display[existing_cols].style.format({'distance_m': '{:.0f} m', 'price': '{:.2f}'}))

                else:
                    st.warning("No stations found. Try increasing the radius.")
            except Exception as e:
                st.error(f"Error fetching stations: {e}")
                logging.error(f"Station fetch error: {e}", exc_info=True)

    st.markdown("---")
    st.info("ℹ️ Real-time data from HERE Maps API. Visit 'Demand Prediction' for forecasts or 'Route Optimization' for navigation.")

# Demand Prediction (Step 11)
def show_demand_prediction():
    st.header("📊 Demand Prediction")
    st.write("Predict charging station demand and get a personalized station recommendation.")

    latitude, longitude = get_user_location()
    if latitude is None or longitude is None:
        st.error("Please provide a valid location to continue.")
        return
    st.info(f"Current location: {st.session_state['user_place_name']} ({latitude:.6f}, {longitude:.6f})")

    if not demand_model:
        st.error("Model not available.")
        return

    # Input form (Step 11.2)
    st.subheader("Station Details")
    input_data = {}
    input_data['latitude'] = latitude
    input_data['longitude'] = longitude

    categorical_cols = [col for col in label_encoders.keys() if col != 'geohash']
    for col in categorical_cols:
        try:
            options = list(label_encoders[col].classes_)
        except AttributeError:
            options = ['CCS', 'Type 2', 'CHAdeMO'] if col == 'supported_connectors' else ['Unknown']
        input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}:", options, index=0)

    numerical_features = ['Availability', 'Wait Time', 'Power_kW', 'num_charging_points', 'temperature', 'humidity', 'jam_factor']
    defaults = {'Availability': 5, 'Wait Time': 10.0, 'Power_kW': 150.0, 'num_charging_points': 10, 'temperature': 20.0, 'humidity': 60.0, 'jam_factor': 3.0}
    for feature in numerical_features:
        if feature in ['Availability', 'num_charging_points']:
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", value=int(defaults.get(feature, 0)), step=1)
        else:
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", value=float(defaults.get(feature, 0.0)), format="%.2f")

    # Review system (Step 11.7)
    st.subheader("Review System")
    station_id = st.text_input("Station ID for Review:", "")
    rating = None
    if station_id:
        try:
            usage_df = fetch_latest_data_from_duckdb('usage_events', entity_ids=[station_id])
            if not usage_df.empty and any(usage_df['event_type'] == 'charging'):
                cols = st.columns(5)
                for i in range(1, 6):
                    if cols[i-1].button(f"⭐ {i}", key=f'rating_{i}'):
                        rating = i
                        st.success(f"Submitted rating: {i} for station {station_id}")
            else:
                st.info("No recent charging event found for this station.")
        except Exception as e:
            st.error(f"Review system error: {e}")
            logging.error(f"Review system error: {e}", exc_info=True)

    # Predict demand and get recommendation (Step 11.3–11.6, Requirement 2)
    if st.button("Predict Demand & Recommend Station"):
        with st.spinner("Fetching data and generating recommendation..."):
            try:
                # Fetch nearby stations
                radius = 5000  # Default radius for recommendation
                cache_key = f"stations_{latitude}_{longitude}_{radius}"
                if cache_key in cache:
                    stations_df = cache[cache_key]
                else:
                    stations_df = fetch_latest_data_from_duckdb('station_status', lat=latitude, lon=longitude, radius=radius)
                    if stations_df.empty:
                        stations_df = fetch_real_time_station_data(latitude, longitude, radius)
                    cache[cache_key] = stations_df

                if stations_df.empty:
                    st.warning("No stations found. Try adjusting the search radius.")
                    return

                # Filter by user-selected connector type
                connector_type = input_data.get('supported_connectors', 'CCS')
                filtered_df = stations_df[stations_df['supported_connectors'] == connector_type].sort_values(by='distance_m')
                if filtered_df.empty:
                    st.warning(f"No stations found with connector type {connector_type}.")
                    return

                # Get real-time traffic and weather
                traffic_df = fetch_latest_data_from_duckdb('traffic_conditions', lat=latitude, lon=longitude)
                weather_df = fetch_latest_data_from_duckdb('weather_info', lat=latitude, lon=longitude)

                # Get route to closest station
                top_station = filtered_df.iloc[0]
                route = fetch_optimal_route_summary(latitude, longitude, top_station['lat'], top_station['lon'], connector_type=connector_type, power_rating=input_data.get('Power_kW', 150.0))

                # Predict demand for top station
                prediction_input = {
                    'latitude': top_station['lat'],
                    'longitude': top_station['lon'],
                    'supported_connectors': top_station['supported_connectors'],
                    'network_provider': top_station.get('network_provider', 'Unknown'),
                    'Availability': top_station.get('available_connectors', 5),
                    'Wait Time': top_station.get('wait_time', 10.0),
                    'Power_kW': input_data.get('Power_kW', 150.0),
                    'num_charging_points': top_station.get('total_connectors', 10),
                    'temperature': weather_df['temperature'].iloc[0] if not weather_df.empty else 20.0,
                    'humidity': weather_df['humidity'].iloc[0] if not weather_df.empty else 60.0,
                    'jam_factor': traffic_df['jam_factor'].iloc[0] if not traffic_df.empty else 3.0
                }
                processed_input = preprocess_input_for_prediction(prediction_input, label_encoders, scaler, feature_names)
                prediction = max(0, int(round(demand_model.predict(processed_input)[0])))

                # Decode categorical features for LLM
                decoded_inputs = decode_categorical_features(prediction_input, label_encoders)
                factors = [f"{k.replace('_', ' ').title()}: {v}" for k, v in decoded_inputs.items() if v != "N/A"]
                factors.extend([f"{k.replace('_', ' ').title()}: {prediction_input[k]:.1f}" for k in numerical_features if k in prediction_input])

                # Prepare LLM data
                llm_data = {
                    'predicted_demand': prediction,
                    'prediction_factors': ", ".join(factors),
                    'user_context': f"User at ({latitude:.4f}, {longitude:.4f}) preferring {connector_type}",
                    'station_data': f"{top_station['title']}: {top_station['supported_connectors']}, {top_station.get('available_connectors', 5)} available, {top_station.get('price', 0):.2f} USD",
                    'traffic_data': f"Jam factor: {traffic_df['jam_factor'].iloc[0] if not traffic_df.empty else 3.0}",
                    'weather_data': f"Temp: {weather_df['temperature'].iloc[0] if not weather_df.empty else 20.0}°C, Humidity: {weather_df['humidity'].iloc[0] if not weather_df.empty else 60.0}%",
                    'route_summary': route['text'] if route else "No route available"
                }
                explanation = get_llm_suggestion(DEMAND_PREDICTION_EXPLANATION_TEMPLATE, llm_data)
                st.success(f"🔮 Predicted Demand for {top_station['title']}: {prediction} EVs")
                st.info(f"**Recommendation**: {explanation}")
            except Exception as e:
                st.error(f"Prediction or recommendation failed: {e}")
                logging.error(f"Prediction error: {e}", exc_info=True)

# Geospatial Analysis (Step 12)
import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
import os
import logging
from streamlit_folium import st_folium

# Assuming these are defined elsewhere
VORONOI_OUTPUT_PATH = "path/to/voronoi.geojson"
GRID_ANALYSIS_OUTPUT_PATH = "path/to/grid_analysis.geojson"
GEOSPATIAL_INSIGHTS_PROMPT_TEMPLATE = "Provide insights for: {analysis_summary}\nContext: {user_context}"

# Cache expensive data loading operations
@st.cache_data
def load_geospatial_data(latitude, longitude):
    """Load Voronoi, grid analysis, and station data."""
    voronoi_gdf = gpd.GeoDataFrame()
    grid_analysis_gdf = gpd.GeoDataFrame()
    station_gdf = gpd.GeoDataFrame()

    # Load Voronoi data
    if os.path.exists(VORONOI_OUTPUT_PATH):
        try:
            voronoi_gdf = gpd.read_file(VORONOI_OUTPUT_PATH)
        except Exception as e:
            logging.warning(f"Failed to load Voronoi data: {e}")
    else:
        logging.warning(f"Voronoi file not found at {VORONOI_OUTPUT_PATH}")

    # Load grid analysis data
    if os.path.exists(GRID_ANALYSIS_OUTPUT_PATH):
        try:
            grid_analysis_gdf = gpd.read_file(GRID_ANALYSIS_OUTPUT_PATH)
        except Exception as e:
            logging.warning(f"Failed to load grid analysis data: {e}")
    else:
        logging.warning(f"Grid analysis file not found at {GRID_ANALYSIS_OUTPUT_PATH}")

    # Fetch station locations
    try:
        station_gdf = fetch_station_locations_from_duckdb(latitude, longitude)
    except Exception as e:
        logging.warning(f"Failed to fetch station locations: {e}")

    # Run geospatial analysis if data is missing
    if voronoi_gdf.empty or grid_analysis_gdf.empty:
        try:
            run_geospatial_analysis(latitude, longitude)
            if os.path.exists(VORONOI_OUTPUT_PATH):
                voronoi_gdf = gpd.read_file(VORONOI_OUTPUT_PATH)
            if os.path.exists(GRID_ANALYSIS_OUTPUT_PATH):
                grid_analysis_gdf = gpd.read_file(GRID_ANALYSIS_OUTPUT_PATH)
        except Exception as e:
            logging.error(f"Geospatial analysis failed: {e}")

    return voronoi_gdf, grid_analysis_gdf, station_gdf

def create_folium_map(latitude, longitude, voronoi_gdf, grid_analysis_gdf, station_gdf):
    """Create and configure the Folium map."""
    # Center map dynamically
    center_lat, center_lon = latitude, longitude
    if not station_gdf.empty and 'geometry' in station_gdf.columns:
        try:
            minx, miny, maxx, maxy = station_gdf.total_bounds
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2
        except Exception as e:
            logging.warning(f"Failed to compute bounds: {e}")

    # Initialize map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add Voronoi layer
    if not voronoi_gdf.empty and 'geometry' in voronoi_gdf.columns:
        fields = [col for col in ['station_id', 'distance_m'] if col in voronoi_gdf.columns]
        try:
            folium.GeoJson(
                voronoi_gdf,
                name="Voronoi",
                tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None
            ).add_to(m)
        except Exception as e:
            logging.warning(f"Failed to add Voronoi layer: {e}")

    # Add grid analysis layer
    if not grid_analysis_gdf.empty and 'station_count' in grid_analysis_gdf.columns and 'geometry' in grid_analysis_gdf.columns:
        try:
            grid_analysis_gdf['index'] = grid_analysis_gdf.index
            folium.Choropleth(
                geo_data=grid_analysis_gdf,
                data=grid_analysis_gdf,
                columns=['index', 'station_count'],
                key_on='feature.properties.index',
                fill_color='YlOrRd',
                legend_name='Station Count',
                line_weight=0.5
            ).add_to(m)
            folium.GeoJson(
                grid_analysis_gdf,
                tooltip=folium.GeoJsonTooltip(fields=['station_count'])
            ).add_to(m)
        except Exception as e:
            logging.warning(f"Failed to add grid analysis layer: {e}")

    # Add station markers
    if not station_gdf.empty and 'lat' in station_gdf.columns and 'lon' in station_gdf.columns:
        for _, row in station_gdf.iterrows():
            if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                try:
                    popup = f"<b>{row.get('title', 'N/A')}</b><br>Status: {row.get('operational_status', 'N/A')}<br>Connectors: {row.get('available_connectors', '?')}/{row.get('total_connectors', '?')}<br>Network: {row.get('network_provider', 'N/A')}"
                    folium.Marker(
                        [row['lat'], row['lon']],
                        popup=popup,
                        tooltip=row.get('title', 'N/A')
                    ).add_to(m)
                except Exception as e:
                    logging.warning(f"Failed to add marker for station {row.get('title', 'N/A')}: {e}")

    # Add user location marker
    folium.Marker(
        [latitude, longitude],
        popup="Your Location",
        icon=folium.Icon(color='green')
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    return m

def show_geospatial_analysis():
    st.header("🌍 Geospatial Analysis")
    st.write("Visualize EV charging station coverage with Voronoi and grid analysis.")

    # Get user location
    latitude, longitude = get_user_location()
    if latitude is None or longitude is None or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        st.error("Invalid location. Using default location (Chennai).")
        latitude, longitude = 13.0827, 80.2707
    st.info(f"Current location: {st.session_state.get('user_place_name', 'Unknown')} ({latitude:.6f}, {longitude:.6f})")

    try:
        # Load data with caching
        with st.spinner("Loading geospatial data..."):
            voronoi_gdf, grid_analysis_gdf, station_gdf = load_geospatial_data(latitude, longitude)

        # Display warning if no stations found
        if station_gdf.empty:
            st.warning(f"No charging stations found within 50km of your location ({latitude:.4f}, {longitude:.4f}).")
        else:
            st.success(f"✅ Loaded {len(station_gdf)} stations near your location.")

        # Log GeoDataFrame columns for debugging
        logging.info(f"voronoi_gdf columns: {voronoi_gdf.columns.tolist()}")
        logging.info(f"grid_analysis_gdf columns: {grid_analysis_gdf.columns.tolist()}")
        logging.info(f"station_gdf columns: {station_gdf.columns.tolist()}")

        # Create and render map
        with st.spinner("Rendering map..."):
            m = create_folium_map(latitude, longitude, voronoi_gdf, grid_analysis_gdf, station_gdf)
            st_folium(m, width=700, height=500, key="folium_map")

        # Filter stations
        st.subheader("Filter Stations")
        connector_options = ["All"] + list(station_gdf['supported_connectors'].unique()) if not station_gdf.empty and 'supported_connectors' in station_gdf.columns else ["All"]
        connector_filter = st.selectbox("Connector Type:", connector_options, key="connector_filter")

        # Apply filter only for insights, not map rendering
        filtered_stations = station_gdf[station_gdf['supported_connectors'] == connector_filter] if connector_filter != "All" else station_gdf

        # Display geospatial insights
        st.subheader("Geospatial Insights")
        summary = []
        if not grid_analysis_gdf.empty and 'station_count' in grid_analysis_gdf.columns:
            top_cells = grid_analysis_gdf.sort_values(by='station_count', ascending=False).head(5)
            for _, row in top_cells.iterrows():
                try:
                    centroid = row.geometry.centroid
                    summary.append(f"Cell at ({centroid.y:.4f}, {centroid.x:.4f}): {row['station_count']} stations")
                except Exception as e:
                    logging.warning(f"Failed to compute centroid: {e}")
        if not filtered_stations.empty:
            network_counts = filtered_stations['network_provider'].value_counts().head(5)
            summary.extend([f"{network}: {count} stations" for network, count in network_counts.items()])

        llm_data = {
            'analysis_summary': "\n".join(summary) or "No analysis available",
            'user_context': f"User at ({latitude:.4f}, {longitude:.4f})"
        }
        with st.spinner("Generating insights..."):
            insights = get_llm_suggestion(GEOSPATIAL_INSIGHTS_PROMPT_TEMPLATE, llm_data)
        st.info(insights)

    except Exception as e:
        logging.error(f"Unexpected error in geospatial analysis: {e}", exc_info=True)
        st.error(f"Failed to load map: {e}")
        # Fallback map
        m = folium.Map(location=[latitude, longitude], zoom_start=10)
        folium.Marker(
            [latitude, longitude],
            popup="Your Location",
            icon=folium.Icon(color='green')
        ).add_to(m)
        st_folium(m, width=700, height=500, key="folium_map_fallback")
        st.warning("Displaying fallback map for your location.")

# Route Optimization (Step 13)
def show_route_optimization():
    st.header("🗺️ Route Optimization")
    st.write("Find the best route to a charging station.")

    latitude, longitude = get_user_location()
    if latitude is None or longitude is None:
        st.error("Please provide a valid location to continue.")
        return
    st.info(f"Origin: ({latitude:.6f}, {longitude:.6f})")

    st.subheader("Select Destination Station")
    try:
        stations_df = fetch_latest_data_from_duckdb('station_status', lat=latitude, lon=longitude, radius=10000)
        if stations_df.empty:
            stations_df = fetch_real_time_station_data(latitude, longitude, 10000)
        station_options = {f"{row['title']} ({row['supported_connectors']} - {row['distance_m']:.0f}m)": (row['lat'], row['lon'], row['station_id']) for _, row in stations_df.iterrows()}
        selected_station = st.selectbox("Destination Station:", list(station_options.keys()))
        dest_lat, dest_lon, station_id = station_options.get(selected_station, (None, None, None))
    except Exception as e:
        st.error(f"Error fetching stations: {e}")
        logging.error(f"Station fetch error: {e}", exc_info=True)
        return

    connector_type = st.selectbox("EV Connector Type:", ['CCS', 'Type 2', 'CHAdeMO'])
    power_rating = st.number_input("Power Rating (kW):", value=150.0, min_value=0.0)

    if st.button("Get Route"):
        if dest_lat and dest_lon:
            with st.spinner("Calculating route..."):
                try:
                    route = fetch_optimal_route_summary(latitude, longitude, dest_lat, dest_lon, connector_type=connector_type, power_rating=power_rating)
                    if route:
                        st.success(f"✅ Route: {route['text']}")
                        geometry = fetch_route_geometry(latitude, longitude, dest_lat, dest_lon)
                        m = folium.Map(location=[(latitude + dest_lat) / 2, (longitude + dest_lon) / 2], zoom_start=10)
                        folium.Marker([latitude, longitude], popup="Origin", icon=folium.Icon(color='green')).add_to(m)
                        folium.Marker([dest_lat, dest_lon], popup="Destination", icon=folium.Icon(color='red')).add_to(m)
                        if geometry:
                            folium.PolyLine(geometry, color="blue", weight=2.5, opacity=0.8).add_to(m)
                        folium_static(m, width=700, height=500)

                        station_data = stations_df[stations_df['station_id'] == station_id].iloc[0]
                        input_data = {
                            'latitude': dest_lat, 'longitude': dest_lon, 'supported_connectors': connector_type,
                            'network_provider': station_data.get('network_provider', 'Unknown'),
                            'Availability': station_data.get('available_connectors', 5),
                            'Wait Time': station_data.get('wait_time', 10.0),
                            'Power_kW': power_rating,
                            'num_charging_points': station_data.get('total_connectors', 10),
                            'temperature': 20.0, 'humidity': 60.0,
                            'jam_factor': route.get('traffic_delay_min', 3.0)
                        }
                        processed_input = preprocess_input_for_prediction(input_data, label_encoders, scaler, feature_names)
                        prediction = max(0, int(round(demand_model.predict(processed_input)[0])))

                        llm_data = {
                            'route_summary': route['text'],
                            'traffic_conditions': f"Traffic delay: {route['traffic_delay_min']:.1f} min",
                            'destination_station_details': f"{selected_station}, {connector_type}, {power_rating}kW",
                            'predicted_demand_at_arrival': f"{prediction} EVs",
                            'user_context': f"User traveling from ({latitude:.4f}, {longitude:.4f}) to ({dest_lat:.4f}, {dest_lon:.4f})"
                        }
                        advice = get_llm_suggestion(ROUTE_ADVICE_PROMPT_TEMPLATE, llm_data)
                        st.info(advice)
                    else:
                        st.error("Could not find route.")
                except Exception as e:
                    st.error(f"Route calculation failed: {e}")
                    logging.error(f"Route error: {e}", exc_info=True)
        else:
            st.error("Select a valid destination.")

# Real-Time Data & Simulation (Step 14)
def show_simulation():
    st.header("📡 Real-Time Data & Simulation")
    st.write("Simulate and visualize charging station data, with fallback for API unavailability.")

    latitude, longitude = get_user_location()
    if latitude is None or longitude is None:
        st.error("Please provide a valid location to continue.")
        return
    st.info(f"Current location: {st.session_state['user_place_name']} ({latitude:.6f}, {longitude:.6f})")

    try:
        import duckdb
        conn = duckdb.connect(':memory:')
        conn.execute("CREATE TABLE simulation_data (station_id VARCHAR, title VARCHAR, lat DOUBLE, lon DOUBLE, available_connectors INTEGER, total_connectors INTEGER, supported_connectors VARCHAR, network_provider VARCHAR, timestamp VARCHAR)")
    except Exception as e:
        st.error(f"Failed to initialize DuckDB: {e}")
        logging.error(f"DuckDB init error: {e}", exc_info=True)
        return

    if st.button("Start Simulation"):
        st.session_state['simulation_running'] = True
    if st.button("Stop Simulation"):
        st.session_state['simulation_running'] = False

    if st.session_state.get('simulation_running', False):
        with st.spinner("Running simulation..."):
            try:
                # Fetch real-time data
                real_data = fetch_latest_data_from_duckdb('station_status', lat=latitude, lon=longitude, radius=5000)
                if real_data.empty:
                    real_data = fetch_real_time_station_data(latitude, longitude, 5000)

                # Generate simulated data
                simulated_data = pd.DataFrame([
                    {
                        'station_id': f"SIM_{i}",
                        'title': f"Simulated Station {i}",
                        'lat': latitude + np.random.uniform(-0.01, 0.01),
                        'lon': longitude + np.random.uniform(-0.01, 0.01),
                        'available_connectors': np.random.randint(1, 10),
                        'total_connectors': 10,
                        'supported_connectors': np.random.choice(['CCS', 'Type 2', 'CHAdeMO']),
                        'network_provider': np.random.choice(['Electrify America', 'ChargePoint']),
                        'timestamp': datetime.datetime.now().isoformat()
                    } for i in range(5)
                ])

                # Store in DuckDB
                conn.register('temp_sim_data', simulated_data)
                conn.execute("INSERT INTO simulation_data SELECT * FROM temp_sim_data")
                combined_data = conn.execute("SELECT * FROM simulation_data").fetchdf()

                # Combine with real data
                if not real_data.empty:
                    combined_data = pd.concat([combined_data, real_data], ignore_index=True)

                st.dataframe(combined_data[['title', 'available_connectors', 'supported_connectors', 'network_provider', 'timestamp']])

                # Visualizations
                fig = px.scatter(
                    combined_data,
                    x='lon',
                    y='lat',
                    color='available_connectors',
                    size='total_connectors',
                    hover_data=['title', 'supported_connectors', 'network_provider'],
                    title="Station Locations and Availability",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig)

                if 'available_connectors' in combined_data.columns:
                    fig = px.line(
                        combined_data,
                        x='timestamp',
                        y='available_connectors',
                        color='title',
                        title="Simulated vs Real-Time Connector Availability"
                    )
                    st.plotly_chart(fig)

                st.info("Simulation data is stored in DuckDB and combined with real-time data when available.")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                logging.error(f"Simulation error: {e}", exc_info=True)

# Data & Insights Panel (Step 16)
def show_data_display():
    st.header("📂 Data & Insights Panel")
    st.write("Explore charging station data and trends with interactive visualizations.")

    latitude, longitude = get_user_location()
    if latitude is None or longitude is None:
        st.error("Please provide a valid location to continue.")
        return
    st.info(f"Current location: {st.session_state['user_place_name']} ({latitude:.6f}, {longitude:.6f})")

    table = st.selectbox("Data Table:", ['station_status', 'traffic_conditions', 'weather_info'])
    connector_filter = st.selectbox("Connector Type:", ["All", "CCS", "Type 2", "CHAdeMO"])
    
    try:
        df = fetch_latest_data_from_duckdb(table)
        
        # Apply connector filter only for station_status table and if supported_connectors exists
        if table == 'station_status' and connector_filter != "All" and 'supported_connectors' in df.columns:
            df = df[df['supported_connectors'] == connector_filter]
        
        # Display the DataFrame
        if df.empty:
            st.warning("No data available for the selected table and filters.")
            return
        st.dataframe(df)

        # Interactive visualizations based on selected table
        if table == 'station_status' and 'available_connectors' in df.columns and 'lon' in df.columns and 'lat' in df.columns:
            # Bar plot for station availability
            fig = px.bar(
                df,
                x='title',
                y='available_connectors',
                color='operational_status',
                title="Station Availability by Status",
                hover_data=['supported_connectors', 'network_provider', 'price'],
                color_discrete_map={'operational': 'green', 'unknown': 'gray'}
            )
            fig.update_traces(marker_line_width=1.5, opacity=0.8)
            st.plotly_chart(fig)

            # Heatmap for station density
            fig = px.density_heatmap(
                df,
                x='lon',
                y='lat',
                z='available_connectors',
                title="Station Density Heatmap",
                hover_data=['title', 'supported_connectors'],
                color_continuous_scale='Blues'
            )
            fig.add_scatter(x=[longitude], y=[latitude], mode='markers', marker=dict(size=10, color='red'), name='Your Location')
            st.plotly_chart(fig)

            # Scatter plot for stations by network provider
            if 'network_provider' in df.columns:
                fig = px.scatter(
                    df,
                    x='lon',
                    y='lat',
                    color='network_provider',
                    size='available_connectors',
                    hover_data=['title', 'supported_connectors', 'price'],
                    title="Stations by Network Provider",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.add_scatter(x=[longitude], y=[latitude], mode='markers', marker=dict(size=10, color='red'), name='Your Location')
                st.plotly_chart(fig)
            else:
                st.warning("Column 'network_provider' not found in the data. Skipping scatter plot.")

        elif table == 'traffic_conditions' and 'jam_factor' in df.columns:
            fig = px.line(
                df,
                x='fetch_timestamp',  # Changed from 'timestamp' to 'fetch_timestamp'
                y='jam_factor',
                title="Traffic Congestion Over Time"
            )
            st.plotly_chart(fig)
        elif table == 'weather_info' and 'temperature' in df.columns:
            fig = px.scatter(
                df,
                x='fetch_timestamp',  # Changed from 'timestamp' to 'fetch_timestamp'
                y='temperature',
                size='humidity',
                title="Weather Trends",
                hover_data=['humidity']
            )
            st.plotly_chart(fig)
        else:
            st.warning("Required columns for visualization are missing in the selected table.")

    except Exception as e:
        st.error(f"Data display failed: {e}")
        logging.error(f"Data display error: {e}", exc_info=True)
# Route to selected page (Step 9.3)
page_functions = {
    "dashboard": show_real_time_dashboard,
    "prediction": show_demand_prediction,
    "geospatial": show_geospatial_analysis,
    "route_optimization": show_route_optimization,
    "simulation": show_simulation,
    "data_display": show_data_display
}
page_functions[st.session_state['current_page']]()

# --- Data Storage Integration (Step 3.3) ---
import streamlit as st
from utils.store_utils import store_user_data_for_api_call, populate_traffic_conditions, populate_weather_info
from utils.api_utils import geocode_place, fetch_real_time_station_data, fetch_optimal_route_summary, fetch_route_geometry
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

# Wrapper function to store API and simulated data
def store_api_and_user_data(api_function: str, result: Any, place_name: str = None, **kwargs) -> Any:
    """Stores API call results and simulated data, returning the original result."""
    try:
        # Get user location from session state (for future extensibility)
        lat = st.session_state.get('user_lat', 9.0548)  # Default to Surandai
        lon = st.session_state.get('user_lon', 77.4335)
        timestamp = datetime.datetime.utcnow()

        # Prepare response data
        response_data = result
        if api_function == "geocode_place" and place_name and result:
            response_data = {"place": place_name, "coordinates": result}
        elif api_function == "fetch_latest_data_from_duckdb":
            # Handle simulated data for traffic_conditions and weather_info
            table = kwargs.get('table')
            if table == 'traffic_conditions' and isinstance(result, pd.DataFrame):
                populate_traffic_conditions(result, timestamp)
            elif table == 'weather_info' and isinstance(result, pd.DataFrame):
                populate_weather_info(result, timestamp)

        # Store data in api_call_logs if result is not None/empty
        if response_data is not None and (not isinstance(response_data, pd.DataFrame) or not response_data.empty):
            store_user_data_for_api_call(
                lat=lat,
                lon=lon,
                api_function=api_function,
                response_data=response_data,
                timestamp=timestamp
            )

        return result
    except Exception as e:
        logging.error(f"Error storing data for {api_function}: {e}")
        return result

# Store original functions
original_geocode_place = geocode_place
original_fetch_real_time_station_data = fetch_real_time_station_data
original_fetch_optimal_route_summary = fetch_optimal_route_summary
original_fetch_route_geometry = fetch_route_geometry
original_fetch_latest_data_from_duckdb = fetch_latest_data_from_duckdb

# Wrapped API functions to store data
def wrapped_geocode_place(place_name: str) -> Optional[Tuple[float, float]]:
    result = original_geocode_place(place_name)
    return store_api_and_user_data("geocode_place", result, place_name=place_name)

def wrapped_fetch_real_time_station_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
    result = original_fetch_real_time_station_data(lat, lon, radius)
    return store_api_and_user_data("fetch_real_time_station_data", result)

def wrapped_fetch_optimal_route_summary(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, **kwargs) -> Optional[Dict[str, Any]]:
    result = original_fetch_optimal_route_summary(origin_lat, origin_lon, dest_lat, dest_lon, **kwargs)
    return store_api_and_user_data("fetch_optimal_route_summary", result)

def wrapped_fetch_route_geometry(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> Optional[List[Tuple[float, float]]]:
    result = original_fetch_route_geometry(origin_lat, origin_lon, dest_lat, dest_lon)
    return store_api_and_user_data("fetch_route_geometry", result)

def wrapped_fetch_latest_data_from_duckdb(table: str, lat: float = None, lon: float = None, radius: float = None, entity_ids: List[str] = None) -> pd.DataFrame:
    result = original_fetch_latest_data_from_duckdb(table, lat, lon, radius, entity_ids)
    return store_api_and_user_data("fetch_latest_data_from_duckdb", result, table=table)

# Override functions with wrapped versions
geocode_place = wrapped_geocode_place
fetch_real_time_station_data = wrapped_fetch_real_time_station_data
fetch_optimal_route_summary = wrapped_fetch_optimal_route_summary
fetch_route_geometry = wrapped_fetch_route_geometry
fetch_latest_data_from_duckdb = wrapped_fetch_latest_data_from_duckdb
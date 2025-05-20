# File: D:\Desktop\ev_charging_optimization\utils\llm_data_prep.py

import duckdb
import os
import pandas as pd # Using pandas DataFrames to handle query results
import numpy as np
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import re # Import regex for basic sanitization

# Setup logging
# Change level to logging.DEBUG for more detailed output if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# DuckDB connection path - should match the path used in fetch_real_time_data.py
DB_PATH = 'data/ev_charging.duckdb'

def get_duckdb_connection():
    """Gets a connection to the DuckDB database."""
    conn = None # Initialize conn to None outside the try block
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        # Use read_only=True for safety if only querying
        conn = duckdb.connect(database=DB_PATH, read_only=True)
        logging.debug("Successfully connected to DuckDB.")
        return conn # Return the connection if successful
    except Exception as e:
        logging.error(f"Error connecting to DuckDB at {DB_PATH}: {e}")
        conn = None # Ensure conn is None on error
    # No finally needed here as the caller's finally handles closing


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


def fetch_latest_data_from_duckdb(table_name: str, entity_ids: Optional[List[str]] = None, lat: Optional[float] = None, lon: Optional[float] = None, radius: int = 5000) -> pd.DataFrame:
    """
    Fetches the most recent data from a given DuckDB table.
    Can filter by a list of entity IDs (e.g., station IDs) or by location/radius.
    Returns a pandas DataFrame. Assumes 'fetch_timestamp' column exists.
    Assumes 'lat', 'lon' columns exist for location filtering in station, traffic, weather tables.
    """
    conn = get_duckdb_connection()
    if conn is None:
        return pd.DataFrame() # Return empty DataFrame on connection error

    df = pd.DataFrame() # Initialize df

    try:
        # Find the latest fetch timestamp for the table
        # Using DuckDB's SQL query for the max timestamp
        latest_timestamp_query = f"SELECT MAX(fetch_timestamp) FROM {table_name}"
        latest_timestamp_result = conn.execute(latest_timestamp_query).fetchone()
        latest_timestamp = latest_timestamp_result[0]

        if latest_timestamp is None:
            logging.warning(f"No data found in table '{table_name}'. Is Step 3 running correctly?")
            return pd.DataFrame()

        logging.debug(f"Fetching data from '{table_name}' for latest timestamp: {latest_timestamp}")

        # Base query filtering by the latest timestamp
        # Select all columns for flexibility in formatting
        query = f"SELECT * FROM {table_name} WHERE fetch_timestamp = ?"
        params = [latest_timestamp]

        # Add filtering by entity IDs if provided
        if entity_ids:
            # Need to identify the ID column name based on the table
            id_col = None
            if table_name == 'station_status':
                id_col = 'api_station_id'
            elif table_name == 'traffic_conditions':
                id_col = 'road_segment_id'
            elif table_name == 'weather_info':
                # Weather typically doesn't have a specific entity ID for filtering multiple obs points
                pass # No ID filtering for weather table

            if id_col and entity_ids: # Ensure id_col is found and list is not empty
                 # FIX (SyntaxError): Build IN clause string more robustly
                 # Escape single quotes by replacing "'" with "''"
                 # Wrap each escaped ID in single quotes for SQL
                 # This avoids backslashes in the f-string expression part
                 quoted_ids = []
                 for id in entity_ids:
                     safe_id = str(id).replace("'", "''") # Escape single quotes
                     quoted_ids.append(f"'{safe_id}'") # Wrap in single quotes using f-string

                 ids_string = ",".join(quoted_ids)

                 if ids_string: # Ensure the resulting string is not empty
                    query += f" AND {id_col} IN ({ids_string})"
                    # No need to append entity_ids to params anymore as they are in the query string
                    logging.debug(f"Filtering '{table_name}' by {len(entity_ids)} {id_col}s using IN clause.")
                 else:
                     logging.warning(f"Empty list of entity_ids provided for table '{table_name}'. No ID filter applied.")

            else:
                 if table_name not in ['traffic_conditions', 'weather_info']: # Only warn for tables expected to have IDs
                     logging.warning(f"Cannot filter table '{table_name}' by entity IDs as ID column is unknown, not applicable, or entity_ids list is empty.")


        # Execute the query and fetch into a DataFrame
        # Use fetchdf() for direct conversion to pandas DataFrame
        df = conn.execute(query, params).fetchdf()

        # Now, if location filter was requested AND Lat/Lon columns exist, filter the DataFrame in Pandas
        # This is a spatial filter applied *after* fetching the latest batch
        if lat is not None and lon is not None and 'lat' in df.columns and 'lon' in df.columns:
            logging.debug(f"Applying location filter ({lat},{lon} radius {radius}m) in Pandas for {table_name}...")
            # Calculate distance in Pandas using Haversine formula (vectorized for performance)
            R = 6371000 # Earth radius in meters
            # Ensure lat/lon are numeric, coercing errors will turn non-numeric into NaN
            df['lat_numeric'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon_numeric'] = pd.to_numeric(df['lon'], errors='coerce')

            # Drop rows where lat/lon became NaN after coercion for distance calculation
            df_filtered_loc = df.dropna(subset=['lat_numeric', 'lon_numeric']).copy()

            if not df_filtered_loc.empty:
                df_filtered_loc['lat_rad'] = np.radians(df_filtered_loc['lat_numeric'])
                df_filtered_loc['lon_rad'] = np.radians(df_filtered_loc['lon_numeric'])
                user_lat_rad = np.radians(lat)
                user_lon_rad = np.radians(lon)

                dlon = user_lon_rad - df_filtered_loc['lon_rad']
                dlat = user_lat_rad - df_filtered_loc['lat_rad']

                a = np.sin(dlat / 2)**2 + np.cos(user_lat_rad) * np.cos(df_filtered_loc['lat_rad']) * np.sin(dlon / 2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                df_filtered_loc['distance_m'] = R * c # Distance in meters

                # Filter by the specified radius
                df = df_filtered_loc[df_filtered_loc['distance_m'] <= radius].copy() # Use copy() after filtering
                logging.debug(f"Filtered '{table_name}' down to {len(df)} items within {radius}m radius.")

                # Optional: drop intermediate columns used for calculation
                df = df.drop(columns=['lat_numeric', 'lon_numeric', 'lat_rad', 'lon_rad'], errors='ignore')

            else:
                logging.warning(f"No valid Lat/Lon found in {table_name} for location filtering after coercing to numeric.")
                df = pd.DataFrame() # Return empty if no valid numeric lat/lon were found


        logging.info(f"Fetched {len(df)} latest records from '{table_name}' after all filters.")

        return df # Return the filtered DataFrame


    except duckdb.Error as e:
        logging.error(f"DuckDB error fetching data from '{table_name}': {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching data from '{table_name}': {e}")
        return pd.DataFrame()
    finally:
        # Ensure the connection is closed even if errors occur
        if conn:
            try:
                 conn.close()
                 logging.debug("DuckDB connection closed.")
            except Exception as close_e:
                 logging.error(f"Error closing DuckDB connection: {close_e}")


def format_data_for_llm(station_df: pd.DataFrame, traffic_df: pd.DataFrame, weather_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Formats the fetched real-time data into a concise structure for LLM prompting.
    Expects dataframes with original string values for categorical fields
    as stored in DuckDB by Step 3.
    """
    formatted_data = {}
    logging.debug("Formatting data for LLM...")

    # --- Format Station Data ---
    formatted_stations = []
    if not station_df.empty:
        # Sort stations by distance if distance column exists for better presentation in prompts
        if 'distance_m' in station_df.columns:
            station_df = station_df.sort_values(by='distance_m')
        # Limit the number of stations included in the prompt to avoid excessive length
        station_df_limited = station_df.head(10) # Limit to top 10 stations

        for _, row in station_df_limited.iterrows():
            # Use .get() with default values for robustness against missing columns/data
            # Data from DuckDB should have correct types, but defensive access is good
            available = row.get('available_connectors', 0)
            total = row.get('total_connectors', 0)
            # Ensure status, network, types are strings - they should be from Step 3 storage
            status = str(row.get('operational_status', 'Unknown Status'))
            price = row.get('price', 'Unknown Price') # Price might be JSON string, keep as is or simplify? Simplify for conciseness.
            # If price is JSON, try to extract a simple representation for the prompt
            price_str = 'Unknown Price' # Default
            if pd.notnull(price):
                try:
                    # Attempt to parse JSON price structure
                    price_data = json.loads(price) if isinstance(price, str) and price.strip().startswith('{') else price
                    if isinstance(price_data, dict) and 'text' in price_data:
                         price_str = str(price_data['text'])
                    elif isinstance(price_data, (int, float)):
                         price_str = f"{price_data:.2f}" # Format numeric price
                    elif isinstance(price_data, str):
                         price_str = price_data # Keep original string if not JSON (e.g., "Free")
                    else:
                        price_str = 'Unknown Format'
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON or other parsing error, just use the raw value as string
                    price_str = str(price)
            else:
                price_str = 'Not Available'


            supported_connectors = str(row.get('supported_connectors', 'Unknown Types'))
            network = str(row.get('network_provider', 'Unknown Network'))
            distance_m = row.get('distance_m', None) # Use None if distance wasn't calculated

            # Construct a concise summary string for each station using an f-string
            # Ensure all variables inside the f-string are properly handled and won't cause issues
            # Correcting potential f-string issues pointed out by Pylance (if any were real)
            station_summary = (
                f"- Station: {str(row.get('title', 'Unknown Station'))} (ID: {str(row.get('api_station_id', 'N/A'))})"
            )
            if distance_m is not None and pd.notnull(distance_m):
                 station_summary += f" ({distance_m:.0f}m away)"
            station_summary += (
                f"\n  Status: {status}, Connectors: {available}/{total} available"
                f"\n  Price: {price_str}, Network: {network}"
            )
            if supported_connectors != 'Unknown Types':
                 station_summary += f", Types: {supported_connectors}"
             # Add more details if needed and available in the DataFrame

            formatted_stations.append(station_summary)

    formatted_data['stations'] = formatted_stations if formatted_stations else ["No nearby station data available."]
    logging.debug(f"Formatted {len(formatted_stations)} stations for LLM.")


    # --- Format Traffic Data ---
    formatted_traffic = []
    if not traffic_df.empty:
        # Traffic data is usually per segment. Summarize or list a few segments.
        # Let's list the few segments with highest jam factor for demo
        # Ensure jam_factor is numeric before sorting, handle potential NaNs
        traffic_df['jam_factor_numeric'] = pd.to_numeric(traffic_df['jam_factor'], errors='coerce').fillna(0.0)
        traffic_df = traffic_df.sort_values(by='jam_factor_numeric', ascending=False).head(3) # Get top 3 most congested segments

        if not traffic_df.empty:
             formatted_traffic.append(f"Summary of nearby traffic (top {len(traffic_df)} most congested segments):")

             for _, row in traffic_df.iterrows():
                 jam_factor = row.get('jam_factor', 0.0) # Use original jam_factor for display
                 speed = row.get('speed', 0.0) # Speed often in m/s from HERE, convert to km/h
                 free_flow = row.get('free_flow', 0.0) # m/s
                 traffic_state = str(row.get('traffic_state', 'unknown'))

                 # Convert speed/free_flow safely to km/h
                 speed_kmh = float(speed) * 3.6 if pd.notnull(speed) and pd.api.types.is_numeric_dtype(type(speed)) else 'N/A'
                 free_flow_kmh = float(free_flow) * 3.6 if pd.notnull(free_flow) and pd.api.types.is_numeric_dtype(type(free_flow)) else 'N/A'

                 # Construct traffic summary using an f-string
                 traffic_summary = f"- Segment (Jam Factor: {jam_factor:.1f}): "
                 if speed_kmh != 'N/A' and free_flow_kmh != 'N/A':
                      traffic_summary += f"Current Speed {speed_kmh:.1f} km/h (Free Flow: {free_flow_kmh:.1f} km/h), "
                 traffic_summary += f"State: {traffic_state}"
                  # Can add location coordinates or segment ID if helpful, but might make prompt too long
                 formatted_traffic.append(traffic_summary)

    formatted_data['traffic'] = formatted_traffic if formatted_traffic else ["No real-time traffic data available for the area."]
    logging.debug(f"Formatted {len(formatted_traffic)} traffic summaries for LLM.")


    # --- Format Weather Data ---
    formatted_weather = []
    if not weather_df.empty:
        # Get the first row for the weather observation (assuming it's the most relevant/closest after filtering)
        weather_obs = weather_df.iloc[0]
        # Ensure types are handled, provide defaults
        temp = weather_obs.get('temperature', 'N/A')
        feels_like = weather_obs.get('temperature_feels_like', 'N/A')
        description = str(weather_obs.get('description', 'Unknown Conditions'))
        wind_speed = weather_obs.get('wind_speed', 'N/A') # Assuming m/s
        humidity = weather_obs.get('humidity', 'N/A') # Assuming percentage

        # Convert temp/feels_like safely to float for formatting
        temp_str = f"{float(temp):.1f}°C" if pd.notnull(temp) and pd.api.types.is_numeric_dtype(type(temp)) else 'N/A °C'
        feels_like_str = f"{float(feels_like):.1f}°C" if pd.notnull(feels_like) and pd.api.types.is_numeric_dtype(type(feels_like)) else 'N/A °C'
        humidity_str = f"{float(humidity):.1f}%" if pd.notnull(humidity) and pd.api.types.is_numeric_dtype(type(humidity)) else 'N/A %'

        # Construct weather summaries using f-strings
        formatted_weather.append(f"Current Weather: {temp_str} (Feels like {feels_like_str})")
        formatted_weather.append(f"Description: {description}")
        if pd.notnull(wind_speed) and pd.api.types.is_numeric_dtype(type(wind_speed)):
             # Convert wind speed from m/s to km/h for better readability
             wind_speed_kmh = float(wind_speed) * 3.6
             formatted_weather.append(f"Wind: {wind_speed_kmh:.1f} km/h, Humidity: {humidity_str}")
        else:
             formatted_weather.append(f"Humidity: {humidity_str}")


    formatted_data['weather'] = formatted_weather if formatted_weather else ["No real-time weather data available for the location."]
    logging.debug(f"Formatted {len(formatted_weather)} weather summaries for LLM.")


    # Combine into a single text string or keep as dictionary
    # Returning as a dictionary is more flexible for building different prompt structures in Step 6
    return formatted_data

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Testing utils/llm_data_prep.py ---")

    # Example usage: Prepare data for a location and a list of dummy station IDs
    # In a real app (Step 10), user_lat/lon would come from user input/geolocation
    # And station_ids would come from the DuckDB query based on radius or user preference
    test_lat = 37.7749 # San Francisco
    test_lon = -122.4194
    test_radius = 5000 # meters

    # --- Option 1: Fetch data based on a radius around the user ---
    # This fetches nearby stations, traffic, and weather within the specified radius
    print(f"\n--- Option 1: Fetching data for LLM prep using location and radius ---")
    print(f"Fetching real-time data for ({test_lat},{test_lon}) within {test_radius}m radius...")

    # Fetch stations by radius
    station_data = fetch_latest_data_from_duckdb('station_status', entity_ids=None, lat=test_lat, lon=test_lon, radius=test_radius)
    print(f"Fetched {len(station_data)} latest station records.")

    # Fetch traffic by radius (traffic has lat/lon, so location filter is applicable)
    traffic_data = fetch_latest_data_from_duckdb('traffic_conditions', entity_ids=None, lat=test_lat, lon=test_lon, radius=test_radius)
    print(f"Fetched {len(traffic_data)} latest traffic records.")

    # Fetch weather for the area (weather has lat/lon)
    # If multiple weather points exist within radius, find the observation point closest to the user
    weather_data_all = fetch_latest_data_from_duckdb('weather_info', entity_ids=None, lat=test_lat, lon=test_lon, radius=test_radius)
    weather_data = pd.DataFrame() # Initialize empty
    if not weather_data_all.empty:
         if 'distance_m' in weather_data_all.columns:
              # Sort by distance and take the first row (closest) as a DataFrame
              weather_data = weather_data_all.sort_values(by='distance_m').head(1)
         else:
              # If distance wasn't calculated (e.g., no Lat/Lon in table?), just take the first row as a DataFrame
              weather_data = weather_data_all.head(1)
    print(f"Fetched {len(weather_data)} latest weather record(s).")


    # Format the fetched data for the LLM
    print("\nFormatting data for LLM from Option 1 fetch...")
    llm_input_data_option1 = format_data_for_llm(station_data, traffic_data, weather_data)

    print("\n--- Formatted Data for LLM (Option 1) ---")
    # Print the structured dictionary or simulate prompt content
    print("Stations:")
    for s in llm_input_data_option1.get('stations', []):
        print(s)
    print("\nTraffic:")
    for t in llm_input_data_option1.get('traffic', []):
        print(t)
    print("\nWeather:")
    for w in llm_input_data_option1.get('weather', []):
        print(w)
    print("-" * 30) # Simple print statement


    # --- Option 2: Fetch data based on a predefined list of station IDs ---
    # This is useful if you've already identified potential stations (e.g., from Step 10's UI filtering)
    # And you want to get their latest status specifically for the LLM.
    # You would also need traffic/weather for the general area or routes to these stations.
    print(f"\n--- Option 2: Fetching data for LLM prep using specific station IDs ---")
    # Example dummy IDs - replace with actual IDs from your DuckDB if testing against populated data
    # Use IDs that you see in the Option 1 output if you don't have other real IDs
    test_station_ids = [
        'here:pds:place:evcp0-NDc0Y2E5NTQtMTI0OS0xMWVlLWI4ODYtNDIwMWBhYTQwMDAy', # Example ID from your output
        'here:pds:place:8409q8yykcg8v-aGVyZS1ldjpjaGFyZ2Vwb2ludDoxNjk3NjI4ODEz', # Example ID from your output
        'here:pds:place:evcp0-MjIyMTEzYTItYTAxNi0xMWVmLWI2YWItNDIwMWBhYTQwMDBh' # Example ID from your output
    ]
    if not test_station_ids:
         print("Warning: test_station_ids list is empty. Skipping Option 2 demo.")
    else:
        print(f"Fetching real-time station data for {len(test_station_ids)} specific IDs...")
        # Fetch stations by specific IDs
        station_data_by_ids = fetch_latest_data_from_duckdb('station_status', entity_ids=test_station_ids)
        print(f"Fetched {len(station_data_by_ids)} latest station records for provided IDs.")

        # For traffic/weather in Option 2, you typically still want the conditions *around the user*
        # or *on potential routes*. Re-using the radius queries from Option 1 is often appropriate
        # unless you have specific road segment IDs from routing (Step 13).
        print(f"Re-using traffic/weather data fetched in Option 1 for the general area.")
        traffic_data_option2 = traffic_data # Re-use from Option 1
        weather_data_option2 = weather_data # Re-use from Option 1

        # Format the fetched data for the LLM
        print("\nFormatting data for LLM from Option 2 fetch...")
        llm_input_data_option2 = format_data_for_llm(station_data_by_ids, traffic_data_option2, weather_data_option2)

        print("\n--- Formatted Data for LLM (Option 2) ---")
        # Print the structured dictionary or simulate prompt content
        print("Stations:")
        for s in llm_input_data_option2.get('stations', []):
            print(s)
        print("\nTraffic:")
        for t in llm_input_data_option2.get('traffic', []):
            print(t)
        print("\nWeather:")
        for w in llm_input_data_option2.get('weather', []):
            print(w)
        print("-" * 30) # Simple print statement


    print("\n--- Testing Complete ---")
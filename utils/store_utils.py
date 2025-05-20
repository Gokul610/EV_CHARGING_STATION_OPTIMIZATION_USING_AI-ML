import duckdb
import logging
import os
import json
from datetime import datetime
from typing import Optional, Any
import pandas as pd

# Setup logging (Step 18.3)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data/store_utils.log',
    filemode='a'
)

# DuckDB configuration
DB_PATH = 'data/ev_charging.duckdb'
os.makedirs('data', exist_ok=True)

def get_db_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Creates and returns a DuckDB connection."""
    try:
        conn = duckdb.connect(database=DB_PATH, read_only=read_only)
        logging.info(f"DuckDB connection established (read_only={read_only}).")
        return conn
    except duckdb.Error as e:
        logging.error(f"Failed to connect to DuckDB at {DB_PATH}: {e}")
        raise

# --- DuckDB Table Creation ---
def initialize_tables() -> None:
    """Initializes DuckDB tables, preserving existing schema."""
    conn = get_db_connection(read_only=False)
    try:
        # Table for API call logs (Step 3.3)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS api_call_logs (
            call_id TEXT PRIMARY KEY,
            api_function TEXT,
            response_data TEXT,
            timestamp TIMESTAMP
        );
        """)

        # Table for station_status (aligned with existing schema: api_station_id, amenities)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS station_status (
            api_station_id TEXT,
            title TEXT,
            lat DOUBLE,
            lon DOUBLE,
            distance_m FLOAT,
            operational_status TEXT,
            available_connectors INTEGER,
            total_connectors INTEGER,
            price FLOAT,
            supported_connectors TEXT,
            network_provider TEXT,
            amenities TEXT,
            timestamp TIMESTAMP,
            fetch_timestamp TIMESTAMP
        );
        """)

        # Table for traffic_conditions
        conn.execute("""
        CREATE TABLE IF NOT EXISTS traffic_conditions (
            jam_factor FLOAT,
            timestamp TIMESTAMP,
            fetch_timestamp TIMESTAMP
        );
        """)

        # Table for weather_info
        conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_info (
            temperature FLOAT,
            humidity FLOAT,
            timestamp TIMESTAMP,
            fetch_timestamp TIMESTAMP
        );
        """)

        logging.info("DuckDB tables checked/created.")
    except duckdb.Error as e:
        logging.error(f"Error initializing tables: {e}")
        raise
    finally:
        conn.close()
        logging.info("DuckDB connection closed after table initialization.")

# Initialize tables on module load
initialize_tables()

# --- Helper Functions ---

def generate_call_id(api_function: str, timestamp: datetime) -> str:
    """Generates a unique ID for API call logs."""
    return f"{api_function}_{timestamp.isoformat().replace(':', '-')}"

def clear_old_sample_data(before_timestamp: datetime) -> None:
    """Removes sample data older than before_timestamp, preserving critical data."""
    conn = get_db_connection(read_only=False)
    try:
        # Delete sample data from api_call_logs (based on call_id pattern)
        conn.execute("""
        DELETE FROM api_call_logs
        WHERE call_id LIKE 'fetch_real_time_station_data_%' OR call_id LIKE 'geocode_place_%'
        AND timestamp < ?
        """, (before_timestamp,))
        logging.info(f"Deleted old sample api_call_logs before {before_timestamp}")

        # Delete sample station_status (based on api_station_id starting with 'SIM_')
        conn.execute("""
        DELETE FROM station_status
        WHERE api_station_id LIKE 'SIM_%' AND fetch_timestamp < ?
        """, (before_timestamp,))
        logging.info(f"Deleted old sample station_status before {before_timestamp}")

        # Delete sample traffic_conditions (based on fetch_timestamp)
        conn.execute("""
        DELETE FROM traffic_conditions
        WHERE fetch_timestamp < ?
        """, (before_timestamp,))
        logging.info(f"Deleted old sample traffic_conditions before {before_timestamp}")

        # Delete sample weather_info (based on fetch_timestamp)
        conn.execute("""
        DELETE FROM weather_info
        WHERE fetch_timestamp < ?
        """, (before_timestamp,))
        logging.info(f"Deleted old sample weather_info before {before_timestamp}")
    except duckdb.Error as e:
        logging.error(f"Error clearing old sample data: {e}")
    finally:
        conn.close()
        logging.info("DuckDB connection closed after clearing old sample data.")

def populate_station_status(station_data: pd.DataFrame, timestamp: datetime) -> None:
    """Populates station_status table from fetch_real_time_station_data response."""
    conn = get_db_connection(read_only=False)
    try:
        if station_data.empty:
            logging.warning("Empty station data, skipping station_status population.")
            return
        # Ensure all required columns exist, fill missing with defaults
        required_cols = {
            'api_station_id': 'Unknown',
            'title': 'Unknown Station',
            'lat': 0.0,
            'lon': 0.0,
            'distance_m': 0.0,
            'operational_status': 'unknown',
            'available_connectors': 0,
            'total_connectors': 0,
            'price': 0.0,
            'supported_connectors': 'Unknown',
            'network_provider': 'Unknown',
            'amenities': 'None',
            'timestamp': timestamp.isoformat()
        }
        for col, default in required_cols.items():
            if col not in station_data.columns:
                station_data[col] = default
        # Add fetch_timestamp
        station_data['fetch_timestamp'] = timestamp
        # Log DataFrame columns and sample data
        logging.info(f"Station data columns: {list(station_data.columns)}")
        logging.debug(f"Station data sample: {station_data.head(1).to_dict()}")
        # Insert into station_status
        conn.register('temp_station_data', station_data)
        conn.execute("""
        INSERT INTO station_status (
            api_station_id, title, lat, lon, distance_m, operational_status,
            available_connectors, total_connectors, price, supported_connectors,
            network_provider, amenities, timestamp, fetch_timestamp
        )
        SELECT
            api_station_id, title, lat, lon, distance_m, operational_status,
            available_connectors, total_connectors, price, supported_connectors,
            network_provider, amenities, timestamp, fetch_timestamp
        FROM temp_station_data
        """)
        # Verify insertion
        inserted_rows = conn.execute("SELECT COUNT(*) FROM station_status WHERE fetch_timestamp = ?", (timestamp,)).fetchone()[0]
        logging.info(f"Populated station_status with {inserted_rows} records at {timestamp}")
    except duckdb.Error as e:
        logging.error(f"Error populating station_status: {e}")
    except Exception as e:
        logging.error(f"Unexpected error populating station_status: {e}")
    finally:
        conn.close()
        logging.info("DuckDB connection closed after populating station_status.")

def populate_traffic_conditions(traffic_data: pd.DataFrame, timestamp: datetime) -> None:
    """Populates traffic_conditions table from simulated or API data."""
    conn = get_db_connection(read_only=False)
    try:
        if traffic_data.empty:
            logging.warning("Empty traffic data, skipping traffic_conditions population.")
            return
        # Ensure required columns
        if 'jam_factor' not in traffic_data.columns:
            traffic_data['jam_factor'] = 0.0
        if 'timestamp' not in traffic_data.columns:
            traffic_data['timestamp'] = timestamp.isoformat()
        traffic_data['fetch_timestamp'] = timestamp
        # Log DataFrame columns
        logging.info(f"Traffic data columns: {list(traffic_data.columns)}")
        # Insert into traffic_conditions
        conn.register('temp_traffic_data', traffic_data)
        conn.execute("""
        INSERT INTO traffic_conditions (jam_factor, timestamp, fetch_timestamp)
        SELECT jam_factor, timestamp, fetch_timestamp
        FROM temp_traffic_data
        """)
        # Verify insertion
        inserted_rows = conn.execute("SELECT COUNT(*) FROM traffic_conditions WHERE fetch_timestamp = ?", (timestamp,)).fetchone()[0]
        logging.info(f"Populated traffic_conditions with {inserted_rows} records at {timestamp}")
    except duckdb.Error as e:
        logging.error(f"Error populating traffic_conditions: {e}")
    except Exception as e:
        logging.error(f"Unexpected error populating traffic_conditions: {e}")
    finally:
        conn.close()
        logging.info("DuckDB connection closed after populating traffic_conditions.")

def populate_weather_info(weather_data: pd.DataFrame, timestamp: datetime) -> None:
    """Populates weather_info table from simulated or API data."""
    conn = get_db_connection(read_only=False)
    try:
        if weather_data.empty:
            logging.warning("Empty weather data, skipping weather_info population.")
            return
        # Ensure required columns
        if 'temperature' not in traffic_data.columns:
            weather_data['temperature'] = 0.0
        if 'humidity' not in weather_data.columns:
            weather_data['humidity'] = 0.0
        if 'timestamp' not in weather_data.columns:
            weather_data['timestamp'] = timestamp.isoformat()
        weather_data['fetch_timestamp'] = timestamp
        # Log DataFrame columns
        logging.info(f"Weather data columns: {list(weather_data.columns)}")
        # Insert into weather_info
        conn.register('temp_weather_data', weather_data)
        conn.execute("""
        INSERT INTO weather_info (temperature, humidity, timestamp, fetch_timestamp)
        SELECT temperature, humidity, timestamp, fetch_timestamp
        FROM temp_weather_data
        """)
        # Verify insertion
        inserted_rows = conn.execute("SELECT COUNT(*) FROM weather_info WHERE fetch_timestamp = ?", (timestamp,)).fetchone()[0]
        logging.info(f"Populated weather_info with {inserted_rows} records at {timestamp}")
    except duckdb.Error as e:
        logging.error(f"Error populating weather_info: {e}")
    except Exception as e:
        logging.error(f"Unexpected error populating weather_info: {e}")
    finally:
        conn.close()
        logging.info("DuckDB connection closed after populating weather_info.")

# --- Data Storage Functions ---

def log_api_call(api_function: str, response_data: Any, timestamp: Optional[datetime] = None) -> None:
    """Logs API call responses from api_utils.py to DuckDB (Step 3.3)."""
    conn = get_db_connection(read_only=False)
    try:
        if timestamp is None:
            timestamp = datetime.utcnow()
        call_id = generate_call_id(api_function, timestamp)
        # Convert response_data to JSON
        if isinstance(response_data, pd.DataFrame):
            response_json = response_data.to_json(orient='records', lines=False)
            logging.info(f"Logging DataFrame for {api_function} with {len(response_data)} rows")
        else:
            response_json = json.dumps(response_data, default=str)
            logging.info(f"Logging non-DataFrame for {api_function}: {response_json[:100]}...")
        conn.execute("""
        INSERT INTO api_call_logs (call_id, api_function, response_data, timestamp)
        VALUES (?, ?, ?, ?)
        """, (call_id, api_function, response_json, timestamp))
        # Verify insertion
        inserted_rows = conn.execute("SELECT COUNT(*) FROM api_call_logs WHERE call_id = ?", (call_id,)).fetchone()[0]
        logging.info(f"Logged API call {api_function} with {inserted_rows} record(s)")
        # Populate specific tables for visualization
        if api_function == "fetch_real_time_station_data" and isinstance(response_data, pd.DataFrame):
            populate_station_status(response_data, timestamp)
    except duckdb.Error as e:
        logging.error(f"Error logging API call {api_function}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error logging API call {api_function}: {e}")
    finally:
        conn.close()
        logging.info("DuckDB connection closed after logging API call.")

def store_user_data_for_api_call(lat: float, lon: float, api_function: str, response_data: Any, timestamp: Optional[datetime] = None) -> None:
    """Stores API call response for each API call (Step 3.3)."""
    logging.info(f"Storing data for API call {api_function} at ({lat}, {lon})")
    try:
        if response_data is None:
            logging.warning(f"No response data for {api_function}, skipping storage.")
            return
        if isinstance(response_data, pd.DataFrame) and response_data.empty:
            logging.warning(f"Empty DataFrame for {api_function}, skipping storage.")
            return
        if timestamp is None:
            timestamp = datetime.utcnow()
        # Log the API call response
        log_api_call(api_function, response_data, timestamp)
        logging.info(f"Successfully stored data for API call {api_function}")
    except Exception as e:
        logging.error(f"Failed to store data for API call {api_function}: {e}")

def insert_sample_data(clear_old: bool = False) -> None:
    """Inserts sample data into api_call_logs and station_status for testing."""
    logging.info("Inserting sample data into DuckDB...")
    test_lat = 9.0548  # Surandai, Tamil Nadu
    test_lon = 77.4335
    test_timestamp = datetime.utcnow()

    if clear_old:
        clear_old_sample_data(test_timestamp)

    # Sample station data
    test_station_data = pd.DataFrame([
        {
            'api_station_id': 'SIM_1',
            'title': 'Simulated Station 1',
            'lat': 9.0549,
            'lon': 77.4336,
            'distance_m': 100.0,
            'operational_status': 'operational',
            'available_connectors': 5,
            'total_connectors': 10,
            'price': 2.5,
            'supported_connectors': 'CCS',
            'network_provider': 'Electrify America',
            'amenities': 'Restrooms, Cafe',
            'timestamp': test_timestamp.isoformat()
        },
        {
            'api_station_id': 'SIM_2',
            'title': 'Simulated Station 2',
            'lat': 9.0550,
            'lon': 77.4337,
            'distance_m': 200.0,
            'operational_status': 'maintenance',
            'available_connectors': 3,
            'total_connectors': 8,
            'price': 3.0,
            'supported_connectors': 'Type 2',
            'network_provider': 'ChargePoint',
            'amenities': 'Parking',
            'timestamp': test_timestamp.isoformat()
        }
    ])
    store_user_data_for_api_call(test_lat, test_lon, "fetch_real_time_station_data", test_station_data, test_timestamp)

    # Sample geocode data
    test_geocode_data = {"place": "Surandai, Tamil Nadu", "coordinates": [9.0548, 77.4335]}
    store_user_data_for_api_call(test_lat, test_lon, "geocode_place", test_geocode_data, test_timestamp)

    # Sample traffic and weather data
    test_traffic_data = pd.DataFrame([
        {'jam_factor': 5.0, 'timestamp': test_timestamp.isoformat()}
    ])
    populate_traffic_conditions(test_traffic_data, test_timestamp)

    test_weather_data = pd.DataFrame([
        {'temperature': 25.0, 'humidity': 60.0, 'timestamp': test_timestamp.isoformat()}
    ])
    populate_weather_info(test_weather_data, test_timestamp)

    logging.info("Sample data insertion completed.")

# --- Testing Block ---
if __name__ == "__main__":
    insert_sample_data(clear_old=True)
    # Verify stored data
    conn = get_db_connection(read_only=True)
    try:
        logging.info("\nLatest 5 API call logs:")
        api_logs_df = conn.execute("""
        SELECT call_id, api_function, timestamp
        FROM api_call_logs
        ORDER BY timestamp DESC
        LIMIT 5
        """).fetchdf()
        print(api_logs_df)

        logging.info("\nLatest 5 station_status records:")
        station_df = conn.execute("""
        SELECT api_station_id, title, available_connectors, fetch_timestamp
        FROM station_status
        ORDER BY fetch_timestamp DESC
        LIMIT 5
        """).fetchdf()
        print(station_df)

        logging.info("\nLatest 5 traffic_conditions records:")
        traffic_df = conn.execute("""
        SELECT jam_factor, fetch_timestamp
        FROM traffic_conditions
        ORDER BY fetch_timestamp DESC
        LIMIT 5
        """).fetchdf()
        print(traffic_df)

        logging.info("\nLatest 5 weather_info records:")
        weather_df = conn.execute("""
        SELECT temperature, humidity, fetch_timestamp
        FROM weather_info
        ORDER BY fetch_timestamp DESC
        LIMIT 5
        """).fetchdf()
        print(weather_df)
    except duckdb.Error as e:
        logging.error(f"Error querying DuckDB for verification: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during verification: {e}")
    finally:
        conn.close()
        logging.info("DuckDB connection closed after testing.")
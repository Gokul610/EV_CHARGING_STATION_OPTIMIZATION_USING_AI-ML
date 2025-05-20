# utils/dbcheck.py
import duckdb
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data/dbcheck.log',
    filemode='a'
)

# Connect to DuckDB in read-only mode
try:
    conn = duckdb.connect('data/ev_charging.duckdb', read_only=True)
    logging.info("DuckDB read-only connection established.")
except duckdb.Error as e:
    logging.error(f"Failed to connect to DuckDB: {e}")
    raise

try:
    # Query api_call_logs
    logging.info("Querying api_call_logs")
    try:
        api_logs_df = conn.execute("""
        SELECT * FROM api_call_logs
        ORDER BY timestamp DESC
        LIMIT 5
        """).fetchdf()
        logging.info(f"api_call_logs retrieved {len(api_logs_df)} rows")
        print("\nLatest 5 API call logs:")
        print(api_logs_df)
    except duckdb.Error as e:
        logging.error(f"Error querying api_call_logs: {e}")
        print("\nLatest 5 API call logs: Empty DataFrame due to error.")

    # Query station_status
    logging.info("Querying station_status")
    try:
        station_df = conn.execute("""
        SELECT api_station_id, title, available_connectors, fetch_timestamp
        FROM station_status
        ORDER BY fetch_timestamp DESC
        LIMIT 5
        """).fetchdf()
        logging.info(f"station_status retrieved {len(station_df)} rows")
        print("\nLatest 5 station_status records:")
        print(station_df)
    except duckdb.Error as e:
        logging.error(f"Error querying station_status: {e}")
        print("\nLatest 5 station_status records: Empty DataFrame due to error.")

    # Query traffic_conditions
    logging.info("Querying traffic_conditions")
    try:
        traffic_df = conn.execute("""
        SELECT jam_factor, fetch_timestamp
        FROM traffic_conditions
        ORDER BY fetch_timestamp DESC
        LIMIT 5
        """).fetchdf()
        logging.info(f"traffic_conditions retrieved {len(traffic_df)} rows")
        print("\nLatest 5 traffic_conditions records:")
        print(traffic_df)
    except duckdb.Error as e:
        logging.error(f"Error querying traffic_conditions: {e}")
        print("\nLatest 5 traffic_conditions records: Empty DataFrame due to error.")

    # Query weather_info
    logging.info("Querying weather_info")
    try:
        weather_df = conn.execute("""
        SELECT temperature, humidity, fetch_timestamp
        FROM weather_info
        ORDER BY fetch_timestamp DESC
        LIMIT 5
        """).fetchdf()
        logging.info(f"weather_info retrieved {len(weather_df)} rows")
        print("\nLatest 5 weather_info records:")
        print(weather_df)
    except duckdb.Error as e:
        logging.error(f"Error querying weather_info: {e}")
        print("\nLatest 5 weather_info records: Empty DataFrame due to error.")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
finally:
    conn.close()
    logging.info("DuckDB connection closed.")
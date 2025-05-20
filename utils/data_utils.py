import os
import time
import duckdb
import pandas as pd

# Set project root & database paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "charging_stations.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "ev_charging.duckdb")  # Persistent DuckDB file

# Connect to persistent DuckDB
conn = duckdb.connect(database=DB_PATH, read_only=False)

def load_data():
    """Loads charging station data from CSV into DuckDB."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

    conn.execute("DROP TABLE IF EXISTS charging_stations")  # Ensure clean load
    conn.execute(f"""
        CREATE TABLE charging_stations AS 
        SELECT * FROM read_csv_auto('{CSV_PATH}')
    """)
    print("✅ Data loaded successfully into DuckDB!")

def query_data(query):
    """Executes an SQL query and returns the result as a Pandas DataFrame."""
    try:
        return conn.execute(query).fetchdf()
    except Exception as e:
        print(f"⚠ Query execution error: {e}")
        return None

def check_table_exists():
    """Checks if the 'charging_stations' table exists in DuckDB."""
    result = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'charging_stations'
    """).fetchone()
    return result[0] > 0

def simulate_real_time_data():
    """Simulates real-time changes in station availability and demand."""
    if not check_table_exists():
        print("⚠ Error: 'charging_stations' table not found. Did you run load_data()?") 
        return

    conn.execute("""
        UPDATE charging_stations
        SET availability = GREATEST(0, availability + CAST(ROUND((random() * 3) - 1, 0) AS INTEGER)),
            demand       = GREATEST(0, demand + CAST(ROUND((random() * 5) - 2, 0) AS INTEGER))
    """)
    print("🔄 Real-time data updated!")

def refresh_data(interval=5, iterations=3):
    """Periodically refresh data for real-time simulation."""
    if not check_table_exists():
        print("⚠ Error: 'charging_stations' table not found. Did you run load_data()?") 
        return

    for i in range(iterations):
        simulate_real_time_data()
        time.sleep(interval)
        print(f"✅ Data refresh cycle {i+1} complete!")

if __name__ == "__main__":
    load_data()
    simulate_real_time_data()
    refresh_data(interval=3, iterations=2)
    print("✅ All processes completed successfully!")

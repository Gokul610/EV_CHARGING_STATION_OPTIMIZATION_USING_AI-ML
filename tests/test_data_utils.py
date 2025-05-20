import os
import sys
import time

# Explicitly add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.data_utils import load_data, query_data, simulate_real_time_data, refresh_data, check_table_exists

print("🔄 Loading data into DuckDB...")
load_data()  # Ensure database is initialized before running any tests

if not check_table_exists():
    print("⚠ Critical Error: 'charging_stations' table still not found! Aborting tests.")
    sys.exit(1)  # Exit the script if table is missing

# Test SQL Query
test_query = "SELECT * FROM charging_stations WHERE city = 'Chennai' LIMIT 5"
result_filtered = query_data(test_query)

assert result_filtered is not None, "⚠ Query returned None"
assert not result_filtered.empty, "⚠ Query returned no results"
print("✅ Filtered query executed successfully!\n", result_filtered)

# Simulate real-time updates
print("🔄 Simulating real-time data once...")
simulate_real_time_data()

# Run data refresh simulation
print("🔄 Running `refresh_data()` with 2 cycles at 3-second intervals...")
refresh_data(interval=3, iterations=2)

print("✅ All tests completed successfully!")

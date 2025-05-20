# evdb_inspect.py

import duckdb

# Connect to your DuckDB database file
# Make sure the file exists at this path or DuckDB will create a new one
db_path = 'data/ev_charging.duckdb'
con = duckdb.connect(db_path)

# Step 1: Show all tables in the database
print("Listing all tables in the database...")
tables = con.execute("SHOW TABLES").fetchall()

# If no tables found, print a message
if not tables:
    print("No tables found in the database.")
else:
    # Step 2: For each table, show its content
    for table in tables:
        table_name = table[0]
        print(f"\n--- Contents of table: {table_name} ---")
        try:
            # Fetch all rows from the current table
            df = con.execute(f"SELECT * FROM {table_name}").df()
            print(df)
        except Exception as e:
            print(f"Error reading table {table_name}: {e}")
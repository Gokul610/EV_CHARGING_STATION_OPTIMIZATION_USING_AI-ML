#D:\Desktop\ev_charging_optimization\utils\fetch_real_time_data.py
import duckdb
import json
import time
import hashlib
import requests
import logging
from diskcache import Cache
from typing import Dict, Any, List, Optional
from datetime import datetime
import os


# --- Configuration ---
# API Key: It's strongly recommended to load this from a secure configuration
# file (e.g., utils/config.py) or environment variables in a real application.
API_KEY = os.environ.get("HERE_API_KEY", "kdL0wYxEBT426TAt0R9I2V_k9A8udwLVqSG3GN-07ic")
if API_KEY == "kdL0wYxEBT426TAt0R9I2V_k9A8udwLVqSG3GN-07ic":
    logging.warning("HERE_API_KEY not set in environment variables. Using placeholder key.")


# API Endpoints
# Using Browse V1 for initial discovery
BROWSE_API_URL_V1 = "https://browse.search.hereapi.com/v1/browse"
# Using EV Charge Points V3 /locations for real-time status by ID
# Using the hostname that worked in your browser test for lookup by ID
EV_LOCATIONS_API_URL_V3 = "https://evcp.hereapi.com/v3/locations"
TRAFFIC_FLOW_API_URL_V7 = "https://data.traffic.hereapi.com/v7/flow"
WEATHER_API_URL_V1 = "https://weather.ls.hereapi.com/weather/1.0/report.json"
ROUTING_API_URL_V8 = "https://router.hereapi.com/v8/routes"


# Setup logging (basic config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup DuckDB and diskcache
os.makedirs('data', exist_ok=True)
DB_PATH = 'data/ev_charging.duckdb'
conn = duckdb.connect(database=DB_PATH, read_only=False)

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
cache = Cache(CACHE_DIR)

# Cache expiration times (in seconds) - Adjust as needed based on data volatility
# Note: Caching for the 2-step station fetch is handled within the function
CACHE_TTL_TRAFFIC = 60  # 1 minute
CACHE_TTL_WEATHER = 300 # 5 minutes
CACHE_TTL_ROUTE = 600   # 10 minutes

# --- DuckDB Table Creation ---
# Schema to store real-time data, combining info from Browse V1 and Locations V3
# Using api_station_id as PRIMARY KEY assuming V1 Browse provides IDs compatible with V3 Lookup
conn.execute("""
CREATE TABLE IF NOT EXISTS station_status (
    api_station_id TEXT PRIMARY KEY, -- Use API ID as primary key if reliable
    title TEXT,
    lat DOUBLE,
    lon DOUBLE,
    address TEXT,
    -- Basic details from Browse V1 (if available)
    browse_categories TEXT, -- Categories from Browse
    browse_distance FLOAT, -- Distance from browse query point

    -- Real-time/detailed info from Locations V3 (if available)
    is_open BOOLEAN, -- Indicates if the station is generally open
    operational_status TEXT, -- E.g., 'OPERATIONAL', 'UNDER_MAINTENANCE', 'FAULTY' (from V3)
    available_connectors INT, -- Number of available connectors of all types (from V3 EVSEs)
    total_connectors INT, -- Total number of connectors at the station (from V3 EVSEs)
    price TEXT, -- Price information (often complex, storing as JSON string from V3)
    supported_connectors TEXT, -- Comma-separated list of supported connector types (from V3 EVSEs)
    power_rating_kW FLOAT, -- Maximum power rating (from V3 EVSEs)
    amenities TEXT, -- Comma-separated list of amenities (from V3)
    network_provider TEXT, -- Name of the charging network provider (from V3)

    fetch_timestamp TIMESTAMP -- Removed DEFAULT CURRENT_TIMESTAMP here, we'll set it explicitly on INSERT for clarity
);
""")
# Re-added fetch_timestamp with default for consistency using ALTER TABLE
conn.execute("""
ALTER TABLE station_status ADD COLUMN IF NOT EXISTS fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
""")


conn.execute("""
CREATE TABLE IF NOT EXISTS traffic_conditions (
    road_segment_id TEXT, -- A unique ID for the road segment (can be generated if API doesn't provide one)
    location_lat DOUBLE, -- Latitude of the traffic point (often start of segment)
    location_lon DOUBLE, -- Longitude of the traffic point
    jam_factor FLOAT, -- Jam factor (0-10, 10 is highest congestion)
    speed FLOAT, -- Current speed in units defined by API (usually m/s, convert as needed)
    free_flow FLOAT, -- Free flow speed
    speed_limit FLOAT, -- Speed limit of the road segment (if available)
    confidence FLOAT, -- Confidence level of the traffic data (0-1)
    fclass INT, -- Functional Road Class (e.g., 1 for highest class roads)
    traffic_state TEXT, -- E.g., 'freeFlow', 'congestion', 'heavyCongestion' (from API)
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS weather_info (
    location_lat DOUBLE,
    location_lon DOUBLE,
    temperature FLOAT, -- Temperature in Celsius
    temperature_feels_like FLOAT, -- "Feels like" temperature in Celsius
    humidity FLOAT, -- Humidity in percentage
    wind_speed FLOAT, -- Wind speed in m/s
    wind_direction INT, -- Wind direction in degrees (0-360)
    description TEXT, -- Weather description (e.g., 'Sunny', 'Partly cloudy')
    icon_code TEXT, -- Weather icon code (for displaying weather icons)
    fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

logging.info("DuckDB tables checked/created.")


# --- Helper Functions ---

def generate_simple_id(lat: float, lon: float, prefix: str = "") -> str:
     """Generates a basic ID from lat/lon. Useful if API items lack stable IDs."""
     # Handle potential None lat/lon during ID generation if location data is missing
     safe_lat = round(lat, 6) if lat is not None else 0
     safe_lon = round(lon, 6) if lon is not None else 0
     return f"{prefix}_{safe_lat}_{safe_lon}"

def get_cache_key(base_key: str, *args) -> str:
    """Generates a cache key based on base key and arguments."""
    # Use hashlib for robustness against special characters in args
    return f"{base_key}_{hashlib.sha256('_'.join(map(str, args)).encode()).hexdigest()[:10]}"

# --- Data Fetching Functions (2-Step for Stations) ---

def fetch_station_basic_details_v1(lat: float, lon: float, radius: int = 50000) -> List[Dict[str, Any]]:
    """
    Fetches basic EV charging station details using HERE Browse API V1 within a radius.
    Returns a list of station items with basic info and IDs.
    """
    cache_key = get_cache_key('ev_stations_v1_browse', lat, lon, radius)

    # Check cache - cache the list of basic station items from V1 browse
    # Note: The AttributeError on 'created' might require clearing the .cache folder manually
    if cache_key in cache and hasattr(cache, 'created') and time.time() - cache.created(cache_key) < 3600: # Added hasattr check as a workaround
        logging.info("Using cached EV station basic details (V1 Browse)")
        return cache[cache_key]['data']

    try:
        url = BROWSE_API_URL_V1
        params = {
            'at': f"{lat},{lon}",
            'categories': "700-7600-0322", # EV charging station category code for V1
            'in': f"circle:{lat},{lon};r={radius}",
            'apiKey': API_KEY,
            'limit': 100 # Limit the number of results per browse call if needed
        }

        logging.info(f"Fetching EV station basic details from V1 Browse API for {lat},{lon} radius {radius}m...")
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # --- DEBUG PRINT (Removed) ---
        # print("--- V1 Browse Raw Data (First 2 Items) ---")
        # if data.get("items"):
        #     print(json.dumps(data["items"][:2], indent=2))
        # else:
        #      print("[] - No items found in V1 Browse raw data")
        # print("---------------------------------------")
        # --- END DEBUG PRINT (Removed) ---

        logging.info(f"Received {len(data.get('items', []))} basic station results from V1 Browse API.")

        basic_stations = []
        for item in data.get("items", []):
            # Extract essential basic details, including the 'id' which we'll use for V3 lookup
            station_id = item.get('id')
            if station_id: # Only include items with a valid ID
                basic_stations.append({
                    'id': station_id,
                    'title': item.get('title'),
                    'lat': item.get('position', {}).get('lat'),
                    'lon': item.get('position', {}).get('lng'),
                    'address': item.get('address', {}).get('label', 'Unknown'),
                    'categories': ", ".join(c.get('name', '') for c in item.get('categories', [])) if item.get('categories') else None,
                    'distance': item.get('distance')
                })

        # --- DEBUG PRINT (Removed) ---
        # print("--- V1 Browse Extracted Items (First 2) ---")
        # print(basic_stations[:2])
        # print("-------------------------------")
        # --- END DEBUG PRINT (Removed) ---


        # Cache the successful fetch
        cache.set(cache_key, {"timestamp": time.time(), "data": basic_stations})
        return basic_stations

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error fetching EV station basic details (V1 Browse): {e}")
        return [] # Return empty list on error
    except json.JSONDecodeError:
         logging.error("Failed to decode JSON response from V1 Browse API.")
         return []
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching EV station basic details (V1 Browse): {e}")
        return []


def fetch_station_real_time_status_v3_by_ids(station_ids: List[str]) -> Dict[str, Any]:
    """
    Fetches real-time status for a list of station IDs using HERE EV Charge Points API V3 /locations.
    Returns a dictionary mapping station ID to detailed status data.
    """
    if not station_ids:
        return {}

    # Ensure unique IDs and split into chunks if needed (API limits might apply)
    unique_ids = list(set(station_ids))
    # HERE /locations endpoint typically supports multiple IDs separated by commas
    # Check API docs for maximum number of IDs allowed in one request. Let's use a chunk size.
    CHUNK_SIZE = 50 # Example chunk size, adjust based on API documentation

    detailed_status_map = {}

    for i in range(0, len(unique_ids), CHUNK_SIZE):
        chunk_ids = unique_ids[i : i + CHUNK_SIZE]
        ids_param = ",".join(chunk_ids)

        # Cache key based on the chunk of IDs
        cache_key = get_cache_key('ev_stations_v3_locations_chunk', ids_param)

        # Check cache for this chunk of IDs
        # Cache V3 status for short duration as it's real-time
        if cache_key in cache and hasattr(cache, 'created') and time.time() - cache.created(cache_key) < 60: # Added hasattr check
            logging.info(f"Using cached EV station real-time status (V3 Locations) for chunk {i//CHUNK_SIZE + 1}")
            detailed_status_map.update(cache[cache_key]['data'])
            continue # Skip API call for this chunk

        try:
            url = EV_LOCATIONS_API_URL_V3
            params = {
                'apiKey': API_KEY,
                'ids': ids_param,
                'relatedAttribute': 'EVSE' # Request EVSE (connector) data for status
                 # Add other parameters if needed, but IDs and apiKey are core for this endpoint
            }

            logging.info(f"Fetching EV station real-time status from V3 Locations API for {len(chunk_ids)} stations (chunk {i//CHUNK_SIZE + 1})...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # --- DEBUG PRINT (Removed) ---
            # print(f"--- V3 Locations Raw Data (First 2 Items, Chunk {i//CHUNK_SIZE + 1}) ---")
            # if data.get("items"):
            #      print(json.dumps(data["items"][:2], indent=2))
            # else:
            #      print("[] - No items found in V3 Locations raw data for this chunk")
            # print("---------------------------------------------------------------")
            # --- END DEBUG PRINT (Removed) ---


            logging.info(f"Received status for {len(data.get('items', []))} stations from V3 Locations API (chunk {i//CHUNK_SIZE + 1}).")

            chunk_status_map = {}
            for item in data.get("items", []):
                station_id = item.get("id")
                if station_id: # Only include items with a valid ID
                     # Extract detailed real-time info from the V3 Locations response structure
                     # This parsing is based on the V3 /locations/{locationId} or ?ids= documentation
                     # We need to be careful about nested structures that might be missing or empty.
                    evses = item.get("evses", []) # Get EVSE list, default to empty list if key is missing
                    available_connectors = sum(1 for evse in evses if evse.get("status") == "AVAILABLE") # Sum available statuses
                    total_connectors = len(evses) # Total number of EVSEs

                    supported_connectors_set = set()
                    power_ratings = set()
                    for evse in evses:
                         # Connectors list is nested inside each EVSE
                         for conn_info in evse.get("connectors", []): # Get connectors list, default to empty
                              connector_type = conn_info.get("type")
                              if connector_type:
                                   supported_connectors_set.add(connector_type)
                              # PowerOutput is nested inside connector info
                              power_output_kw = conn_info.get("powerOutput", {}).get("kW") # Get powerOutput dict, default to empty, then get kW
                              if power_output_kw is not None: # Check specifically for None, as 0 is a valid power
                                   power_ratings.add(power_output_kw)

                    supported_connectors_str = ", ".join(supported_connectors_set) if supported_connectors_set else None
                    power_rating_kW = max(power_ratings) if power_ratings else None # Use max power rating, handle empty set

                    chunk_status_map[station_id] = {
                        'is_open': item.get("openingHours", {}).get("isOpen", None), # Access nested dict, default to empty dict
                        'operational_status': item.get("operationalStatus"),
                        'available_connectors': available_connectors,
                        'total_connectors': total_connectors,
                        'price': json.dumps(item.get("chargingPrice")) if item.get("chargingPrice") else None, # Store price as JSON string
                        'supported_connectors': supported_connectors_str,
                        'power_rating_kW': power_rating_kW,
                        'amenities': ", ".join(item.get("amenities", [])) if item.get("amenities") else None, # Join amenities list
                        'network_provider': item.get("chargingStationOperator", {}).get("title") # Access nested dict
                    }

            # --- DEBUG PRINT (Removed) ---
            # print(f"--- V3 Locations Extracted Map (First 2, Chunk {i//CHUNK_SIZE + 1}) ---")
            # print(dict(list(chunk_status_map.items())[:2]))
            # print("-----------------------------------------------------")
            # --- END DEBUG PRINT (Removed) ---


            # Cache the successful fetch for this chunk
            cache.set(cache_key, {"timestamp": time.time(), "data": chunk_status_map})
            detailed_status_map.update(chunk_status_map) # Add chunk results to overall map

        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP request error fetching EV station real-time status (V3 Locations) chunk {i//CHUNK_SIZE + 1}: {e}")
            # Continue with other chunks even if one fails
        except json.JSONDecodeError:
             logging.error(f"Failed to decode JSON response from V3 Locations API (chunk {i//CHUNK_SIZE + 1}).")
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching EV station real-time status (V3 Locations) chunk {i//CHUNK_SIZE + 1}: {e}")

    return detailed_status_map # Return the combined map


def fetch_and_store_station_data_two_step(lat: float, lon: float, radius: int = 50000) -> None:
    """
    Executes the two-step process to fetch station data (Browse V1 + Locations V3)
     and stores the combined data in DuckDB.
    """
    logging.info(f"Starting 2-step fetch for station data for location {lat},{lon} radius {radius}m...")

    # Step 1: Fetch basic details using V1 Browse API
    basic_stations = fetch_station_basic_details_v1(lat, lon, radius)

    if not basic_stations:
        logging.warning("No basic station details found from V1 Browse API. Skipping V3 Locations fetch and data storage.")
        return

    # Extract IDs from the basic details - only use IDs that look like HERE IDs if possible
    # HERE IDs often have a specific format (e.g., starts with "here:"), might need refining
    station_ids = [station.get('id') for station in basic_stations if station.get('id')] # Simple ID extraction
    # Example filter for HERE IDs if needed:
    # station_ids = [station.get('id') for station in basic_stations if station.get('id') and station.get('id').startswith('here:')]


    if not station_ids:
        logging.warning("No valid station IDs found from V1 Browse API to use for V3 Locations fetch. Skipping data storage.")
        return

    # Step 2: Fetch real-time status using V3 Locations API based on IDs
    detailed_status_map = fetch_station_real_time_status_v3_by_ids(station_ids)

    # Combine data and insert into DuckDB
    # MODIFIED: We will attempt to insert basic details even if detailed status is not found/empty
    logging.info(f"Combining basic details and (possibly empty) real-time status for {len(basic_stations)} stations and storing in DuckDB.")
    inserted_count = 0
    # Get current timestamp to use for all inserted records from this fetch run
    current_fetch_timestamp = datetime.utcnow()
    insert_print_limit = 5 # Limit insertion prints to the first 5 stations

    for i, basic_station in enumerate(basic_stations): # Iterate through basic stations found
        station_id = basic_station.get('id')
        if not station_id:
             logging.warning(f"Skipping station with missing ID: {basic_station}")
             continue # Skip if basic station is missing an ID

        # Get detailed info if available, otherwise use defaults or None
        # Use .get() with an empty dictionary as default if station_id is not in detailed_status_map
        detailed_info = detailed_status_map.get(station_id, {})

        try:
             # --- DEBUG PRINT (Removed) ---
             # if i < insert_print_limit:
             #     print(f"--- Attempting Insert for Station ID: {station_id} ({i + 1}/{min(len(basic_stations), insert_print_limit)}) ---")
             #     print(f"Basic Info: {basic_station}")
             #     print(f"Detailed Info (from V3 lookup, might be empty): {detailed_info}")
             #     print("---------------------------------------------")
             # elif i == insert_print_limit:
             #     print(f"--- Skipping station insert prints after {insert_print_limit} ---")
             # --- END DEBUG PRINT (Removed) ---

             # Use basic station info for core fields, and detailed_info for others, providing defaults
             # Use INSERT OR REPLACE to handle duplicate keys when running multiple times
             conn.execute("""
             INSERT OR REPLACE INTO station_status (
                 api_station_id, title, lat, lon, address,
                 browse_categories, browse_distance,
                 is_open, operational_status, available_connectors, total_connectors,
                 price, supported_connectors, power_rating_kW, amenities, network_provider,
                 fetch_timestamp
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             """, (
                 station_id,
                 basic_station.get('title'),
                 basic_station.get('lat'),
                 basic_station.get('lon'),
                 basic_station.get('address'),
                 basic_station.get('categories'),
                 basic_station.get('distance'),
                 # Get from detailed_info dictionary, providing defaults if key is missing
                 detailed_info.get('is_open', None),
                 detailed_info.get('operational_status', None),
                 detailed_info.get('available_connectors', 0), # Default to 0 if not found
                 detailed_info.get('total_connectors', 0),   # Default to 0 if not found
                 detailed_info.get('price', None),
                 detailed_info.get('supported_connectors', None),
                 detailed_info.get('power_rating_kW', None), # Default to None if not found
                 detailed_info.get('amenities', None),
                 detailed_info.get('network_provider', None),
                 current_fetch_timestamp # Use the consistent timestamp
             ))
             inserted_count += 1
        except Exception as e:
            # Log the error but continue with other stations
            # The most common error now should be the duplicate key, which INSERT OR REPLACE prevents
            # Other errors might indicate schema issues or data problems
            logging.error(f"Error inserting data for station {station_id}: {e}")

    logging.info(f"Finished 2-step fetch. Attempted to insert {len(basic_stations)} stations. Successfully inserted {inserted_count} records into DuckDB.")


def fetch_traffic_flow(lat: float, lon: float, radius: int = 1000) -> None:
    """
    Fetches real-time traffic flow using HERE Traffic API v7 and stores in DuckDB.
    """
    cache_key = get_cache_key('traffic_flow_v7', lat, lon, radius)
    if cache_key in cache and hasattr(cache, 'created') and time.time() - cache.created(cache_key) < CACHE_TTL_TRAFFIC: # Added hasattr check
        logging.info("Using cached traffic data (V7)")
        return

    try:
        url = TRAFFIC_FLOW_API_URL_V7
        params = {
            'in': f"circle:{lat},{lon};r={radius}", # Search radius around the point
            'locationReferencing': 'shape', # Request shape for location data
            'apiKey': API_KEY
        }
        logging.info(f"Fetching traffic flow from V7 API for {lat},{lon} radius {radius}m...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # --- DEBUG PRINT (Removed) ---
        # print("--- V7 Traffic Raw Data (First 2 Items) ---")
        # if data.get("results"):
        #      print(json.dumps(data["results"][:2], indent=2))
        # else:
        #      print("[] - No items found in V7 Traffic raw data")
        # print("-----------------------------------------")
        # --- END DEBUG PRINT (Removed) ---


        logging.info(f"Received {len(data.get('results', []))} traffic flow results from V7 API.")

        # Get current timestamp for insertion
        current_fetch_timestamp = datetime.utcnow()

        insert_print_limit = 5 # Limit insertion prints to the first 5 traffic items
        traffic_inserted_count = 0

        for idx, item in enumerate(data.get("results", [])):
            # Extract data with corrected and robust defaulting based on observed structure
            location_data = item.get("location", {})
            flow_item_data = item.get("flowItem", {}) # Use flow_item_data dict

            # --- CORRECTED: Extract lat/lon from the first point of the first link in the shape ---
            # Access elements safely with checks at each step
            lat_val, lon_val = None, None
            shape_data = location_data.get("shape", {})
            links = shape_data.get("links", [])
            if links and len(links) > 0:
                 first_link = links[0]
                 points = first_link.get("points", [])
                 if points and len(points) > 0:
                      first_point = points[0]
                      lat_val = first_point.get('lat')
                      lon_val = first_point.get('lng') # Note: 'lng' for longitude in HERE APIs
            # ----------------------------------------------------------------------------------


            # Only attempt to process/insert if location data was successfully extracted
            if lat_val is not None and lon_val is not None:
                try:
                    # Extract other fields from nested dicts with defaults *only if location exists*
                    current = flow_item_data.get("currentFlow", {}) # Use current dict, default to empty
                    free = flow_item_data.get("freeFlow", {}) # Use free dict, default to empty

                    jam_factor = current.get("jamFactor", 0.0) # Default to 0.0 if missing
                    speed = current.get("speed", 0.0)         # Default to 0.0 if missing (assuming m/s)
                    free_flow = free.get("speed", 0.0)       # Default to 0.0 if missing (assuming m/s)
                    confidence = current.get("confidence", 0.0) # Default to 0.0 if missing

                    speed_limit = item.get("speedLimit", None) # Keep None if not available
                    fclass = item.get("functionalRoadClass", 0) # Default to 0 if missing
                    traffic_state = item.get("trafficState", "unknown") # Default to "unknown" if missing


                    # Extract road segment ID if available, otherwise generate one using the extracted lat/lon
                    # Use flow_item_data for locationReference
                    road_segment_id = flow_item_data.get("locationReference", {}).get("id")
                    if not road_segment_id:
                         # Generate a simple ID based on the extracted lat/lon
                         road_segment_id = generate_simple_id(lat_val, lon_val, "traffic")


                    # --- DEBUG PRINT (Removed) ---
                    # if traffic_inserted_count < insert_print_limit:
                    #      print(f"--- Extracted Traffic Data for Segment: {road_segment_id} ({traffic_inserted_count + 1}/{min(len(data.get('results', [])), insert_print_limit)}) ---")
                    #      print({
                    #           'road_segment_id': road_segment_id,
                    #           'location_lat': lat_val,
                    #           'location_lon': lon_val,
                    #           'jam_factor': jam_factor,
                    #           'speed': speed,
                    #           'free_flow': free_flow,
                    #           'speed_limit': speed_limit,
                    #           'confidence': confidence,
                    #           'fclass': fclass,
                    #           'traffic_state': traffic_state,
                    #           'fetch_timestamp': current_fetch_timestamp
                    #      })
                    #      print("---------------------------------------------------")
                    #      print(f"--- Attempting Traffic Insert for Segment: {road_segment_id} ---")
                    #      print("---------------------------------------------------")
                    # elif traffic_inserted_count == insert_print_limit:
                    #      print(f"--- Skipping traffic extraction and insert prints after {insert_print_limit} ---")
                    # --- END DEBUG PRINT (Removed) ---


                    conn.execute("""
                    INSERT INTO traffic_conditions (
                        road_segment_id, location_lat, location_lon, jam_factor,
                        speed, free_flow, speed_limit, confidence, fclass, traffic_state,
                        fetch_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        road_segment_id,
                        lat_val, # Use extracted values
                        lon_val,
                        jam_factor,
                        speed,
                        free_flow,
                        speed_limit,
                        confidence,
                        fclass,
                        traffic_state,
                        current_fetch_timestamp
                    ))
                    traffic_inserted_count += 1
                except Exception as e:
                    logging.error(f"Error inserting traffic data for segment {road_segment_id}: {e}")
            else:
                 # Log a warning if location extraction failed
                 # This helps diagnose *why* extraction failed by showing the raw location data
                 logging.warning(f"Skipping traffic segment due to failed location data extraction attempt. Raw location data: {location_data}")


        cache.set(cache_key, {"timestamp": time.time(), "count": len(data.get("results", []))})
        logging.info(f"Successfully fetched {len(data.get('results', []))} traffic results. Successfully inserted {traffic_inserted_count} records.")

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error fetching traffic flow (V7): {e}")
    except json.JSONDecodeError:
         logging.error("Failed to decode JSON response from Traffic API (V7).")
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching traffic flow (V7): {e}")


def fetch_weather(lat: float, lon: float) -> None:
    """
    Fetches weather information using HERE Weather API v1 (observation product)
    and stores in DuckDB.
    """
    cache_key = get_cache_key('weather_v1_obs', lat, lon) # Use lat/lon directly for key
    if cache_key in cache and hasattr(cache, 'created') and time.time() - cache.created(cache_key) < CACHE_TTL_WEATHER: # Added hasattr check
        logging.info("Using cached weather data (V1)")
        return

    try:
        url = WEATHER_API_URL_V1
        params = {
            'product': 'observation', # Request current weather observation
            'latitude': lat,
            'longitude': lon,
            'apiKey': API_KEY
        }
        logging.info(f"Fetching weather info from V1 API for {lat},{lon}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # --- DEBUG PRINT (Removed) ---
        # print("--- V1 Weather Raw Data (First Item) ---")
        # if data.get("observations", {}).get("location"):
        #     print(json.dumps(data["observations"]["location"][0], indent=2))
        # else:
        #     print("[] - No location found in V1 Weather raw data")
        # print("-------------------------------------")
        # --- END DEBUG PRINT (Removed) ---


        logging.info("Received weather data from V1 API.")

        # Get current timestamp for insertion
        current_fetch_timestamp = datetime.utcnow()

        # The response structure for observation is nested
        observation = data.get("observations", {}).get("location", [])
        if observation and observation[0].get("observation"):
             obs_data = observation[0]["observation"][0]

             # --- DEBUG PRINT (Removed) ---
             # print("--- V1 Weather Extracted Data ---")
             # print(obs_data)
             # print("-------------------------------")
             # --- END DEBUG PRINT (Removed) ---


             temperature = float(obs_data.get("temperature", 0))
             temperature_feels_like = float(obs_data.get("comfort", temperature)) # 'comfort' often holds feels like
             humidity = float(obs_data.get("humidity", 0))
             wind_speed = float(obs_data.get("windSpeed", 0))
             # Basic parsing for wind direction text like "270 deg"
             wind_direction_text = obs_data.get("windDesc", "0")
             try:
                 wind_direction = int(wind_direction_text.split(" ")[0].strip())
             except ValueError:
                 wind_direction = 0 # Default to 0 if parsing fails

             description = obs_data.get("description", "Unknown")
             icon_code = obs_data.get("skyInfo", "unknown") # 'skyInfo' often contains a code

             try:
                 conn.execute("""
                 INSERT INTO weather_info (
                     location_lat, location_lon, temperature, temperature_feels_like,
                     humidity, wind_speed, wind_direction, description, icon_code,
                     fetch_timestamp
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 """, (
                     lat, lon, temperature, temperature_feels_like, humidity,
                     wind_speed, wind_direction, description, icon_code,
                     current_fetch_timestamp
                 ))
                 logging.info("Fetched and stored weather record.")
             except Exception as e:
                logging.error(f"Error inserting weather data: {e}")

        else:
             logging.warning(f"No weather observation data found for {lat},{lon}.")


    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error fetching weather info (V1): {e}")
    except json.JSONDecodeError:
         logging.error("Failed to decode JSON response from Weather API (V1).")
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching weather info (V1): {e}")


def fetch_optimal_route_summary(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[Dict[str, Any]]:
    """
    Fetches optimal route summary using HERE Routing API v8. Returns the summary dictionary.
    Does NOT store in DuckDB by default, as route requests are usually one-off.
    """
    # Cache key includes both origin and destination
    cache_key = get_cache_key('route_summary_v8', lat1, lon1, lat2, lon2)
    if cache_key in cache and hasattr(cache, 'created') and time.time() - cache.created(cache_key) < CACHE_TTL_ROUTE: # Added hasattr check
        logging.info("Using cached route summary (V8)")
        return cache[cache_key]['data'] # Return cached data

    try:
        url = ROUTING_API_URL_V8
        params = {
            'origin': f"{lat1},{lon1}",
            'destination': f"{lat2},{lon2}",
            'transportMode': 'car', # Or 'electriccar' if available and needed
            'return': 'summary,polyline', # Request summary and polyline geometry
            'apiKey': API_KEY
        }
        logging.info(f"Fetching route summary from V8 API from {lat1},{lon1} to {lat2},{lon2}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        logging.info("Received route data from V8 API.")

        if data.get("routes"):
            # Assuming the first route and first section are the relevant ones
            route_summary = data["routes"][0]["sections"][0]["summary"]
            route_geometry = data["routes"][0]["sections"][0].get("polyline") # Polyline for map visualization

            # Calculate traffic delay by comparing base duration (no traffic) to actual duration
            base_duration_sec = route_summary.get("baseDuration", 0)
            traffic_duration_sec = route_summary.get("duration", 0)
            traffic_delay_min = max(0, (traffic_duration_sec - base_duration_sec) / 60)


            result = {
                "distance_km": route_summary.get("length", 0) / 1000, # Length in meters, convert to km
                "duration_min": traffic_duration_sec / 60, # Duration in seconds, convert to minutes
                "traffic_delay_min": traffic_delay_min,
                "text": f"Approx. {route_summary.get('length', 0)/1000:.1f} km, {traffic_duration_sec/60:.1f} min",
                "polyline": route_geometry # Include polyline in return for map display
            }
            cache.set(cache_key, {"timestamp": time.time(), "data": result}) # Cache the result
            logging.info("Successfully fetched and cached route summary.")
            return result
        else:
            logging.warning(f"No route found from {lat1},{lon1} to {lat2},{lon2}.")
            return None # Return None if no route is found

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error fetching optimal route (V8): {e}")
        return None # Return None on error
    except json.JSONDecodeError:
         logging.error("Failed to decode JSON response from Routing API (V8).")
         return None # Return None on error
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching optimal route (V8): {e}")
        return None # Return None on error


# --- Central Call Function ---

def fetch_all_real_time_data(lat: float, lon: float, radius: int = 50000) -> None:
    """
    Fetches all relevant real-time data (stations, traffic, weather) for a given location
    and stores it in DuckDB. Designed to be called periodically or on demand.
    Implements a 2-step process for station data (Browse V1 + Locations V3).
    """
    logging.info(f"Attempting to fetch all real-time data for location {lat},{lon} with radius {radius}m...")

    # Fetch station data using the 2-step process (Browse V1 + Locations V3)
    fetch_and_store_station_data_two_step(lat, lon, radius)

    # Fetch traffic conditions for the area (using the same radius for consistency)
    fetch_traffic_flow(lat, lon, radius)

    # Fetch weather for the central point
    fetch_weather(lat, lon)

    logging.info(f"Finished attempting to fetch all real-time data for location {lat},{lon}.")

# --- Example Usage (for testing when running the script directly) ---
# This block demonstrates how to run the script and verify data when executed directly.
if __name__ == "__main__":
    logging.info("--- Running utils/fetch_real_time_data.py directly for demonstration ---")

    # Example coordinates for testing (e.g., San Francisco, USA)
    # Replace with coordinates relevant to your testing needs if different
    # and ensure you have API access for that region.
    test_lat = 37.7749  # San Francisco Latitude
    test_lon = -122.4194 # San Francisco Longitude
    test_radius = 50000 # 5 km radius for fetching stations and traffic

    # For testing, you might want to try coordinates in a city with more known EV charging stations
    # test_lat = 52.5200 # Berlin
    # test_lon = 13.4050

    # --- Optional: Manually clear cache before run if you suspect cache issues ---
    # print("Attempting to clear cache...")
    # try:
    #      if os.path.exists(CACHE_DIR):
    #           import shutil
    #           shutil.rmtree(CACHE_DIR)
    #           print("Cache directory cleared.")
    #           # Re-initialize cache after clearing
    #           cache = Cache(CACHE_DIR)
    #      else:
    #           print("Cache directory does not exist, no need to clear.")
    # except Exception as e:
    #      print(f"Error clearing cache: {e}")
    # ----------------------------------------------------------------------


    logging.info(f"Triggering fetch_all_real_time_data for location {test_lat},{test_lon} within {test_radius}m radius...")
    # Call the central function to fetch and store all data
    fetch_all_real_time_data(test_lat, test_lon, radius=test_radius)

    logging.info("\n--- Verifying recent data in DuckDB after fetch ---")

    try:
        # Query the latest records inserted during this run based on timestamp
        print("Latest 5 station status records from DuckDB:")
        # Using ORDER BY and LIMIT is more reliable for demonstration than MAX(fetch_timestamp)
        latest_stations_df = conn.execute("""
            SELECT api_station_id, title, lat, lon, available_connectors, operational_status, fetch_timestamp
            FROM station_status
            ORDER BY fetch_timestamp DESC
            LIMIT 5
        """).df()
        print(latest_stations_df)
        if latest_stations_df.empty:
             logging.info("No station status records found in the last minute (using ORDER BY/LIMIT).")


        print("\nLatest 5 traffic conditions records from DuckDB:")
        # Using ORDER BY and LIMIT for reliability
        latest_traffic_df = conn.execute("""
            SELECT road_segment_id, location_lat, location_lon, jam_factor, speed, traffic_state, fetch_timestamp
            FROM traffic_conditions
            ORDER BY fetch_timestamp DESC
            LIMIT 5
        """).df()
        print(latest_traffic_df)
        if latest_traffic_df.empty:
             logging.info("No traffic records found in the last minute (using ORDER BY/LIMIT).")


        print("\nLatest weather info record from DuckDB:")
        # Using ORDER BY and LIMIT for reliability
        latest_weather_df = conn.execute("""
            SELECT location_lat, location_lon, temperature, description, fetch_timestamp
            FROM weather_info
            ORDER BY fetch_timestamp DESC
            LIMIT 1
        """).df()
        print(latest_weather_df)
        if latest_weather_df.empty:
             logging.info("No weather record found in the last minute (using ORDER BY/LIMIT).")


    except duckdb.Error as e:
        logging.error(f"Error querying DuckDB: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during DuckDB verification: {e}")


    # Example of fetching a single route summary (optional demonstration)
    # This is typically done on demand, not part of the batch 'fetch_all_real_time_data'
    route_origin_lat, route_origin_lon = test_lat, test_lon
    # Example destination (a different point near San Francisco)
    # Change this to a location near San Francisco for a valid route test
    route_dest_lat, route_dest_lon = 37.7749, -122.4500 # Example point west of SF center


    logging.info(f"Fetching example route summary from {route_origin_lat},{route_origin_lon} to {route_dest_lat},{route_dest_lon}...")
    route_summary = fetch_optimal_route_summary(route_origin_lat, route_origin_lon, route_dest_lat, route_dest_lon)

    if route_summary:
        logging.info(f"Example Route Summary: {route_summary.get('text')}")
        logging.info(f"Estimated Traffic Delay: {route_summary.get('traffic_delay_min', 0):.1f} minutes")
        # You could print the polyline here if you want to see it:
        # print(f"Polyline: {route_summary.get('polyline')}")
    else:
        logging.warning("Failed to fetch example route summary.")


    logging.info("--- Demonstration finished ---")

    # In a real application's lifecycle, especially in a server or long-running process,
    # you should explicitly close the database connection and cache when they are no longer needed.
    # For this simple script execution, they are usually closed automatically on process exit,
    # but explicit closing is good practice in more complex scenarios.
    # try:
    #     conn.close()
    #     logging.info("DuckDB connection closed.")
    # except Exception as e:
    #     logging.error(f"Error closing DuckDB connection: {e}")
    # try:
    #     cache.close()
    #     logging.info("Diskcache closed.")
    # except Exception as e:
    #      logging.error(f"Error closing Diskcache: {e}")
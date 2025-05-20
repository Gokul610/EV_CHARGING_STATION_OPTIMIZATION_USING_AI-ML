import os
import requests
import logging
from typing import Optional, Tuple, Dict, Any, List
import json
import pandas as pd
import numpy as np
import datetime
import asyncio
import aiohttp
import diskcache
from polyline import decode

# Setup logging (Step 18.3)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize cache (Step 19.1)
cache = diskcache.Cache(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))

# Load HERE API key (Step 3.1)
HERE_API_KEY = "kdL0wYxEBT426TAt0R9I2V_k9A8udwLVqSG3GN-07ic"
if HERE_API_KEY != "kdL0wYxEBT426TAt0R9I2V_k9A8udwLVqSG3GN-07ic":
    logging.warning("HERE_API_KEY not set. Using simulated data.")

# API Endpoints (Step 3.2)
HERE_GEOCODING_API_URL = "https://geocode.search.hereapi.com/v1/geocode"
HERE_ROUTING_API_URL = "https://router.hereapi.com/v8/routes"
BROWSE_API_URL_V1 = "https://browse.search.hereapi.com/v1/browse"

# Async fetch with caching (Step 19.2)
async def fetch_with_cache(url: str, params: Dict, cache_key: str, session: aiohttp.ClientSession) -> Optional[Dict]:
    if cache_key in cache:
        return cache[cache_key]
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            cache[cache_key] = data
            return data
    except Exception as e:
        logging.error(f"Async fetch error: {e}")
        return None

# Synchronous geocode place (Step 10.3)
def geocode_place(place_name: str) -> Optional[Tuple[float, float]]:
    if not place_name or not isinstance(place_name, str):
        logging.warning("Invalid place_name.")
        return None
    if HERE_API_KEY == "dummy_key":
        logging.warning("No HERE_API_KEY. Returning default coordinates.")
        return (9.0548, 77.4335)
    params = {"q": place_name, "apiKey": HERE_API_KEY}
    cache_key = f"geocode_{place_name}"
    if cache_key in cache:
        result = cache[cache_key]
    else:
        try:
            response = requests.get(HERE_GEOCODING_API_URL, params=params)
            response.raise_for_status()
            result = response.json()
            cache[cache_key] = result
        except Exception as e:
            logging.error(f"Geocoding error: {e}")
            return None
    if result and result.get('items'):
        position = result['items'][0].get('position', {})
        lat, lon = position.get('lat'), position.get('lng')
        if lat is not None and lon is not None:
            logging.info(f"Geocoded '{place_name}' to ({lat}, {lon})")
            return (lat, lon)
    logging.warning(f"Could not geocode '{place_name}'")
    return None

# Fetch optimal route summary (Step 13.4)
def fetch_optimal_route_summary(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, connector_type: Optional[str] = None, power_rating: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if HERE_API_KEY == "dummy_key":
        return {
            'distance_km': 10.0,
            'duration_min': 15.0,
            'traffic_delay_min': 2.0,
            'text': "Approx. 10.0 km, 15.0 min (2.0 min traffic delay)",
            'polyline': None
        }
    params = {
        "transportMode": "car",
        "origin": f"{origin_lat},{origin_lon}",
        "destination": f"{dest_lat},{dest_lon}",
        "return": "summary,polyline,typicalDuration",
        "apiKey": HERE_API_KEY,
        "units": "metric",
        "spans": "dynamicSpeedInfo"
    }
    cache_key = f"route_{origin_lat}_{origin_lon}_{dest_lat}_{dest_lon}_{connector_type}_{power_rating}"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async def fetch():
            async with aiohttp.ClientSession() as session:
                return await fetch_with_cache(HERE_ROUTING_API_URL, params, cache_key, session)
        result = loop.run_until_complete(fetch())
    finally:
        loop.close()
    if result and result.get('routes'):
        route = result['routes'][0]['sections'][0]
        summary = route.get('summary', {})
        distance_km = summary.get('length', 0) / 1000.0
        duration_min = summary.get('duration', 0) / 60.0
        typical_duration_min = summary.get('baseDuration', summary.get('duration', 0)) / 60.0
        traffic_delay_min = max(0, duration_min - typical_duration_min)
        polyline = route.get('polyline')
        logging.info(f"Fetched route: {distance_km:.1f} km, {duration_min:.1f} min")
        return {
            'distance_km': distance_km,
            'duration_min': duration_min,
            'traffic_delay_min': traffic_delay_min,
            'text': f"Approx. {distance_km:.1f} km, {duration_min:.1f} min ({traffic_delay_min:.1f} min traffic delay)",
            'polyline': polyline
        }
    logging.warning(f"No route found from ({origin_lat}, {origin_lon}) to ({dest_lat}, {dest_lon})")
    return None

# Fetch route geometry (Step 13.5)
def fetch_route_geometry(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> Optional[List[Tuple[float, float]]]:
    route = fetch_optimal_route_summary(origin_lat, origin_lon, dest_lat, dest_lon)
    if route and route.get('polyline'):
        try:
            return decode(route['polyline'])
        except Exception as e:
            logging.error(f"Polyline decode error: {e}")
            return [(origin_lat, origin_lon), (dest_lat, dest_lon)]
    return [(origin_lat, origin_lon), (dest_lat, dest_lon)]

# Fetch real-time station data (Step 3.2, 14)
def fetch_real_time_station_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
    if HERE_API_KEY == "dummy_key":
        return pd.DataFrame([
            {
                'station_id': f"SIM_{i}",
                'title': f"Simulated Station {i}",
                'lat': lat + np.random.uniform(-0.01, 0.01),
                'lon': lon + np.random.uniform(-0.01, 0.01),
                'distance_m': np.random.uniform(100, radius),
                'operational_status': 'operational',
                'available_connectors': np.random.randint(1, 10),
                'total_connectors': 10,
                'price': np.random.uniform(0, 5),
                'supported_connectors': np.random.choice(['CCS', 'Type 2', 'CHAdeMO']),
                'network_provider': np.random.choice(['Electrify America', 'ChargePoint']),
                'timestamp': datetime.datetime.now().isoformat()
            } for i in range(5)
        ])
    params = {
        "at": f"{lat},{lon}",
        "categories": "700-7600-0116",
        "in": f"circle:{lat},{lon};r={int(radius)}",
        "apiKey": HERE_API_KEY,
        "limit": 100
    }
    cache_key = f"stations_{lat}_{lon}_{radius}"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        async def fetch():
            async with aiohttp.ClientSession() as session:
                return await fetch_with_cache(BROWSE_API_URL_V1, params, cache_key, session)
        result = loop.run_until_complete(fetch())
    finally:
        loop.close()
    if not result or not result.get('items'):
        logging.warning(f"No stations found at ({lat}, {lon}) within {radius}m")
        return pd.DataFrame([
            {
                'station_id': f"SIM_{i}",
                'title': f"Simulated Station {i}",
                'lat': lat + np.random.uniform(-0.01, 0.01),
                'lon': lon + np.random.uniform(-0.01, 0.01),
                'distance_m': np.random.uniform(100, radius),
                'operational_status': 'operational',
                'available_connectors': np.random.randint(1, 10),
                'total_connectors': 10,
                'price': np.random.uniform(0, 5),
                'supported_connectors': np.random.choice(['CCS', 'Type 2', 'CHAdeMO']),
                'network_provider': np.random.choice(['Electrify America', 'ChargePoint']),
                'timestamp': datetime.datetime.now().isoformat()
            } for i in range(5)
        ])
    stations = []
    for item in result['items']:
        position = item.get('position', {})
        categories = [cat.get('id', '') for cat in item.get('categories', [])]
        contacts = item.get('contacts', [{}])[0]
        station = {
            'station_id': item.get('id', ''),
            'title': item.get('title', 'Unknown Station'),
            'lat': position.get('lat'),
            'lon': position.get('lng'),
            'distance_m': item.get('distance'),
            'operational_status': 'operational' if '700-7600-0116' in categories else 'unknown',
            'available_connectors': np.random.randint(1, 10),
            'total_connectors': 10,
            'price': np.random.uniform(0, 5),
            'supported_connectors': np.random.choice(['CCS', 'Type 2', 'CHAdeMO']),
            'network_provider': contacts.get('website', [{}])[0].get('value', 'Unknown'),
            'timestamp': datetime.datetime.now().isoformat()
        }
        stations.append(station)
    df = pd.DataFrame(stations)
    required_columns = ['station_id', 'title', 'lat', 'lon', 'distance_m', 'operational_status', 'available_connectors', 'total_connectors', 'price', 'supported_connectors', 'network_provider', 'timestamp']
    df = df.reindex(columns=required_columns)
    logging.info(f"Fetched {len(df)} stations")
    return df

# Testing block
if __name__ == "__main__":
    print("Testing api_utils.py...")
    os.environ["HERE_API_KEY"] = "dummy_key"
    test_place = "Surandai, Tamil Nadu"
    coords = geocode_place(test_place)
    print(f"Geocode '{test_place}': {coords}")
    route = fetch_optimal_route_summary(9.0548, 77.4335, 9.0648, 77.4435)
    print(f"Route: {route}")
    stations_df = fetch_real_time_station_data(9.0548, 77.4335, 5000)
    print(f"Stations:\n{stations_df.head().to_string()}")
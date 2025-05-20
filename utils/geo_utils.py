import os
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, box
import logging
from typing import Optional, Tuple
import duckdb
from shapely.validation import make_valid

# Setup logging (Step 18.3)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
VORONOI_OUTPUT_PATH = os.path.join(DATA_DIR, 'voronoi.geojson')
GRID_OUTPUT_PATH = os.path.join(DATA_DIR, 'grid.geojson')
GRID_ANALYSIS_OUTPUT_PATH = os.path.join(DATA_DIR, 'grid_analysis.geojson')

# Default fallback location (Chennai, India)
FALLBACK_LAT = 13.0827
FALLBACK_LON = 80.2707
RADIUS_KM = 50  # Analysis radius in kilometers

# Fallback for DuckDB connection
try:
    from utils.llm_data_prep import get_duckdb_connection
except ImportError:
    logging.warning("llm_data_prep.py not found. Using fallback DuckDB connection.")
    def get_duckdb_connection():
        logging.error("Dummy DuckDB connection. Data fetching will fail.")
        return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometers between two points using Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def fetch_station_locations_from_duckdb(user_lat: float, user_lon: float) -> gpd.GeoDataFrame:
    """
    Fetches station locations from DuckDB, filtering near user location.
    Falls back to simulated data around user location if no stations found.
    """
    # Use fallback location if user coordinates are invalid
    if not (-90 <= user_lat <= 90 and -180 <= user_lon <= 180):
        logging.warning(f"Invalid user location ({user_lat}, {user_lon}). Using fallback: Chennai.")
        user_lat, user_lon = FALLBACK_LAT, FALLBACK_LON

    conn = get_duckdb_connection()
    df = pd.DataFrame()
    if conn:
        try:
            query = """
            SELECT api_station_id AS station_id, title, lat, lon, operational_status,
                   network_provider, supported_connectors, available_connectors, total_connectors
            FROM station_status
            WHERE fetch_timestamp = (SELECT MAX(fetch_timestamp) FROM station_status)
            """
            df = conn.execute(query).fetchdf()
            logging.info(f"Fetched {len(df)} latest station records.")
        except Exception as e:
            logging.error(f"Error fetching station data: {e}", exc_info=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    if not df.empty:
        # Filter for stations near user location
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df = df.dropna(subset=['lat', 'lon']).copy()
        if not df.empty:
            df['distance_km'] = df.apply(
                lambda row: haversine_distance(row['lat'], row['lon'], user_lat, user_lon), axis=1
            )
            df = df[df['distance_km'] <= RADIUS_KM].copy()
            logging.info(f"Filtered {len(df)} stations within {RADIUS_KM}km of user ({user_lat}, {user_lon}).")

    if df.empty:
        logging.warning(f"No stations near user ({user_lat}, {user_lon}). Using fallback data.")
        df = pd.DataFrame([
            {
                'station_id': f'IND_{i}',
                'title': f'Indian Station {i}',
                'lat': user_lat + np.random.uniform(-0.05, 0.05),  # ~5km radius
                'lon': user_lon + np.random.uniform(-0.05, 0.05),
                'operational_status': 'operational',
                'network_provider': np.random.choice(['Tata Power', 'Ather Grid', 'Magenta']),
                'supported_connectors': np.random.choice(['CCS', 'Type 2', 'Bharat AC001']),
                'available_connectors': np.random.randint(1, 10),
                'total_connectors': 10
            } for i in range(50)
        ])

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon']).copy()
    if df.empty:
        logging.warning("No valid lat/lon data.")
        return gpd.GeoDataFrame()

    geometry = gpd.points_from_xy(df['lon'], df['lat'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    logging.info(f"Created GeoDataFrame with columns: {gdf.columns.tolist()}")
    return gdf

def encode_geohash(latitude: float, longitude: float, precision: int = 6) -> str:
    """Encodes latitude and longitude into a geohash."""
    try:
        import geohash2
        return geohash2.encode(latitude, longitude, precision)
    except ImportError:
        logging.warning("geohash2 not installed.")
        return ""
    except Exception as e:
        logging.error(f"Geohash encoding error: {e}")
        return ""

def apply_geohash_encoding_to_df(df: pd.DataFrame, precision: int = 6) -> pd.DataFrame:
    """Applies geohash encoding to a DataFrame."""
    if 'lat' in df.columns and 'lon' in df.columns:
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df_clean = df.dropna(subset=['lat', 'lon']).copy()
        if not df_clean.empty:
            df_clean['geohash'] = df_clean.apply(
                lambda row: encode_geohash(row['lat'], row['lon'], precision), axis=1
            )
            df = df.merge(df_clean[['geohash']], left_index=True, right_index=True, how='left')
        else:
            df['geohash'] = ""
    else:
        df['geohash'] = ""
    return df

def generate_voronoi_polygons(station_gdf: gpd.GeoDataFrame, bounds: Optional[Tuple[float, float, float, float]] = None) -> gpd.GeoDataFrame:
    """
    Generates Voronoi polygons with station_id and distance_m.
    Clips to bounds if provided.
    """
    if station_gdf.empty or len(station_gdf) < 4:
        logging.warning(f"Insufficient data for Voronoi: {len(station_gdf)} points.")
        return gpd.GeoDataFrame()

    points = station_gdf[['lon', 'lat']].drop_duplicates().values
    if len(points) < 4:
        logging.warning(f"Insufficient unique points for Voronoi: {len(points)}.")
        return gpd.GeoDataFrame()

    try:
        vor = Voronoi(points)
        polygons = []
        station_ids = []
        distances = []

        point_to_station = {tuple(pt): station_gdf.iloc[i]['station_id'] for i, pt in enumerate(station_gdf[['lon', 'lat']].values)}

        for i, region in enumerate(vor.point_region):
            region_indices = vor.regions[region]
            if -1 in region_indices or not region_indices:
                continue
            try:
                vertices = [vor.vertices[j] for j in region_indices]
                poly = Polygon(vertices)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid:
                    polygons.append(poly)
                    pt = vor.points[i]
                    station_id = point_to_station.get(tuple(pt), 'Unknown')
                    station_ids.append(station_id)
                    distances.append(poly.area * 1000000)
            except Exception as e:
                logging.warning(f"Error creating polygon for region {i}: {e}")

        if not polygons:
            logging.warning("No valid Voronoi polygons generated.")
            return gpd.GeoDataFrame()

        voronoi_gdf = gpd.GeoDataFrame(
            {'station_id': station_ids, 'distance_m': distances},
            geometry=polygons,
            crs=station_gdf.crs
        )
        logging.info(f"Generated {len(voronoi_gdf)} Voronoi polygons.")

        if bounds:
            bbox = box(*bounds)
            voronoi_gdf = gpd.clip(voronoi_gdf, bbox)
            voronoi_gdf = voronoi_gdf[voronoi_gdf.geometry.is_valid]
            logging.info(f"Clipped to bounds: {len(voronoi_gdf)} polygons.")

        logging.info(f"voronoi_gdf columns: {voronoi_gdf.columns.tolist()}")
        return voronoi_gdf

    except Exception as e:
        logging.error(f"Voronoi generation error: {e}", exc_info=True)
        return gpd.GeoDataFrame()

def create_grid_gdf(bounds: Tuple[float, float, float, float], cell_size_degrees: float = 0.005) -> gpd.GeoDataFrame:
    """Creates a square grid within bounds."""
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy or cell_size_degrees <= 0:
        logging.error(f"Invalid bounds {bounds} or cell_size {cell_size_degrees}.")
        return gpd.GeoDataFrame()

    x = np.arange(minx, maxx, cell_size_degrees)
    y = np.arange(miny, maxy, cell_size_degrees)
    polygons = [
        box(minx, miny, minx + cell_size_degrees, miny + cell_size_degrees)
        for miny in y for minx in x
    ]
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    logging.info(f"Created grid with {len(grid_gdf)} cells.")
    return grid_gdf

def analyze_grid_with_stations(grid_gdf: gpd.GeoDataFrame, station_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Counts stations per grid cell using spatial join."""
    if grid_gdf.empty or station_gdf.empty:
        logging.warning("Empty grid or stations.")
        if not grid_gdf.empty:
            grid_gdf['station_count'] = 0
            return grid_gdf
        return gpd.GeoDataFrame()

    try:
        if grid_gdf.crs != station_gdf.crs:
            station_gdf = station_gdf.to_crs(grid_gdf.crs)
        sjoin_gdf = gpd.sjoin(grid_gdf, station_gdf, how="left", predicate="intersects")
        grid_gdf['station_count'] = sjoin_gdf.groupby(sjoin_gdf.index).size().reindex(grid_gdf.index, fill_value=0).astype(int)
        grid_gdf = grid_gdf.drop(columns=['index_right'], errors='ignore')
        logging.info(f"Analyzed grid: {len(grid_gdf)} cells.")
        logging.info(f"grid_gdf columns: {grid_gdf.columns.tolist()}")
        return grid_gdf
    except Exception as e:
        logging.error(f"Grid analysis error: {e}", exc_info=True)
        if not grid_gdf.empty:
            grid_gdf['station_count'] = 0
            return grid_gdf
        return gpd.GeoDataFrame()

def run_geospatial_analysis(user_lat: float, user_lon: float):
    """Orchestrates geospatial analysis based on user location."""
    logging.info(f"Starting geospatial analysis for user location ({user_lat}, {user_lon})...")
    station_gdf = fetch_station_locations_from_duckdb(user_lat, user_lon)
    if station_gdf.empty:
        logging.error("No station data. Aborting.")
        return

    minx, miny, maxx, maxy = station_gdf.total_bounds
    buffer = 0.05
    bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    logging.info(f"Bounds: {bounds}")

    voronoi_gdf = generate_voronoi_polygons(station_gdf, bounds)
    if not voronoi_gdf.empty:
        try:
            voronoi_gdf.to_file(VORONOI_OUTPUT_PATH, driver='GeoJSON')
            logging.info(f"Saved Voronoi to {VORONOI_OUTPUT_PATH}")
        except Exception as e:
            logging.error(f"Error saving Voronoi: {e}")

    grid_gdf = create_grid_gdf(bounds, cell_size_degrees=0.005)
    if not grid_gdf.empty:
        try:
            grid_gdf.to_file(GRID_OUTPUT_PATH, driver='GeoJSON')
            logging.info(f"Saved grid to {GRID_OUTPUT_PATH}")
        except Exception as e:
            logging.error(f"Error saving grid: {e}")

    grid_analysis_gdf = analyze_grid_with_stations(grid_gdf, station_gdf)
    if not grid_analysis_gdf.empty:
        try:
            grid_analysis_gdf.to_file(GRID_ANALYSIS_OUTPUT_PATH, driver='GeoJSON')
            logging.info(f"Saved grid analysis to {GRID_ANALYSIS_OUTPUT_PATH}")
        except Exception as e:
            logging.error(f"Error saving grid analysis: {e}")

    logging.info("Geospatial analysis complete.")

if __name__ == "__main__":
    # Test with Chennai as default
    run_geospatial_analysis(FALLBACK_LAT, FALLBACK_LON)
    for path in [VORONOI_OUTPUT_PATH, GRID_OUTPUT_PATH, GRID_ANALYSIS_OUTPUT_PATH]:
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            logging.info(f"{path} columns: {gdf.columns.tolist()}")
            logging.info(f"{path} sample:\n{gdf.head().to_string()}")
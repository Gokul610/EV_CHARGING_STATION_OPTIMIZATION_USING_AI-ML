import pandas as pd
import sys
import os
import pandas

# Explicitly add project root to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.geo_utils import apply_geohash_encoding, generate_voronoi, create_grid

# Load sample data
df = pd.read_csv("data/charging_stations.csv").head(1000)

# Apply geohash encoding
df = apply_geohash_encoding(df)
print("✅ Geohash encoding applied successfully!")

# Generate Voronoi diagram
voronoi_gdf = generate_voronoi(df)
print("✅ Voronoi diagram generated!")

# Create grid analysis
grid_gdf = create_grid(df)
print("✅ Grid-based geospatial analysis completed!")

# Save results
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
voronoi_gdf.to_file(os.path.join(output_dir, "voronoi.geojson"), driver="GeoJSON")
grid_gdf.to_file("output/grid_analysis.geojson", driver="GeoJSON")

#D:\Desktop\ev_charging_optimization\utils\generate_data.py
import pandas as pd
import numpy as np
import random
import pygeohash

def generate_realistic_ev_data(num_stations=12000):
    """Generates a realistic EV charging station dataset with enhanced features for full-stack ML + geospatial + UI."""

    location_weights = {
        "Delhi": (28.6139, 77.2090, 0.15),
        "Mumbai": (19.0760, 72.8777, 0.12),
        "Bangalore": (12.9716, 77.5946, 0.14),
        "Chennai": (12.8827, 80.2707, 0.10),
        "Kolkata": (22.5726, 88.3639, 0.08),
        "Hyderabad": (17.3850, 78.4867, 0.08),
        "Lucknow": (26.8467, 80.9462, 0.07),
        "Ahmedabad": (23.0225, 72.5714, 0.06),
        "Pune": (18.5204, 73.8567, 0.07),
        "Coimbatore": (11.0168, 76.9558, 0.05),
    }

    facilities_list = ["Restaurant", "Shopping Mall", "Rest Area", "Hotel", "Gas Station"]
    ev_types = ["Tesla", "Generic", "Two-Wheelers", "All"]
    weather_conditions = ["Clear", "Rain", "Storm", "Cloudy"]
    congestion_levels = ["Low", "Medium", "High"]
    station_types = ["Public Fast", "Public Slow", "Private Fleet", "Highway Hub"]
    maintenance_states = ["Operational", "Under Maintenance", "Faulty"]
    payment_options_pool = ["UPI", "Card", "App Wallet", "RFID"]

    cities, locations, weights = zip(*[(city, (lat, lon), weight) for city, (lat, lon, weight) in location_weights.items()])
    weights = np.array(weights) / sum(weights)

    data = []
    for station_id in range(1, num_stations + 1):
        idx = np.random.choice(len(cities), p=weights)
        city, (lat, lon) = cities[idx], locations[idx]
        lat += random.uniform(-0.05, 0.05)
        lon += random.uniform(-0.05, 0.05)

        availability = random.randint(0, 15)
        price = round(random.uniform(5, 25), 2)
        base_demand = max(5.0, round(np.random.normal(40, 15), 1))
        connector_type = random.choice(["CCS2", "CHAdeMO", "Type 2 AC", "GB/T", "Tesla"])
        power_rating = random.choice([22, 50, 100, 150, 250])
        hour = random.randint(0, 23)
        demand = round(base_demand * 1.3 if 7 <= hour <= 10 or 18 <= hour <= 21 else base_demand, 1)
        wait_time = round(random.uniform(5, 45), 1) if availability == 0 else round(random.uniform(0, 10), 1)

        num_charging_points = random.randint(2, 20)
        charger_downtime = round(random.uniform(0, 20), 2)
        nearby_facility = random.choice(facilities_list)
        renewable_energy = round(random.uniform(0, 100), 1)
        ev_type_preference = random.choice(ev_types)

        # Added new essential fields
        weather = random.choice(weather_conditions)
        traffic_congestion = random.choice(congestion_levels)
        station_type = random.choice(station_types)
        maintenance_status = random.choice(maintenance_states)
        geohash = pygeohash.encode(lat, lon, precision=6)
        user_rating = round(random.uniform(3.0, 5.0), 1)
        num_reviews = random.randint(0, 500)
        avg_battery_kWh = random.choice([30, 50, 75])
        charging_time_est_min = round((avg_battery_kWh / power_rating) * 60, 1)
        payment_methods = ", ".join(random.sample(payment_options_pool, random.randint(1, 3)))

        data.append({
            "station_id": f"EVS{station_id}",
            "latitude": lat,
            "longitude": lon,
            "geohash": geohash,
            "city": city,
            "price": price,
            "availability": availability,
            "demand": demand,
            "connector_type": connector_type,
            "power_rating_kW": power_rating,
            "time_of_day": hour,
            "wait_time": wait_time,
            "num_charging_points": num_charging_points,
            "charger_downtime_%": charger_downtime,
            "nearby_facilities": nearby_facility,
            "renewable_energy_%": renewable_energy,
            "ev_type_preference": ev_type_preference,
            "weather_condition": weather,
            "traffic_congestion": traffic_congestion,
            "station_type": station_type,
            "maintenance_status": maintenance_status,
            "user_rating": user_rating,
            "num_reviews": num_reviews,
            "charging_time_est_min": charging_time_est_min,
            "payment_methods": payment_methods
        })

    df = pd.DataFrame(data)
    df.to_csv("data/charging_stations.csv", index=False)

    print("✅ Charging station dataset generated and saved successfully with all required project features!")

# Run the script
if __name__ == "__main__":
    generate_realistic_ev_data()

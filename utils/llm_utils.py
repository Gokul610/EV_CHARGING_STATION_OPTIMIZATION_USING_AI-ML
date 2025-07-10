import os
import requests
import json
import logging
from typing import Dict, Any
import pandas as pd
from utils.llm_data_prep import format_data_for_llm, fetch_latest_data_from_duckdb
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Google API key
GOOGLE_API_KEY = "AIzaSyBdW7BTy4iEfGEZCeaKVbz1lv9bkYH5H4k"
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not set.")
    raise ValueError("GOOGLE_API_KEY not set.")

# API configuration
MODEL_NAME = "gemini-2.0-flash"
API_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Prompt Templates
STATION_SUGGESTION_PROMPT_TEMPLATE = """
Analyze real-time EV charging station data and suggest the best station for a driver.
User: {user_context}
Stations: {station_data}
Traffic: {traffic_data}
Weather: {weather_data}
Suggest one station and explain why (max 80 words).
"""

DEMAND_PREDICTION_EXPLANATION_TEMPLATE = """
Generate a personalized EV charging station recommendation based on real-time data and demand prediction.
User: {user_context}
Station: {station_data}
Traffic: {traffic_data}
Weather: {weather_data}
Route: {route_summary}
Predicted Demand: {predicted_demand} EVs
Factors: {prediction_factors}
Recommend the station and explain why it's the best choice, considering user preferences, traffic, weather, route, and demand (max 100 words).
"""

GEOSPATIAL_INSIGHTS_PROMPT_TEMPLATE = """
Provide one actionable insight from EV charging station geospatial analysis.
Summary: {analysis_summary}
User: {user_context}
Insight (max 60 words): 
"""

ROUTE_ADVICE_PROMPT_TEMPLATE = """
Give one tip for an EV driver's route to a charging station.
Route: {route_summary}
Traffic: {traffic_conditions}
Station: {destination_station_details}
Demand: {predicted_demand_at_arrival}
User: {user_context}
Tip (max 60 words):
"""
def get_llm_suggestion(prompt_template: str, data_dict: Dict[str, Any], retries: int = 3) -> str:
    headers = {'Content-Type': 'application/json'}
    api_url = f"{API_BASE_URL}?key={GOOGLE_API_KEY}"
    prompt_text = prompt_template.format(**{k: str(data_dict.get(k, 'N/A')) for k in data_dict})
    logging.info(f"LLM prompt: {prompt_text[:200]}...")

    for attempt in range(retries):
        try:
            payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            if result.get('candidates'):
                text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logging.info(f"LLM response: {text[:100]}...")
                return text
            logging.warning(f"LLM response: {result}")
            return "No suggestion generated."
        except Exception as e:
            logging.error(f"LLM attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"LLM failed: {e}"

if __name__ == "__main__":
    print("Testing llm_utils.py...")
    test_data = {
        'user_context': "User in San Francisco",
        'station_data': "Station A: CCS, 5 available",
        'traffic_data': "Moderate traffic",
        'weather_data': "Clear, 20°C",
        'predicted_demand': 7,
        'prediction_factors': "Morning, CCS, Clear",
        'analysis_summary': "High density downtown",
        'route_summary': "10 km, 15 min",
        'traffic_conditions': "Moderate",
        'destination_station_details': "Station A: CCS",
        'predicted_demand_at_arrival': "Low"
    }
    for template in [STATION_SUGGESTION_PROMPT_TEMPLATE, DEMAND_PREDICTION_EXPLANATION_TEMPLATE, GEOSPATIAL_INSIGHTS_PROMPT_TEMPLATE, ROUTE_ADVICE_PROMPT_TEMPLATE]:
        print(f"\nTesting {template.splitlines()[0]}:")
        print(get_llm_suggestion(template, test_data))
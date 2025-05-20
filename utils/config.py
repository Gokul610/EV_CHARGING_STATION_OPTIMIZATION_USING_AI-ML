# utils/config.py
import os
HERE_API_KEY = "kdL0wYxEBT426TAt0R9I2V_k9A8udwLVqSG3GN-07ic"  # 🔹 Replace with your HERE Maps API Key
# File: D:\Desktop\ev_charging_optimization\utils\llm_utils.py

import requests
import json
import os
import logging
from typing import Dict, Any, List, Optional # Import Optional
# Assuming llm_data_prep is in the same utils directory
import os
import sys
# Explicitly add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from utils.llm_data_prep import format_data_for_llm, fetch_latest_data_from_duckdb # Import fetch_latest_data_from_duckdb

# Setup logging
# Change level to logging.DEBUG for more detailed output if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load the API key from environment variables (Still the recommended practice)
# Replace with the actual environment variable name you plan to use if different
GOOGLE_API_KEY = "AIzaSyBWOQVIPBH0nkg8wg5Smcjkkp4Y4cWF9L4"

if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY environment variable not set. LLM features will not work.")
    # In a production app, you might want to raise an exception or handle this more gracefully.
    # For now, we'll log an error and functions will return informative messages.

# API Endpoint and Model Name based on the user's provided structure
MODEL_NAME = "gemini-2.0-flash" # Using the model name specified by the user
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"


# --- Define Prompt Templates ---

# Template for a station suggestion based on real-time data
STATION_SUGGESTION_PROMPT_TEMPLATE = """
You are an AI assistant helping an EV driver find a charging station.
Analyze the following real-time information and provide a brief, helpful suggestion
for choosing a nearby charging station. Highlight key factors like availability,
traffic, price, and maybe the network or connector types available.

User's current location context: {user_context}

Nearby Charging Station Data:
{station_data}

Real-time Traffic Conditions:
{traffic_data}

Current Weather:
{weather_data}

Based on this information, which station seems like a good option and why?
Be concise (under 100 words).
"""

# Template for explaining a demand prediction
DEMAND_PREDICTION_EXPLANATION_TEMPLATE = """
A demand prediction model has predicted the charging demand at a station.
Explain this prediction in simple terms to an EV driver.

Prediction: {predicted_demand}
Factors influencing the prediction (including decoded categorical values):
{prediction_factors}

Explain what this prediction means (e.g., high, medium, low demand) and suggest
an action (e.g., charge now if low, expect delays if high).
Mention the most influential factors from the list provided.
Be concise.
"""

# Add templates for Geospatial Insights and Route Optimization Advice as needed


# --- LLM Interaction Functions using requests ---

def get_llm_suggestion(prompt_template: str, data_dict: Dict[str, Any]) -> str:
    """
    Generates a suggestion using the LLM via the requests library based on a template and dynamic data.
    """
    if not GOOGLE_API_KEY:
        return "LLM not available due to missing API key."

    # Headers for the HTTP request
    headers = {'Content-Type': 'application/json'}
    # URL includes the API key as a query parameter as shown in user's example
    api_url_with_key = f"{API_URL}?key={GOOGLE_API_KEY}"

    try:
        # Format the prompt with dynamic data from the dictionary
        # Use .get() with default empty strings to prevent KeyError if a key is missing
        # Ensure data is passed as strings as expected by the template
        prompt_text = prompt_template.format(
            user_context=data_dict.get('user_context', 'Unknown'),
            station_data="\n".join(data_dict.get('stations', ['No station data available.'])), # Provide a default if empty
            traffic_data="\n".join(data_dict.get('traffic', ['No traffic data available.'])), # Provide a default if empty
            weather_data="\n".join(data_dict.get('weather', ['No weather data available.'])), # Provide a default if empty
            predicted_demand=data_dict.get('predicted_demand', 'N/A'),
            prediction_factors=data_dict.get('prediction_factors', 'N/A Factors')
            # Add other placeholders as needed for other templates
        )

        logging.debug(f"Sending prompt to LLM:\n{prompt_text}")

        # Construct the request payload as shown in user's example
        request_payload = {
            "contents": [
                {
                    "parts": [{"text": prompt_text}]
                }
            ]
        }

        # Make the POST request to the Gemini API
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(request_payload))

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()

        # Extract and return the generated text, handling potential nested structure and missing keys
        if 'candidates' in result and result['candidates']:
            # Assuming the first candidate and its first part contain the text
            first_candidate = result['candidates'][0]
            if 'content' in first_candidate and 'parts' in first_candidate['content']:
                first_part = first_candidate['content']['parts'][0]
                if 'text' in first_part:
                    cleaned_text = first_part['text'].strip()
                    return cleaned_text
                else:
                    logging.warning("LLM response part found but no 'text' key.")
                    logging.debug(f"Full LLM response part: {first_part}")
                    return "Could not generate suggestion (no text content in part)."
            else:
                 logging.warning("LLM response candidate found but no 'content' or 'parts' key.")
                 logging.debug(f"Full LLM response candidate: {first_candidate}")
                 return "Could not generate suggestion (missing content structure)."
        elif 'promptFeedback' in result and result['promptFeedback']:
             # Handle cases where content generation was blocked
             logging.warning(f"LLM generation blocked: {result['promptFeedback']}")
             block_reason = result['promptFeedback'].get('blockReason', 'unknown')
             return f"Could not generate suggestion: Content blocked ({block_reason})."
        else:
            logging.warning("LLM response received but no 'candidates' or 'promptFeedback' found.")
            logging.debug(f"Full LLM response: {result}")
            return "Could not generate suggestion (unexpected response format)."

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error calling Gemini API: {e}")
        return f"Error generating suggestion: HTTP request failed ({e})"
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response from Gemini API: {e}")
        return f"Error generating suggestion: Invalid JSON response ({e})"
    except Exception as e:
        logging.error(f"An unexpected error occurred calling Gemini API: {e}")
        return f"Error generating suggestion: An unexpected error occurred ({e})"


# --- Example Usage (for testing when running the script directly) ---
if __name__ == "__main__":
    print("--- Testing utils/llm_utils.py (using requests) ---")

    # Note: For this test to work, you must have:
    # 1. Set the GOOGLE_API_KEY environment variable.
    # 2. Run fetch_real_time_data.py at least once to populate data/ev_charging.duckdb.

    if not GOOGLE_API_KEY:
        print("\nSkipping LLM function tests because GOOGLE_API_KEY environment variable is not set.")
    else:
        # --- Simulate fetching and formatting data (using functions from Step 5) ---
        test_lat = 37.7749 # Example location (San Francisco)
        test_lon = -122.4194
        test_radius = 5000 # meters

        print(f"\nFetching sample data for LLM from DuckDB ({test_lat},{test_lon} radius {test_radius}m)...")
        station_data_df = fetch_latest_data_from_duckdb('station_status', lat=test_lat, lon=test_lon, radius=test_radius)
        traffic_data_df = fetch_latest_data_from_duckdb('traffic_conditions', lat=test_lat, lon=test_lon, radius=test_radius)
        weather_data_df = fetch_latest_data_from_duckdb('weather_info', lat=test_lat, lon=test_lon, radius=test_radius)

        llm_formatted_data = format_data_for_llm(station_data_df, traffic_data_df, weather_data_df)

        # Add dummy prediction data and user context for the demand explanation prompt
        llm_formatted_data['predicted_demand'] = 7 # Example integer prediction
        # Example decoded factors - derived from your Step 4 encoding/decoding logic
        llm_formatted_data['prediction_factors'] = "Time of day (Morning Peak), Connector Type (CCS), Network Provider (Electrify America), Weather (Clear)"
        llm_formatted_data['user_context'] = "The user is planning to charge within the next hour and prefers fast chargers." # Example user context

        # --- Test the LLM suggestion function ---
        print("\nTesting LLM Station Suggestion:")
        station_suggestion = get_llm_suggestion(STATION_SUGGESTION_PROMPT_TEMPLATE, llm_formatted_data)
        print("\n--- LLM Station Suggestion ---")
        print(station_suggestion)
        print("-" * 30)

        print("\nTesting LLM Demand Prediction Explanation:")
        demand_explanation = get_llm_suggestion(DEMAND_PREDICTION_EXPLANATION_TEMPLATE, llm_formatted_data)
        print("\n--- LLM Demand Explanation ---")
        print(demand_explanation)
        print("-" * 30)

        # Add tests for other LLM prompt templates as you create them

    print("\n--- Testing Complete ---")
import requests
import json

# Replace with your actual API key
GEMINI_API_KEY = "AIzaSyBWOQVIPBH0nkg8wg5Smcjkkp4Y4cWF9L4"
model_name = "gemini-2.0-flash"
api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
headers = {'Content-Type': 'application/json'}
data = {
  "contents": [
    {
      "parts": [{"text": "tell about muthuramalingathevar"}]
    }
  ]
}

try:
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for HTTP errors
    result = response.json()
    # Process the response here
    if 'candidates' in result and result['candidates']:
        for candidate in result['candidates']:
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'text' in part:
                        print(part['text'])
    else:
        print("No response text found.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON response: {e}")
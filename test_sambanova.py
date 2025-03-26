import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("SAMBANOVA_API_KEY")
if not api_key:
    raise ValueError("SAMBANOVA_API_KEY environment variable not set")

# SambaNova API settings
api_url = "https://api.sambanova.ai/v1/chat/completions"
model = "DeepSeek-R1"  # The model you're using for complex queries

# Sample messages
messages = [
    {"role": "system", "content": "You are a helpful assistant who answers database queries."},
    {
        "role": "user",
        "content": "Analyze the profit trends across different customer segments over the last 12 months and identify the top performing cohorts.",
    },
]

# API request payload
payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512}

# Headers
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Make the API request
print(f"Testing SambaNova API with model: {model}")
print(f"Request payload: {payload}")
print("Sending request...")

try:
    response = requests.post(api_url, headers=headers, json=payload)

    # Check response
    if response.status_code == 200:
        result = response.json()
        print("\nAPI call successful!")
        print("\nResponse:")
        print(f"Status code: {response.status_code}")
        print(f"Generated text: {result['choices'][0]['message']['content'][:200]}...")
    else:
        print(f"\nAPI call failed with status code: {response.status_code}")
        print(f"Error message: {response.text}")
except Exception as e:
    print(f"\nException occurred: {str(e)}")

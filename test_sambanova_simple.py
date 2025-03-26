import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime

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
    {"role": "system", "content": "You are a helpful SQL assistant who answers database queries."},
    {
        "role": "user",
        "content": "Analyze the customer purchase patterns based on demographic data.",
    },
]

# API request payload
payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512}

# Headers
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Create output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"sambanova_test_{timestamp}.json"

print("=" * 60)
print("SAMBANOVA API TEST")
print("=" * 60)
print(f"Testing model: {model}")
print(f"Query: {messages[-1]['content']}")
print("-" * 60)
print("Sending request to SambaNova API...")

try:
    # Make the API request
    response = requests.post(api_url, headers=headers, json=payload)

    print(f"Response status code: {response.status_code}")

    # Check response
    if response.status_code == 200:
        result = response.json()
        print("\nAPI CALL SUCCESSFUL!")

        # Get the response content
        response_content = result["choices"][0]["message"]["content"]

        print("\nRESPONSE PREVIEW (first 300 chars):")
        print("-" * 60)
        print(response_content[:300])
        if len(response_content) > 300:
            print("...")

        # Save full response to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nFull response saved to: {output_file}")
    else:
        print("\nAPI CALL FAILED!")
        print("\nERROR DETAILS:")
        print("-" * 60)
        try:
            error_data = response.json()
            print(json.dumps(error_data, indent=2))
        except json.JSONDecodeError:
            print(response.text)
except Exception as e:
    print(f"Error: {str(e)}")

print("=" * 60)

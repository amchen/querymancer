import os
import json
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

# Sample messages - using a complex analytical query that would be routed to SambaNova
messages = [
    {"role": "system", "content": "You are a helpful assistant who answers database queries."},
    {
        "role": "user",
        "content": "How many tables are in the database and what's the schema of each?",
    },
]

# API request payload
payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512}

# Headers
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

print("=" * 80)
print("SAMBANOVA API TEST")
print("=" * 80)
print(f"Testing model: {model}")
print(f"Query: {messages[-1]['content']}")
print("-" * 80)

try:
    # Make the API request
    print("Sending request to SambaNova API...")
    response = requests.post(api_url, headers=headers, json=payload)

    print(f"Response status code: {response.status_code}")

    # Check response
    if response.status_code == 200:
        result = response.json()
        print("\n✅ API CALL SUCCESSFUL!")

        # Get the response content
        response_content = result["choices"][0]["message"]["content"]

        print("\nRESPONSE CONTENT:")
        print("-" * 80)
        print(response_content[:500])  # Print first 500 chars

        if len(response_content) > 500:
            print("...")
            print(f"[Response truncated - total length: {len(response_content)} characters]")
    else:
        print("\n❌ API CALL FAILED!")
        print("\nERROR DETAILS:")
        print("-" * 80)
        try:
            error_data = response.json()
            print(json.dumps(error_data, indent=2))
        except json.JSONDecodeError:
            print(response.text)
except Exception as e:
    print(f"Error: {str(e)}")

print("=" * 80)

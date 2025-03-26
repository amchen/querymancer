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

# Sample messages - using a complex analytical query that would be routed to SambaNova
messages = [
    {"role": "system", "content": "You are a helpful SQL assistant who answers database queries."},
    {
        "role": "user",
        "content": "Analyze the customer purchase patterns based on demographic data and create a segmentation report.",
    },
]

# API request payload
payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512}

# Headers
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Create output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"sambanova_test_result_{timestamp}.txt"

# Write to both console and file
with open(output_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("SAMBANOVA API TEST\n")
    f.write("=" * 80 + "\n")
    f.write(f"Testing model: {model}\n")
    f.write(f"Query: {messages[-1]['content']}\n")
    f.write("-" * 80 + "\n")

    print("=" * 80)
    print("SAMBANOVA API TEST")
    print("=" * 80)
    print(f"Testing model: {model}")
    print(f"Query: {messages[-1]['content']}")
    print("-" * 80)

    try:
        # Make the API request
        print("Sending request to SambaNova API...")
        f.write("Sending request to SambaNova API...\n")

        response = requests.post(api_url, headers=headers, json=payload)

        print(f"Response status code: {response.status_code}")
        f.write(f"Response status code: {response.status_code}\n")

        # Check response
        if response.status_code == 200:
            result = response.json()

            print("\n✅ API CALL SUCCESSFUL!")
            f.write("\n✅ API CALL SUCCESSFUL!\n")

            # Get the response content
            response_content = result["choices"][0]["message"]["content"]

            print("\nRESPONSE SAVED TO FILE:")
            print(f"Check {output_file} for the complete response")

            f.write("\nRESPONSE CONTENT:\n")
            f.write("-" * 80 + "\n")
            f.write(response_content)
            f.write("\n" + "-" * 80 + "\n")
            f.write("\nFull response details:\n")
            f.write(json.dumps(result, indent=2))
        else:
            print("\n❌ API CALL FAILED!")
            f.write("\n❌ API CALL FAILED!\n")

            print("\nERROR DETAILS:")
            f.write("\nERROR DETAILS:\n")
            f.write("-" * 80 + "\n")

            try:
                error_data = response.json()
                error_json = json.dumps(error_data, indent=2)
                print(error_json)
                f.write(error_json + "\n")
            except json.JSONDecodeError:
                print(response.text)
                f.write(response.text + "\n")
    except Exception as e:
        error_msg = f"\n❌ EXCEPTION OCCURRED: {str(e)}"
        print(error_msg)
        f.write(error_msg + "\n")

    print("=" * 80)
    print(f"Complete results saved to: {output_file}")
    f.write("=" * 80)
